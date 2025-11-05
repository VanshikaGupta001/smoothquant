import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
import tqdm
from datasets import load_dataset
import argparse
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
# --- End of new imports ---


parser = argparse.ArgumentParser()
# parser.add_argument("--alpha", type=float, default=0.5) # No longer needed
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="act_scales/llama-2-7b-svd.pt", # Recommend renaming your file
    help="Path to the precomputed SVD '{S_scaling, S_scaling_inv}' factors."
)
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--smooth", action="store_true", help="Apply SVD Whitening (Steps 3 & 4)")
parser.add_argument("--quantize", action="store_true")


args = parser.parse_args()
model_path = args.model_path
act_scales_path = args.act_scales_path
n_samples = args.n_samples


# --- CODE FOR STEP 3: ACTIVATION WHITENING ---

class ActivationWhitener(nn.Module):
    """
    A simple nn.Module to apply the precomputed S_scaling_inv matrix.
    This will be inserted after each LayerNorm.
    """
    def __init__(self, S_scaling_inv):
        super().__init__()
        # Register S_scaling_inv as a persistent buffer
        self.register_buffer("S_inv", S_scaling_inv.contiguous())
        
    @torch.no_grad()
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        # S_inv shape: (hidden_dim, hidden_dim)
        
        # Apply the whitening transformation: X_hat = X * S_inv
        orig_dtype = x.dtype
        x_whiten = (x.float() @ self.S_inv.float()).to(orig_dtype)
        return x_whiten

@torch.no_grad()
def apply_whitening_to_model(model, whitening_factors):
    """
    This function performs Step 3: X_hat = X * S_inv
    It uses the pre-computed 'S_scaling_inv' from the whitening_factors dictionary.
    """
    print("Applying 'S_scaling_inv' to activations (Step 3) by patching model...")
    
    print("Loading pre-computed S_scaling_inv factors...")

    print("Patching model forward pass...")    
    for name, module in model.named_modules():
        
        # Helper to get the pre-computed S_scaling_inv
        def get_s_inv(key):
            if key not in whitening_factors:
                print(f"Warning: Could not find key {key} in whitening factors. Skipping patch.")
                return None
            # Load the S_scaling_inv field from the loaded dictionary
            return whitening_factors[key]['S_scaling_inv'].to(model.device)

        # --- Handle OPT ---
        if isinstance(module, OPTDecoderLayer):
            old_ln_attn = module.self_attn_layer_norm
            S_inv_qkv = get_s_inv(name + ".self_attn.q_proj")
            if S_inv_qkv is None: continue
            module.self_attn_layer_norm = nn.Sequential(
                old_ln_attn,
                ActivationWhitener(S_inv_qkv)
            ).to(model.device)
            
            old_ln_ffn = module.final_layer_norm
            S_inv_fc1 = get_s_inv(name + ".fc1")
            if S_inv_fc1 is None: continue
            module.final_layer_norm = nn.Sequential(
                old_ln_ffn,
                ActivationWhitener(S_inv_fc1)
            ).to(model.device)
        
        # --- Handle Llama/Mistral ---
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            old_ln_attn = module.input_layernorm
            S_inv_qkv = get_s_inv(name + ".self_attn.q_proj")
            if S_inv_qkv is None: continue
            module.input_layernorm = nn.Sequential(
                old_ln_attn,
                ActivationWhitener(S_inv_qkv)
            ).to(model.device)
            
            old_ln_ffn = module.post_attention_layernorm
            S_inv_ffn = get_s_inv(name + ".mlp.gate_proj")
            if S_inv_ffn is None: continue
            module.post_attention_layernorm = nn.Sequential(
                old_ln_ffn,
                ActivationWhitener(S_inv_ffn)
            ).to(model.device)

        # --- Handle Mixtral ---
        elif isinstance(module, MixtralDecoderLayer):
            old_ln_attn = module.input_layernorm
            S_inv_qkv = get_s_inv(name + ".self_attn.q_proj")
            if S_inv_qkv is None: continue
            module.input_layernorm = nn.Sequential(
                old_ln_attn,
                ActivationWhitener(S_inv_qkv)
            ).to(model.device)
            
            old_ln_ffn = module.post_attention_layernorm
            S_inv_ffn = get_s_inv(name + ".block_sparse_moe.gate")
            if S_inv_ffn is None: continue
            module.post_attention_layernorm = nn.Sequential(
                old_ln_ffn,
                ActivationWhitener(S_inv_ffn)
            ).to(model.device)

        # --- Handle Falcon ---
        elif isinstance(module, FalconDecoderLayer):
            S_inv_qkv = get_s_inv(name + ".self_attention.query_key_value")
            S_inv_fc1 = get_s_inv(name + ".mlp.dense_h_to_4h")
            if S_inv_qkv is None or S_inv_fc1 is None: continue
            
            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                old_ln_attn = module.input_layernorm
                module.input_layernorm = nn.Sequential(
                    old_ln_attn,
                    ActivationWhitener(S_inv_qkv) # S_inv_qkv used for both
                ).to(model.device)
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                
                # Patch attn_ln
                new_attn_ln = nn.Sequential(attn_ln, ActivationWhitener(S_inv_qkv)).to(model.device)
                if module.config.new_decoder_architecture:
                    module.ln_attn = new_attn_ln
                else:
                    module.input_layernorm = new_attn_ln

                # Patch ffn_ln
                new_ffn_ln = nn.Sequential(ffn_ln, ActivationWhitener(S_inv_fc1)).to(model.device)
                if module.config.new_decoder_architecture:
                    module.ln_mlp = new_ffn_ln
                else:
                    module.post_attention_layernorm = new_ffn_ln

    print("Model patching (Step 3) complete.")

# --- END OF NEW CODE ---


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples_eff = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        
        if n_samples_eff == 0:
             # Handle small datasets for testing
            n_samples_eff = 1
            seq_len = self.dataset.size(1)
            if seq_len == 0:
                print("Error: No data in dataset.")
                return torch.tensor(float('nan'))
        else:
            seq_len = 2048
            
        total_tokens_evaluated = 0

        for i in tqdm.tqdm(range(n_samples_eff), desc="Evaluating..."):
            start_idx = i * seq_len
            end_idx = (i + 1) * seq_len
            
            # Ensure batch indices are within bounds
            if end_idx > self.dataset.size(1):
                end_idx = self.dataset.size(1)
                if start_idx >= end_idx:
                    break # Stop if we're out of data

            batch = self.dataset[:, start_idx : end_idx].to(model.device)
            if batch.size(1) < 2:
                continue # Skip batches that are too short

            with torch.no_grad():
                lm_logits = model(batch).logits
            
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:]
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            
            # Calculate total negative log-likelihood for the sequence
            # loss is the mean NLL, so multiply by sequence length
            current_tokens = shift_labels.size(1)
            total_tokens_evaluated += current_tokens
            neg_log_likelihood = loss.float() * current_tokens
            nlls.append(neg_log_likelihood)

        if not nlls:
            print("Warning: No samples were evaluated.")
            return torch.tensor(float('nan'))
             
        # Calculate perplexity: exp(total_nll / total_tokens)
        total_nll = torch.stack(nlls).sum()
        return torch.exp(total_nll / total_tokens_evaluated)


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples)

print(f"Loading model: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
print("Model loaded.")

if args.smooth:
    print(f"Loading SVD whitening factors from: {act_scales_path}")
    # 'whitening_factors' is now a dict: {name: {'S_scaling': S_tensor, 'S_scaling_inv': S_inv_tensor}}
    whitening_factors = torch.load(act_scales_path, map_location="cpu") 
    
    # Extract just the 'S_scaling' matrices for smooth_lm (Step 4)
    s_factors_for_smoothing = {name: factors['S_scaling'] for name, factors in whitening_factors.items()}
    
    # Step 4: Whiten Weights (W_hat = S_scaling * W)
    print("Applying Step 4: Weight Whitening (W_hat = S_scaling * W)...")
    smooth_lm(model, s_factors_for_smoothing) # Pass the dict of S_scaling
    
    # Step 3: Whiten Activations (X_hat = X * S_inv)
    # This function will now use the pre-computed S_scaling_inv
    print("Applying Step 3: Activation Whitening (X_hat = X * S_inv)...")
    apply_whitening_to_model(model, whitening_factors) 
else:
    print("Skipping whitening.")

if args.quantize:
    print("Applying fake quantization...")
    model = quantize_model(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=True,
    )
    print("Quantization complete.")
else:
    print("Skipping quantization.")

ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl.item()}")

