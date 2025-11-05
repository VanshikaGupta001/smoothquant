import torch
import torch.nn as nn

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


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales):
    """
    Applies the SVD S_scaling factor (passed as act_scales) to the weights.
    This performs Step 4: W_hat = S_scaling * W
    
    The LayerNorm (ln) argument is IGNORED.
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    # 'act_scales' is now the SVD 'S_scaling' matrix
    S_scaling = act_scales 
        
    # Check assertions
    assert isinstance(ln, (nn.LayerNorm, LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert S_scaling.shape[0] == S_scaling.shape[1] == fc.in_features, \
            f"Mismatched dimensions: S_scaling has shape {S_scaling.shape}, but fc.in_features is {fc.in_features}"

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    
    # We need S_scaling^T, on the correct device and dtype
    S_scaling_T = S_scaling.to(device=device, dtype=dtype).T

    for fc in fcs:
        # W_hat^T = (S_scaling * W)^T = W^T * S_scaling^T
        fc.weight.data = fc.weight.data @ S_scaling_T


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales):
    """
    Applies the SVD S_scaling factor (passed as act_scales) to the weights.
    This performs Step 4: W_hat = S_scaling * W
    
    The RMSNorm (ln) argument is IGNORED.
    """
    # This function is now identical to smooth_ln_fcs, but kept for compatibility
    smooth_ln_fcs(ln, fcs, act_scales)


@torch.no_grad()
def smooth_lm(model, scales):
    """
    Modifies the model's weights in-place using the precomputed SVD S_scaling factors,
    which are passed in the 'scales' dictionary.
    
    This function ONLY performs Step 4 (W_hat = S_scaling * W).
    """
    # 'scales' is now the dictionary of S_scaling factors
    svd_factors = scales

    print("Applying 'S_scaling' matrix to model weights (Step 4)...")
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            S_qkv = svd_factors[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, S_qkv) # Pass S_scaling as act_scales

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            S_fc1 = svd_factors[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, S_fc1) # Pass S_scaling as act_scales

        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            S_qkv = svd_factors[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, S_qkv)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            S_fc1 = svd_factors[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, S_fc1)
            
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            S_qkv = svd_factors[name + ".self_attention.query_key_value"]
            
            fc1 = module.mlp.dense_h_to_4h
            S_fc1 = svd_factors[name + ".mlp.dense_h_to_4h"]

            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], S_qkv)
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
                smooth_ln_fcs(attn_ln, qkv, S_qkv)
                smooth_ln_fcs(ffn_ln, fc1, S_fc1)
                
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            S_qkv = svd_factors[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, S_qkv)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            S_ffn = svd_factors[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, S_ffn)
            
        elif isinstance(module, MixtralDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            S_qkv = svd_factors[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, S_qkv)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            
            S_ffn = svd_factors[name + ".block_sparse_moe.gate"]
            
            fcs_moe = [module.block_sparse_moe.gate]
            for expert in module.block_sparse_moe.experts:
                fcs_moe.append(expert.w1)
                fcs_moe.append(expert.w3)

            smooth_ln_fcs_llama_like(ffn_ln, fcs_moe, S_ffn)

    
    print("Weight whitening (Step 4) complete.")

