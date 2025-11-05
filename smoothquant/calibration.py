import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def get_act_svd(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    """
    Computes the PCA Whitening factors (S_scaling, S_scaling_inv)
    using SVD of the activation covariance matrix (X^T * X).
    
    This version is memory-optimized to delete covariance matrices after processing.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # This dictionary will store the covariance matrices, C = X^T * X
    act_covariances = {}

    def accumulate_covariance(name, tensor):
        """
        Calculates the covariance matrix for the current batch (X^T * X)
        and adds it to the running sum.
        """
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach().to(torch.float32) # Shape (m, d)
        current_cov = tensor.T @ tensor

        if name not in act_covariances:
            act_covariances[name] = torch.zeros_like(current_cov)

        act_covariances[name] += current_cov

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        accumulate_covariance(name, x)

    hooks = []
    print("Registering hooks to collect activations...")
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    print("Collecting activations and computing covariance...")
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    print("\nComputing SVD, S_scaling, and S_scaling_inv...")
    # Store factors on CPU to save GPU RAM
    whitening_factors = {} 
    
    # --- MEMORY OPTIMIZATION ---
    # Iterate over a .keys() list so we can safely delete from the dict
    layer_names = list(act_covariances.keys())
    
    for name in tqdm(layer_names):
        # Get the covariance matrix C = H = X^T * X
        C = act_covariances[name]
        d = C.shape[0]
        
        # Add regularization for stability (handles non-positive definite cases)
        # A smaller regularization is needed for SVD than for Cholesky
        reg = 1e-5 * torch.eye(d, device=C.device)
        C_reg = C.float() + reg
        
        S_scaling = None
        S_scaling_inv = None

        try:
            # Perform SVD: H = U * Sigma * U^T
            # .svd() returns U, S (vector), Vh (V transpose)
            # For symmetric matrix C, U == Vh.T
            U, Sigma_diag, _ = torch.linalg.svd(C_reg)

            # Clamp small singular values for numerical stability
            Sigma_diag = Sigma_diag.clamp(min=1e-6)

            # Create diagonal matrices
            # sqrt(Sigma)
            sqrt_Sigma = torch.diag(torch.sqrt(Sigma_diag))
            # Sigma^(-1/2)
            Sigma_inv_sqrt = torch.diag(1.0 / torch.sqrt(Sigma_diag))

            # S_scaling = sqrt(Sigma) * U^T
            S_scaling = (sqrt_Sigma @ U.T).cpu()
            # S_scaling_inv = U * Sigma^(-1/2)
            S_scaling_inv = (U @ Sigma_inv_sqrt).cpu()
            
        except torch.linalg.LinAlgError as e:
            print(f"FATAL: SVD failed for {name}: {e}")
            # Fallback to identity
            S_scaling = torch.eye(d)
            S_scaling_inv = torch.eye(d)

        # Store factors on CPU
        whitening_factors[name] = {
            'S_scaling': S_scaling,
            'S_scaling_inv': S_scaling_inv
        }
        
        # --- MEMORY OPTIMIZATION ---
        # Delete the covariance matrix from GPU memory immediately
        del act_covariances[name]
        del C
    
    # Clear any remaining GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # --- END OF OPTIMIZATION ---

    # Renamed from cholesky_factors
    return whitening_factors


# --- This function is unchanged as it serves a different purpose ---

@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
):
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["input"] / 127
        )
        scale_dict["q_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["output"] / 127
        )
        scale_dict["k_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"]["output"] / 127
        )
        scale_dict["v_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"]["output"] / 127
        )
        scale_dict["out_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"]["input"] / 127
        )
        scale_dict["fc1_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.fc1"]["input"] / 127
        )
        scale_dict["fc2_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.fc2"]["input"] / 127
        )
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict

#----------------------------------------------------------------
#memory efficient can work on dgx 32 gb gpu-- basically offloads covariance to cpu

# import torch
# import torch.nn as nn

# from datasets import load_dataset
# import functools
# from collections import defaultdict

# from functools import partial
# import numpy as np
# from tqdm import tqdm


# @torch.no_grad()
# def get_act_svd(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
#     """
#     Computes the SVD-based scaling factors (PCA Whitening)
#     by storing the large covariance matrices on the CPU to save VRAM.
#     """
#     model.eval()
#     device = next(model.parameters()).device
    
#     # This dictionary will store the covariance matrices, C = X^T * X
#     # MODIFICATION: We will store this on the CPU
#     act_covariances = {}

#     def accumulate_covariance(name, tensor):
#         """
#         Calculates the covariance matrix for the current batch (X^T * X)
#         and adds it to the running sum on the CPU.
#         """
#         hidden_dim = tensor.shape[-1]
#         tensor = tensor.view(-1, hidden_dim).detach().to(torch.float32) # Shape (m, d)
        
#         # Calculate X^T * X for the current batch on GPU (fast)
#         current_cov = tensor.T @ tensor

#         if name not in act_covariances:
#             # MODIFICATION: Initialize covariance matrix on CPU
#             act_covariances[name] = torch.zeros_like(current_cov, device='cpu')

#         # MODIFICATION: Move result to CPU and accumulate
#         act_covariances[name] += current_cov.cpu()

#     def stat_input_hook(m, x, y, name):
#         if isinstance(x, tuple):
#             x = x[0]
#         accumulate_covariance(name, x)

#     hooks = []
#     print("Registering hooks to collect activations...")
#     for name, m in model.named_modules():
#         if isinstance(m, nn.Linear):
#             hooks.append(
#                 m.register_forward_hook(functools.partial(stat_input_hook, name=name))
#             )

#     dataset = load_dataset("json", data_files=dataset_path, split="train")
#     dataset = dataset.shuffle(seed=42)

#     print("Collecting activations and computing covariance...")
#     for i in tqdm(range(num_samples)):
#         input_ids = tokenizer(
#             dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
#         ).input_ids.to(device)
#         model(input_ids)

#     for h in hooks:
#         h.remove()

#     print("\nComputing SVD decomposition and S_scaling factors...")
#     svd_factors = {}
    
#     # MODIFICATION: We clear the covariance dict as we go to save CPU RAM
#     covariance_keys = list(act_covariances.keys())
    
#     for name in tqdm(covariance_keys):
#         # H is on the CPU
#         H = act_covariances[name]
#         d = H.shape[0]
#         reg = 1e-1 * torch.eye(d) # Regularization on CPU
        
#         try:
#             # MODIFICATION: Move H to GPU for SVD, perform SVD
#             H_gpu = H.float().to(device)
#             U, S_diag, V = torch.linalg.svd(H_gpu)
#             del H_gpu # Free VRAM
            
#         except torch.linalg.LinAlgError:
#             print(f"Warning: SVD failed for {name}. Applying regularization.")
#             try:
#                 # Try again with regularization
#                 H_gpu = (H.float() + reg).to(device)
#                 U, S_diag, V = torch.linalg.svd(H_gpu)
#                 del H_gpu # Free VRAM
#             except torch.linalg.LinAlgError as e:
#                 print(f"FATAL: SVD failed for {name} even with regularization: {e}")
#                 U = torch.eye(d, device=device)
#                 S_diag = torch.ones(d, device=device)

#         # Calculate S_scaling = sqrt(Sigma) * U^T on GPU
#         S_diag_sqrt = torch.sqrt(S_diag).diag()
#         S_scaling = (S_diag_sqrt @ U.T)
        
#         # Calculate S_scaling_inv = U * 1/sqrt(Sigma) on GPU
#         S_diag_inv_sqrt = torch.diag(1.0 / torch.sqrt(S_diag))
#         S_scaling_inv = (U @ S_diag_inv_sqrt)

#         # MODIFICATION: Store the final factors on CPU
#         svd_factors[name] = {
#             'S_scaling': S_scaling.cpu(),
#             'S_scaling_inv': S_scaling_inv.cpu(),
#         }
        
#         # Free CPU and GPU memory
#         del act_covariances[name]
#         del U, S_diag, V, S_diag_sqrt, S_scaling, S_diag_inv_sqrt, S_scaling_inv
#         torch.cuda.empty_cache()

#     return svd_factors


# # --- This function is unchanged as it serves a different purpose ---
# # --- But we'll apply the same CPU offloading just in case ---

# @torch.no_grad()
# def get_static_decoder_layer_scales(
#     model,
#     tokenizer,
#     dataset_path,
#     num_samples=512,
#     seq_len=512,
# ):
#     model.eval()
#     device = next(model.parameters()).device

#     # MODIFICATION: Store this on the CPU
#     act_dict = defaultdict(lambda: defaultdict(float))

#     def stat_io_hook(m, x, y, name):
#         if isinstance(x, tuple):
#             x = x[0]
        
#         # Get max value on GPU, then move scalar to CPU
#         current_input_max = x.detach().abs().max().item()
#         if name not in act_dict or "input" not in act_dict[name]:
#             act_dict[name]["input"] = current_input_max
#         else:
#             act_dict[name]["input"] = max(
#                 act_dict[name]["input"], current_input_max
#             )
            
#         if isinstance(y, tuple):
#             y = y[0]
        
#         # Get max value on GPU, then move scalar to CPU
#         current_output_max = y.detach().abs().max().item()
#         if name not in act_dict or "output" not in act_dict[name]:
#             act_dict[name]["output"] = current_output_max
#         else:
#             act_dict[name]["output"] = max(
#                 act_dict[name]["output"], current_output_max
#             )

#     hooks = []
#     for name, m in model.named_modules():
#         if isinstance(m, torch.nn.Linear):
#             hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

#     print("Collecting activation scales...")
#     pbar = tqdm(range(num_samples))
#     dataset = load_dataset("json", data_files=dataset_path, split="train")
#     dataset = dataset.shuffle(seed=42)
#     for i in pbar:
#         input_ids = tokenizer(
#             dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
#         ).input_ids.to(device)
#         model(input_ids)
#         # act_dict is on CPU, so this is fine
#         mean_scale = np.mean([v["input"] for v in act_dict.values()])
#         pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
#     for hook in hooks:
#         hook.remove()

#     decoder_layer_scales = []
#     for idx in range(model.config.num_hidden_layers):
#         scale_dict = {}
#         scale_dict["attn_input_scale"] = (
#             act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["input"] / 127
#         )
#         scale_dict["q_output_scale"] = (
#             act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["output"] / 127
#         )
#         scale_dict["k_output_scale"] = (
#             act_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"]["output"] / 127
#         )
#         scale_dict["v_output_scale"] = (
#             act_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"]["output"] / 127
#         )
#         scale_dict["out_input_scale"] = (
#             act_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"]["input"] / 127
#         )
#         scale_dict["fc1_input_scale"] = (
#             act_dict[f"model.decoder.layers.{idx}.fc1"]["input"] / 127
#         )
#         scale_dict["fc2_input_scale"] = (
#             act_dict[f"model.decoder.layers.{idx}.fc2"]["input"] / 127
#         )
#         decoder_layer_scales.append(scale_dict)

#     return decoder_layer_scales, act_dict

