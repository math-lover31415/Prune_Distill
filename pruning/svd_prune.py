import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.return_types import mode


def get_target_layers(model):
    # transformer encoder layers
    return [
        (module, "weight")
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ".layer." in name
    ]


def replace_module(model, layer_name, new_module):
    parts = layer_name.split(".")
    parent = model

    for part in parts[:-1]:
        parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)


def svd_prune_bert(model, sparsity):

    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("Must be a real number in [0,1]")
    
    #not doing in place here, so changed it
    thinned_model = copy.deepcopy(model)
    if sparsity == 0.0:
        return thinned_model

    targets = get_target_layers(thinned_model)
    
    for layer_name, old_layer in targets:
            weight = old_layer.weight.detach()

            out_features, in_features = weight.shape

            # Keep this fraction of the original rank
            rank = max(
                1,
                int((1.0 - sparsity) * min(out_features, in_features))
            )

            # Compute SVD in float32 for numerical stability
            U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)

            U_k = U[:, :rank]
            S_k = S[:rank]
            Vh_k = Vh[:rank, :]

            # W ≈ (U_k @ diag(S_k)) @ Vh_k
            first_layer = nn.Linear(
                in_features,
                rank,
                bias=False
            )

            second_layer = nn.Linear(
                rank,
                out_features,
                bias=old_layer.bias is not None
            )

            with torch.no_grad():
                # First layer contains Vh
                first_layer.weight.copy_(
                    Vh_k.to(
                        device=weight.device,
                        dtype=weight.dtype
                    )
                )

                # Second layer contains U @ Sigma
                second_layer.weight.copy_(
                    (U_k * S_k).to(
                        device=weight.device,
                        dtype=weight.dtype
                    )
                )

                if old_layer.bias is not None:
                    second_layer.bias.copy_(old_layer.bias.detach())

            # Preserve device and dtype
            first_layer = first_layer.to(
                device=weight.device,
                dtype=weight.dtype
            )
            second_layer = second_layer.to(
                device=weight.device,
                dtype=weight.dtype
            )

            # Replace:
            # Linear(in_features, out_features)
            # with:
            # Linear(in_features, rank) -> Linear(rank, out_features)
            replace_module(
                thinned_model,
                layer_name,
                nn.Sequential(first_layer, second_layer)
            )

    return thinned_model


def sparsity_report(model):
    per_layer, zeros, total = {}, 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and ".layer." in name:
            z = int((m.weight == 0).sum().item())
            per_layer[name] = z / m.weight.numel()
            zeros += z
            total += m.weight.numel()
    return {"overall": zeros / total if total else 0.0, "per_layer": per_layer}
