import copy
import torch.nn as nn
import torch.nn.utils.prune as prune


def get_target_layers(model):
    # transformer encoder layers
    return [
        (module, "weight")
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ".layer." in name
    ]


def magnitude_prune_bert(model, sparsity, mode):
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("Must be a real number in [0,1]")
    if mode not in ("global", "local"):
        raise ValueError("mode must be 'global' or 'local' ")
    
    #not doing in place here, so changed it
    pruned_model = copy.deepcopy(model)
    if sparsity == 0.0:
        return pruned_model

    targets = get_target_layers(pruned_model)
    
    if mode == "global":
        prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=sparsity)
    else:
        for module, name in targets:
            prune.l1_unstructured(module, name=name, amount=sparsity)

    
    for module, name in targets:
        prune.remove(module, name)
    return pruned_model


def sparsity_report(model):
    per_layer, zeros, total = {}, 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and ".layer." in name:
            z = int((m.weight == 0).sum().item())
            per_layer[name] = z / m.weight.numel()
            zeros += z
            total += m.weight.numel()
    return {"overall": zeros / total if total else 0.0, "per_layer": per_layer}
