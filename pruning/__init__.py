from .magnitude_prune import magnitude_prune_bert
from .svd_prune import svd_prune_bert

def identity(model, sparsity):
    return model

def local_magnitude(model, sparsity):
    return magnitude_prune_bert(model, sparsity, "local")

def global_magnitude(model, sparsity):
    return magnitude_prune_bert(model, sparsity, "global")

def svd_pruning(model, sparsity):
    return svd_prune_bert(model, sparsity)

# pruning_functions = {"svd": svd_pruning}
pruning_functions = {"identity": identity, "local_mag": local_magnitude, "global_mag": global_magnitude, "svd": svd_pruning}
