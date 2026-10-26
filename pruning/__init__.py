from .magnitude_prune import magnitude_prune_bert

def identity(model, sparsity):
    return model

def local_magnitude(model, sparsity):
    return magnitude_prune_bert(model, sparsity, "local")

def global_magnitude(model, sparsity):
    return magnitude_prune_bert(model, sparsity, "global")



pruning_functions = {"identity": identity, "local_mag": local_magnitude, "global_mag": global_magnitude}