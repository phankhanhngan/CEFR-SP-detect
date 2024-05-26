import torch
import numpy as np


def convert_numeral_to_six_levels(levels):
    level_thresholds = np.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5])
    return _conversion(level_thresholds, levels)


def _conversion(level_thresholds, values):
    thresh_array = np.tile(level_thresholds, reps=(values.shape[0], 1))
    array = np.tile(values, reps=(1, level_thresholds.shape[0]))
    levels = np.maximum(np.zeros((values.shape[0], 1)),
                        np.count_nonzero(thresh_array <= array, axis=1, keepdims=True) - 1).astype(int)
    return levels


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
