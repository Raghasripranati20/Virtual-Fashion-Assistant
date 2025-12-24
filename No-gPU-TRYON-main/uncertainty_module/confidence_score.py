def confidence_from_uncertainty(uncertainty_map):
    mean_uncertainty = uncertainty_map.mean().item()
    confidence = 1 / (1 + mean_uncertainty)
    return round(confidence * 100, 2)
