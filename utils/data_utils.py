import torch


def mask_features(features: list, mask: torch.Tensor):
    """
    Resize the mask to the desired dimensions
    :param dims: list of desired dimensions (e.g, [(512,384), (256,182)])
    :param mask: mask to be resized
    :return: resized mask
    """

    for i, feature in enumerate(features):
        mask = torch.nn.functional.interpolate(mask, size=feature.shape[-2:])
        features[i] = feature * (1 - mask)
    
    return features