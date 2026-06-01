"""
Grad-CAM visualisation (Selvaraju et al., 2017).

Generates class activation maps that highlight which image regions
contributed most to the model's prediction. Used to verify that the
model focuses on the lesion area rather than image artefacts (hair,
dermatoscope frame, ruler marks).

The last convolutional layer of each backbone is used, as it captures
the highest-level semantic features before the global pooling operation.
"""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


_LAST_CONV = {
    'resnet50':        lambda m: m.layer4[-1].conv3,
    'densenet121':     lambda m: m.features.denseblock4.denselayer16.conv2,
    'efficientnet_b3': lambda m: m.features[8][0],
}


def compute_gradcam(
    model:        torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    model_name:   str,
    device:       torch.device = None,
) -> np.ndarray:
    """
    Compute the Grad-CAM heatmap for a single image.

    Args:
        model:        trained model in eval mode
        image_tensor: (C, H, W) normalised image tensor
        target_class: class index to explain
        model_name:   one of 'resnet50', 'densenet121', 'efficientnet_b3'
        device:       if None, uses model's current device

    Returns:
        (H', W') normalised heatmap in [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device
    if model_name not in _LAST_CONV:
        raise ValueError(f"Grad-CAM not implemented for '{model_name}'")

    target_layer = _LAST_CONV[model_name](model)
    grads, acts  = [], []

    def _save_grad(g):
        grads.append(g.detach().cpu())

    def _forward_hook(module, input, output):
        acts.append(output.detach().cpu())
        output.register_hook(_save_grad)

    handle = target_layer.register_forward_hook(_forward_hook)
    model.eval()

    out = model(image_tensor.unsqueeze(0).to(device))
    model.zero_grad()
    out[0, target_class].backward()
    handle.remove()

    g = grads[0].squeeze().numpy()       # (C, H, W)
    a = acts[0].squeeze().numpy()        # (C, H, W)

    # Global average pooling of gradients, then weighted sum of activations
    weights = g.mean(axis=(1, 2))        # (C,)
    cam = np.maximum(np.einsum('c,chw->hw', weights, a), 0)

    # Normalise to [0, 1]
    mn, mx = cam.min(), cam.max()
    if mx - mn > 1e-8:
        cam = (cam - mn) / (mx - mn)
    return cam


def overlay_gradcam(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Args:
        image: (H, W, 3) uint8 RGB image
        cam:   (H', W') heatmap in [0, 1], will be resized to match image

    Returns:
        (H, W, 3) uint8 blended image
    """
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8))
        .resize((image.shape[1], image.shape[0]), Image.BILINEAR)
    ) / 255.0

    heatmap = (plt.get_cmap('jet')(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = (0.6 * image + 0.4 * heatmap).clip(0, 255).astype(np.uint8)
    return blended
