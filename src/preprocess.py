import torch
from torchvision.transforms import Resize
import einops

IMAGE_MEAN = (127.5,) * 3
IMAGE_STD = (127.5,) * 3

def preprocess_image(
    image: torch.Tensor,
    image_shape: tuple[int, int] | None = (896, 896),
) -> torch.Tensor:              # [h, w, c]
    """ bi-linear resize, anti-aliasing, normalize """
    # Ensure image is [C, H, W] for Resize
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
        
    image = Resize(size=image_shape, antialias=True)(image)
    image = image.float()
    image = normalize_image(image)
    image = torch.clip(image, -1, 1)
    return image

def normalize_image(
    image: torch.Tensor,
) -> torch.Tensor:
    # image is [C, H, W]
    mean = torch.tensor(IMAGE_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGE_STD).view(-1, 1, 1)
    image = (image - mean) / std
    return image

def patchify_image(
    image: torch.Tensor,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Converts image [C, H, W] to patches [num_patches, patch_dim]
    patch_dim = C * patch_size * patch_size
    """
    c, h, w = image.shape
    p = patch_size
    # [c, (h_p p), (w_p p)] -> [h_p, w_p, c, p, p]
    patches = einops.rearrange(
        image, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=p, p2=p
    )
    return patches
