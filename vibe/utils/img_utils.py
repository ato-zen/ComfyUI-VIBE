"""Image utils."""

import torch
from PIL import Image
from torchvision import transforms as T  # type: ignore[import-untyped] # noqa: N812
from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]


def get_divisible_size(height: int, width: int, factor: int = 16, *, expand_to_fit: bool = False) -> tuple[int, int]:
    """Adjusts the given dimensions (height and width) to be divisible by the specified factor.

    Args:
        height (int): The original height of the input.
        width (int): The original width of the input.
        factor (int): The factor by which the dimensions should be divisible. Default is 16.
        expand_to_fit (bool): If True, dimensions will be adjusted upwards to the nearest multiple of factor.
    """
    h_target = height // factor * factor
    w_target = width // factor * factor
    h_target = max(h_target, factor)
    w_target = max(w_target, factor)

    if expand_to_fit:
        if w_target < width:
            w_target += factor
        if h_target < height:
            h_target += factor

    return h_target, w_target


def get_multiscale_transform(  # noqa: PLR0913
    closest_size: tuple[int, int],
    orig_size: tuple[int, int],
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    resize_factor: int = 8,
    out_img_type: str = "pt",
    *,
    crop: bool = False,
) -> tuple[T.Compose, tuple[int, ...] | None]:
    """Return transformations for the image and x,y,x,y coordinates to remove padding after generations.

    Args:
        closest_size (tuple[int, int]): The closest size to the original size.
        orig_size (tuple[int, int]): The original size of the image.
        interpolation (InterpolationMode): The interpolation mode to use.
        resize_factor (int): The resize factor to use.
        out_img_type (str): The type of the output image.
        crop (bool): Whether to crop the image.

    Returns:
        tuple[T.Compose, tuple[int, ...] | None]: The transformations and the crop coordinates.
    """
    transforms = [T.Lambda(lambda img: img.convert("RGB"))]

    height, width = closest_size
    if height % resize_factor != 0 or width % resize_factor != 0:
        closest_size = get_divisible_size(height, width, resize_factor, expand_to_fit=not crop)  # type: ignore

    ori_h, ori_w = orig_size
    # if crop==True => resize by biggest size; if crop==False => resize by smallest size
    if (crop and closest_size[0] / ori_h > closest_size[1] / ori_w) or (
        not crop and closest_size[0] / ori_h < closest_size[1] / ori_w
    ):
        resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
    else:
        resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]  # type: ignore

    if not crop:
        # resize to resize_size and pad to closest_size
        left, top = abs(closest_size[1] - resize_size[1]) // 2, abs(closest_size[0] - resize_size[0]) // 2
        right, bottom = closest_size[1] - resize_size[1] - left, closest_size[0] - resize_size[0] - top
        pad = [left, top, right, bottom]
        transforms.extend(
            [
                T.Resize(resize_size, interpolation=interpolation),
                T.Pad(pad, fill=0, padding_mode="constant"),
            ]
        )
        crop_coords = (left, top, closest_size[1] - right, closest_size[0] - bottom)
    else:
        # resize to resize_size and crop to closest_size
        crop_coords = None
        transforms.extend(
            [
                T.Resize(resize_size, interpolation=interpolation),
                T.CenterCrop(closest_size),
            ]
        )

    if out_img_type == "pt":
        transforms.extend([T.ToTensor(), T.Normalize([0.5], [0.5])])
    transform = T.Compose(transforms)
    return transform, crop_coords  # type: ignore


def postprocess_padded_image(
    image: Image.Image | torch.Tensor,
    crop_coords: tuple[int, ...] | list[tuple[int, ...]] | None,
) -> Image.Image | torch.Tensor:
    """Remove paddings from padded images."""
    if crop_coords is None:
        return image

    def _crop(img: torch.Tensor, coords: tuple[int, ...]) -> torch.Tensor:
        assert len(coords) == 4  # noqa: S101,PLR2004
        if img.shape[0] in [1, 3, 6]:
            # for (C, H, W) tensor
            left, top = coords[:2]
            width, height = coords[2] - coords[0], coords[3] - coords[1]
            if img.shape[0] in [1, 3]:
                return T.functional.crop(img, top, left, height, width)
            winner, looser = img.chunk(2, dim=0)
            winner = T.functional.crop(winner, top, left, height, width)
            looser = T.functional.crop(looser, top, left, height, width)
            return torch.cat((winner, looser), dim=0)
        # for (H, W, C) tensor
        left, top, right, bottom = coords[:4]
        return img[top:bottom, left:right, :]

    if isinstance(image, torch.Tensor) and image.ndim == 4:  # noqa: PLR2004
        assert isinstance(crop_coords, list) and image.shape[0] == len(crop_coords)  # noqa: PT018,S101
        output_images = []
        for img, coords in zip(image, crop_coords, strict=True):
            output_images.append(_crop(img, coords)[None, :])
        return torch.cat(output_images, dim=0)

    if isinstance(image, torch.Tensor) and image.ndim == 3:  # noqa: PLR2004
        return _crop(image, crop_coords)  # type: ignore

    x1, y1, x2, y2 = crop_coords
    return image.crop([x1, y1, x2, y2])  # type: ignore


def revert_resize(img: Image.Image, initial_hw: tuple[int, int]) -> Image.Image:
    """Resize the image to the initial height and width.

    Args:
        img (Image.Image): The input image.
        initial_hw (tuple[int, int]): The initial height and width.

    Returns:
        Image.Image: The resized image.
    """
    height, width = initial_hw
    if (width, height) == img.size:
        return img

    if hasattr(Image, "Resampling"):
        return img.resize((width, height), Image.Resampling.LANCZOS)
    return img.resize((width, height), Image.LANCZOS)  # type: ignore[attr-defined]
