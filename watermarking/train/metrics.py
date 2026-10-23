import torch
import torch.nn.functional as functional


def message_accuracy(prediction: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    predicted = prediction > 0.5
    return (predicted == message).all(dim=-1).float()


def psnr(cover: torch.Tensor, watermarked: torch.Tensor) -> torch.Tensor:
    mse = ((cover - watermarked) ** 2).flatten(1).mean(dim=1).clamp(min=1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def ssim(cover: torch.Tensor, watermarked: torch.Tensor) -> torch.Tensor:
    window = _ssim_window(cover.device, cover.dtype)
    pad = window.shape[-1] // 2
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = functional.conv2d(cover, window, padding=pad, groups=cover.shape[1])
    mu_y = functional.conv2d(watermarked, window, padding=pad, groups=watermarked.shape[1])
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = functional.conv2d(cover * cover, window, padding=pad, groups=cover.shape[1]) - mu_x2
    sigma_y2 = functional.conv2d(watermarked * watermarked, window, padding=pad, groups=watermarked.shape[1]) - mu_y2
    sigma_xy = functional.conv2d(cover * watermarked, window, padding=pad, groups=cover.shape[1]) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


def _ssim_window(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=dtype) - 5.0
    gauss = torch.exp(-(coords.pow(2)) / (2.0 * 1.5**2))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel / kernel.sum()
    return kernel.expand(3, 1, 11, 11).contiguous()
