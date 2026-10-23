import torch

import watermarking.train.metrics as metrics


def test_message_accuracy_requires_all_bits() -> None:
    prediction = torch.tensor([[0.9, 0.1, 0.8], [0.9, 0.9, 0.1]])
    message = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert metrics.message_accuracy(prediction, message).tolist() == [1.0, 0.0]


def test_psnr_is_high_for_identical_images() -> None:
    image = torch.rand(2, 3, 32, 32)
    values = metrics.psnr(image, image)
    assert values.shape == (2,)
    assert torch.all(values > 80.0)


def test_ssim_is_one_for_identical_images() -> None:
    image = torch.rand(2, 3, 32, 32)
    values = metrics.ssim(image, image)
    assert values.shape == (2,)
    assert torch.allclose(values, torch.ones_like(values), atol=1e-5)


def test_psnr_and_ssim_drop_with_noise() -> None:
    image = torch.rand(2, 3, 32, 32)
    noisy = (image + 0.2).clamp(0.0, 1.0)
    assert torch.all(metrics.psnr(image, noisy) < metrics.psnr(image, image))
    assert torch.all(metrics.ssim(image, noisy) < metrics.ssim(image, image))
