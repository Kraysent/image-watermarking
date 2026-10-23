import pytest
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


def test_image_loss_mse_matches_residual_mse() -> None:
    cover = torch.rand(2, 3, 32, 32)
    diff = torch.randn(2, 3, 32, 32) * 0.01
    watermarked = (cover + diff).clamp(0.0, 1.0)
    expected = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
    assert torch.allclose(metrics.image_loss(cover, watermarked, diff, "mse"), expected)


def test_image_loss_ssim_is_zero_for_identical_images() -> None:
    image = torch.rand(2, 3, 32, 32)
    diff = torch.zeros_like(image)
    assert metrics.image_loss(image, image, diff, "ssim").item() == pytest.approx(0.0, abs=1e-5)


def test_image_loss_psnr_decreases_with_noise() -> None:
    image = torch.rand(2, 3, 32, 32)
    noisy = (image + 0.2).clamp(0.0, 1.0)
    diff = noisy - image
    clean_loss = metrics.image_loss(image, image, torch.zeros_like(image), "psnr")
    noisy_loss = metrics.image_loss(image, noisy, diff, "psnr")
    assert noisy_loss > clean_loss


def test_image_loss_rejects_unknown_type() -> None:
    image = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match="unknown image_loss_type"):
        metrics.image_loss(image, image, torch.zeros_like(image), "l1")
