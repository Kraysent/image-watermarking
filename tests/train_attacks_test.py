import torch
from omegaconf import OmegaConf

import watermarking.train.attacks as attacks


def test_build_attack_pipeline_empty() -> None:
    pipeline = attacks.build_attack_pipeline(OmegaConf.create([]))
    image = torch.rand(2, 3, 16, 16)
    assert torch.equal(pipeline.apply(image), image)


def test_build_attack_pipeline_rejects_probability_sum_over_one() -> None:
    try:
        attacks.build_attack_pipeline(
            OmegaConf.create(
                [
                    {"type": "hflip", "probability": 0.6, "args": {}},
                    {"type": "blur", "probability": 0.5, "args": {"radius": 1.0}},
                ]
            )
        )
    except ValueError as error:
        assert "must be <= 1" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_hflip_mirrors_width() -> None:
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 11.0
    flipped = attacks.hflip(image[0])
    assert torch.equal(flipped, image[0].flip(-1))


def test_brightness_scales_and_clamps() -> None:
    image = torch.full((3, 4, 4), 0.8)
    bright = attacks.brightness(image, factor=2.0)
    assert torch.all(bright == 1.0)


def test_resize_preserves_shape() -> None:
    image = torch.rand(3, 32, 32)
    output = attacks.resize(image, scale=0.5)
    assert output.shape == image.shape


def test_crop_preserves_shape() -> None:
    torch.manual_seed(0)
    image = torch.rand(3, 32, 32)
    output = attacks.crop(image, scale=0.7)
    assert output.shape == image.shape


def test_rotate_preserves_shape() -> None:
    image = torch.rand(3, 32, 32)
    output = attacks.rotate(image, degrees=15.0)
    assert output.shape == image.shape


def test_blur_preserves_shape() -> None:
    image = torch.rand(3, 32, 32)
    output = attacks.blur(image, radius=1.5)
    assert output.shape == image.shape


def test_jpeg_preserves_shape_and_uses_ste() -> None:
    image = torch.rand(3, 32, 32, requires_grad=True)
    output = attacks.jpeg(image, quality=50.0)
    assert output.shape == image.shape
    output.sum().backward()
    assert image.grad is not None
    assert torch.allclose(image.grad, torch.ones_like(image))


def test_pipeline_always_applies_when_probability_one() -> None:
    torch.manual_seed(0)
    pipeline = attacks.AttackPipeline([{"type": "hflip", "probability": 1.0, "args": {}}])
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 11.0
    output = pipeline.apply(image)
    assert torch.equal(output, image.flip(-1))
