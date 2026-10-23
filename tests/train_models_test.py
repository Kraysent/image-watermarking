import torch

import watermarking.train.models as models


def test_encoder_decoder_shapes() -> None:
    message_bits = 8
    batch = 2
    size = 32
    image = torch.rand(batch, 3, size, size)
    message = torch.randint(0, 2, (batch, message_bits), dtype=torch.float32)

    encoder = models.Encoder(message_bits=message_bits, channels=[8], residual_scale=0.1)
    decoder = models.Decoder(message_bits=message_bits, channels=[8])

    diff = encoder(image, message)
    prediction = decoder((image + diff).clamp(0.0, 1.0))

    assert diff.shape == image.shape
    assert prediction.shape == message.shape


def test_encoder_decoder_multi_layer_shapes() -> None:
    message_bits = 8
    image = torch.rand(2, 3, 32, 32)
    message = torch.randint(0, 2, (2, message_bits), dtype=torch.float32)
    channels = [8, 16, 16]

    encoder = models.Encoder(message_bits=message_bits, channels=channels, residual_scale=0.1)
    decoder = models.Decoder(message_bits=message_bits, channels=channels)

    diff = encoder(image, message)
    prediction = decoder((image + diff).clamp(0.0, 1.0))

    assert diff.shape == image.shape
    assert prediction.shape == message.shape


def test_training_step_reduces_graph() -> None:
    message_bits = 4
    image = torch.rand(2, 3, 16, 16)
    message = torch.randint(0, 2, (2, message_bits), dtype=torch.float32)
    encoder = models.Encoder(message_bits=message_bits, channels=[4, 8], residual_scale=0.1)
    decoder = models.Decoder(message_bits=message_bits, channels=[4, 8])
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    diff = encoder(image, message)
    prediction = decoder((image + diff).clamp(0.0, 1.0))
    loss = torch.nn.functional.mse_loss(prediction, message) + torch.nn.functional.mse_loss(
        diff, torch.zeros_like(diff)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.ndim == 0
