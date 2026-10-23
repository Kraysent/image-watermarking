import torch

import watermarking.demo.messages as messages


def test_text_bits_roundtrip() -> None:
    bits = messages.text_to_bits("hi", message_bits=32)
    assert bits.shape == (32,)
    assert messages.bits_to_text(bits).startswith("hi")


def test_hex_bits_roundtrip() -> None:
    bits = messages.hex_to_bits("deadbeef", message_bits=32)
    assert messages.bits_to_hex(bits) == "deadbeef"


def test_random_bits_shape() -> None:
    generator = torch.Generator().manual_seed(0)
    bits = messages.random_bits(16, generator=generator)
    assert bits.shape == (16,)
    assert set(bits.tolist()) <= {0.0, 1.0}


def test_bit_accuracy_and_confidence() -> None:
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])
    prediction = torch.tensor([0.1, 0.9, 0.6, 0.4])
    assert messages.bit_accuracy(prediction, target) == 1.0
    assert abs(messages.confidence(prediction) - 0.25) < 1e-6
