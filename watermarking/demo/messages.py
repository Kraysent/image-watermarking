import torch


def random_bits(message_bits: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randint(0, 2, (message_bits,), dtype=torch.float32, generator=generator)


def text_to_bits(text: str, message_bits: int) -> torch.Tensor:
    n_bytes = (message_bits + 7) // 8
    raw = text.encode("utf-8")[:n_bytes].ljust(n_bytes, b"\0")
    bits: list[float] = []
    for byte in raw:
        for shift in range(7, -1, -1):
            bits.append(float((byte >> shift) & 1))
    return torch.tensor(bits[:message_bits], dtype=torch.float32)


def hex_to_bits(value: str, message_bits: int) -> torch.Tensor:
    cleaned = value.strip().removeprefix("0x").removeprefix("0X").replace(" ", "")
    if len(cleaned) % 2 == 1:
        cleaned = f"0{cleaned}"
    n_bytes = (message_bits + 7) // 8
    raw = bytes.fromhex(cleaned)[:n_bytes].ljust(n_bytes, b"\0")
    bits: list[float] = []
    for byte in raw:
        for shift in range(7, -1, -1):
            bits.append(float((byte >> shift) & 1))
    return torch.tensor(bits[:message_bits], dtype=torch.float32)


def bits_from_mode(mode: str, value: str, message_bits: int) -> torch.Tensor:
    if mode == "random":
        return random_bits(message_bits)
    if mode == "text":
        return text_to_bits(value, message_bits)
    if mode == "hex":
        return hex_to_bits(value, message_bits)
    raise ValueError(f"unknown message mode: {mode}")


def threshold_bits(prediction: torch.Tensor) -> torch.Tensor:
    return (prediction >= 0.5).to(dtype=torch.float32)


def format_bits(bits: torch.Tensor) -> str:
    values = [str(int(bit.item())) for bit in bits.detach().cpu().flatten()]
    return " ".join("".join(values[index : index + 8]) for index in range(0, len(values), 8))


def bits_to_bytes(bits: torch.Tensor) -> bytes:
    values = [int(bit.item()) for bit in bits.detach().cpu().flatten()]
    packed: list[int] = []
    for index in range(0, len(values), 8):
        chunk = values[index : index + 8]
        if len(chunk) < 8:
            chunk = chunk + [0] * (8 - len(chunk))
        byte = 0
        for bit in chunk:
            byte = (byte << 1) | bit
        packed.append(byte)
    return bytes(packed)


def bits_to_text(bits: torch.Tensor) -> str:
    return bits_to_bytes(bits).rstrip(b"\0").decode("utf-8", errors="replace")


def bits_to_hex(bits: torch.Tensor) -> str:
    return bits_to_bytes(bits).hex()


def bit_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = threshold_bits(prediction)
    return float((predicted == target).float().mean().item())


def confidence(prediction: torch.Tensor) -> float:
    return float((prediction.detach() - 0.5).abs().mean().item())
