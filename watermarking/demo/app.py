import tempfile
from pathlib import Path

import gradio as gr
import PIL.Image

import watermarking.demo.model as model_module


def build_app(session: model_module.DemoSession) -> gr.Blocks:
    with gr.Blocks(title="Watermarking demo") as demo:
        gr.Markdown(f"# Watermarking demo\nLoaded run `{session.run_id}` · {session.message_bits} bits")

        with gr.Tab("Embed"):
            with gr.Row():
                embed_image = gr.Image(type="pil", label="Image")
                with gr.Column():
                    message_mode = gr.Radio(
                        choices=["random", "text", "hex"],
                        value="text",
                        label="Message mode",
                    )
                    message_value = gr.Textbox(label="Message", value="hi")
                    embed_button = gr.Button("Embed", variant="primary")
            embed_grid = gr.Image(type="pil", label="Original | watermarked | residual")
            embed_download = gr.File(label="Download watermarked PNG")
            embedded_bits = gr.Textbox(label="Embedded bits", interactive=False)
            recovered_bits = gr.Textbox(label="Recovered bits", interactive=False)
            bit_accuracy = gr.Number(label="Bit accuracy", interactive=False)

            embed_button.click(
                fn=lambda image, mode, value: _embed(session, image, mode, value),
                inputs=[embed_image, message_mode, message_value],
                outputs=[embed_grid, embed_download, embedded_bits, recovered_bits, bit_accuracy],
            )

        with gr.Tab("Extract"):
            extract_image = gr.Image(type="pil", label="Image")
            extract_button = gr.Button("Extract", variant="primary")
            extract_bits = gr.Textbox(label="Bits", interactive=False)
            extract_text = gr.Textbox(label="Text", interactive=False)
            extract_hex = gr.Textbox(label="Hex", interactive=False)
            extract_confidence = gr.Number(label="Confidence (mean |pred − 0.5|)", interactive=False)

            extract_button.click(
                fn=lambda image: _extract(session, image),
                inputs=[extract_image],
                outputs=[extract_bits, extract_text, extract_hex, extract_confidence],
            )

    return demo


def _embed(
    session: model_module.DemoSession,
    image: PIL.Image.Image | None,
    mode: str,
    value: str,
) -> tuple[PIL.Image.Image, str, str, str, float]:
    if image is None:
        raise gr.Error("Upload an image first")
    grid, watermarked, embedded, recovered, accuracy = model_module.embed(
        session,
        image,
        mode=mode,
        message_value=value or "",
    )
    output_path = Path(tempfile.mkdtemp()) / "watermarked.png"
    watermarked.save(output_path)
    return grid, str(output_path), embedded, recovered, accuracy


def _extract(
    session: model_module.DemoSession,
    image: PIL.Image.Image | None,
) -> tuple[str, str, str, float]:
    if image is None:
        raise gr.Error("Upload an image first")
    return model_module.extract(session, image)
