import math
from typing import List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


EPS = 1e-6


def to_grayscale_tensor(image: Image.Image, size: int = 224) -> torch.Tensor:
    image = image.convert("L").resize((size, size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None, ...]


def kernel_from_name(name: str, kernel_size: int) -> torch.Tensor:
    if name == "Sobel-X":
        base = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    elif name == "Sobel-Y":
        base = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    elif name == "Sharpen":
        base = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])
    elif name == "Blur":
        base = torch.ones((3, 3), dtype=torch.float32) / 9.0
    elif name == "Laplacian":
        base = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    else:
        gen = torch.Generator().manual_seed(0)
        base = torch.randn((3, 3), generator=gen)

    if kernel_size == 3:
        return base[None, None, ...]

    if kernel_size == 5:
        out = torch.zeros((5, 5), dtype=torch.float32)
        out[1:4, 1:4] = base
        return out[None, None, ...]

    raise ValueError("Only kernel sizes 3 or 5 are supported in this demo")


def normalize_to_image(arr: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < 1e-12:
        normed = np.zeros_like(arr)
    else:
        normed = (arr - arr_min) / (arr_max - arr_min)
    colored = plt.get_cmap(cmap)(normed)[..., :3]
    return (colored * 255).astype(np.uint8)


def signed_to_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    vmax = max(np.max(np.abs(arr)), 1e-6)
    normed = (arr / vmax + 1.0) * 0.5
    colored = plt.get_cmap("coolwarm")(normed)[..., :3]
    return (colored * 255).astype(np.uint8)


def bcos_single_layer(
    image: Image.Image,
    kernel_name: str,
    kernel_size: int,
    b_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if image is None:
        raise gr.Error("Hay upload mot anh de bat dau.")

    x = to_grayscale_tensor(image)
    kernel = kernel_from_name(kernel_name, kernel_size)
    weight_hat = F.normalize(kernel, dim=(1, 2, 3))
    pad = kernel_size // 2

    linear = F.conv2d(x, weight_hat, padding=pad)
    patch_norm = torch.sqrt(F.conv2d(x.square(), torch.ones_like(kernel), padding=pad) + EPS)

    cos_abs = torch.abs(linear / patch_norm)

    if abs(b_value - 1.0) < 1e-8:
        b_map = torch.ones_like(cos_abs)
    else:
        b_map = torch.pow(cos_abs + EPS, b_value - 1.0)

    out = b_map * linear

    linear_np = linear[0, 0].detach().cpu().numpy()
    norm_np = patch_norm[0, 0].detach().cpu().numpy()
    cos_np = cos_abs[0, 0].detach().cpu().numpy()
    b_np = b_map[0, 0].detach().cpu().numpy()
    out_np = out[0, 0].detach().cpu().numpy()

    summary = (
        "### Giai thich nhanh\n"
        f"- Cong thuc layer: y = (|cos(theta)|^(b-1)) * (w_hat * x)\n"
        f"- Gia tri b hien tai: {b_value:.2f}\n"
        f"- Mean |linear|: {np.mean(np.abs(linear_np)):.4f}\n"
        f"- Mean |cos|: {np.mean(cos_np):.4f}\n"
        f"- Mean B-map: {np.mean(b_np):.4f}\n"
        f"- Mean |output|: {np.mean(np.abs(out_np)):.4f}\n"
        "\n"
        "B-map cao tai cac vi tri co huong patch can hang tot voi filter. "
        "Khi b tang, su khac biet giua vi tri hop huong va lech huong duoc nhan manh hon."
    )

    return (
        signed_to_image(linear_np),
        normalize_to_image(norm_np),
        normalize_to_image(cos_np),
        normalize_to_image(b_np),
        signed_to_image(out_np),
        summary,
    )


def stack_dynamics(
    image: Image.Image,
    kernel_name: str,
    kernel_size: int,
    b_value: float,
    depth: int,
) -> Tuple[plt.Figure, np.ndarray, str]:
    if image is None:
        raise gr.Error("Hay upload mot anh de bat dau.")

    x = to_grayscale_tensor(image)
    kernel = kernel_from_name(kernel_name, kernel_size)
    weight_hat = F.normalize(kernel, dim=(1, 2, 3))
    ones = torch.ones_like(kernel)
    pad = kernel_size // 2

    stats: List[float] = []
    activ = x
    for _ in range(depth):
        linear = F.conv2d(activ, weight_hat, padding=pad)
        patch_norm = torch.sqrt(F.conv2d(activ.square(), ones, padding=pad) + EPS)
        cos_abs = torch.abs(linear / patch_norm)

        if abs(b_value - 1.0) < 1e-8:
            b_map = torch.ones_like(cos_abs)
        else:
            b_map = torch.pow(cos_abs + EPS, b_value - 1.0)

        activ = b_map * linear
        stats.append(float(torch.mean(torch.abs(activ)).cpu().item()))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, depth + 1), stats, marker="o", linewidth=2)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean |activation|")
    ax.set_title(f"Do lon kich hoat qua {depth} layer B-cos (b={b_value:.2f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    final_map = activ[0, 0].detach().cpu().numpy()

    explanation = (
        "### Nhin theo chieu sau\n"
        "- Neu b > 1: cac vi tri can huong duoc uu tien manh hon qua tung layer.\n"
        "- Neu b ~ 1: gan nhu conv thuong voi trong so da normalize.\n"
        "- Neu b rat lon: de tan mat thong tin o cac vung lech huong."
    )

    return fig, signed_to_image(final_map), explanation


def intro_text() -> str:
    return (
        "## B-cos Interactive Lab\n"
        "Web nay giup ban nhin truc tiep cach B-cos tao output:\n"
        "1) Linear term: w_hat * x\n"
        "2) Cosine alignment: |cos(theta)| = |(w_hat*x)| / ||x_patch||\n"
        "3) B-map: |cos(theta)|^(b-1)\n"
        "4) Output: B-map * linear term\n"
        "\n"
        "Hay upload mot anh, chon filter, doi b va quan sat cac ban do trung gian."
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="B-cos Interactive Lab") as demo:
        gr.Markdown(intro_text())

        with gr.Row():
            image_input = gr.Image(type="pil", label="Input image")
            with gr.Column():
                kernel_name = gr.Dropdown(
                    choices=["Sobel-X", "Sobel-Y", "Sharpen", "Blur", "Laplacian", "Random"],
                    value="Sobel-X",
                    label="Kernel",
                )
                kernel_size = gr.Radio(
                    choices=[3, 5],
                    value=3,
                    label="Kernel size",
                )
                b_value = gr.Slider(
                    minimum=1.0,
                    maximum=8.0,
                    value=2.0,
                    step=0.1,
                    label="B value",
                )

        gr.Markdown("### Phan tich mot layer B-cos")
        run_one = gr.Button("Chay phan tich 1 layer", variant="primary")

        with gr.Row():
            linear_img = gr.Image(label="Linear map (signed)")
            norm_img = gr.Image(label="Patch norm ||x_patch||")
            cos_img = gr.Image(label="|cos(theta)| map")

        with gr.Row():
            bmap_img = gr.Image(label="B-map = |cos|^(b-1)")
            out_img = gr.Image(label="Final output map")
            summary_md = gr.Markdown()

        run_one.click(
            fn=bcos_single_layer,
            inputs=[image_input, kernel_name, kernel_size, b_value],
            outputs=[linear_img, norm_img, cos_img, bmap_img, out_img, summary_md],
        )

        gr.Markdown("### Dong hoc khi xep nhieu layer")
        depth = gr.Slider(
            minimum=1,
            maximum=20,
            value=6,
            step=1,
            label="So layer mo phong",
        )
        run_stack = gr.Button("Mo phong stack", variant="secondary")

        with gr.Row():
            depth_plot = gr.Plot(label="Mean |activation| theo layer")
            final_depth_map = gr.Image(label="Activation sau layer cuoi")
            depth_md = gr.Markdown()

        run_stack.click(
            fn=stack_dynamics,
            inputs=[image_input, kernel_name, kernel_size, b_value, depth],
            outputs=[depth_plot, final_depth_map, depth_md],
        )

        gr.Markdown(
            "Mo rong de gan hon voi model that: thay kernel thu cong bang weight that cua tung kenh trong BcosConv2d "
            "va them theo doi theo block ResNet."
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="127.0.0.1", server_port=7860)
