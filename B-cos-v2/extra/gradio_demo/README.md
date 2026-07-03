# B-cos Interactive Lab

Web demo nay giup quan sat truc tiep cac thanh phan trong B-cos layer:

- Linear term: `w_hat * x`
- Cosine alignment: `|cos(theta)| = |(w_hat*x)| / ||x_patch||`
- B-map: `|cos(theta)|^(b-1)`
- Final output: `B-map * linear term`

## Chay demo

Tu thu muc `B-cos-v2`:

```bash
pip install gradio matplotlib numpy pillow torch
python extra/gradio_demo/bcos_interactive_demo.py
```

Mo trinh duyet tai:

- http://127.0.0.1:7860

## Cach dung

1. Upload anh.
2. Chon filter (Sobel/Blur/Sharpen...).
3. Keo slider `B value` de nhin tac dong cua B-map.
4. Bam `Chay phan tich 1 layer` de xem cac map trung gian.
5. Bam `Mo phong stack` de xem xu huong kich hoat khi xep nhieu layer.

## Ghi chu

- Demo tap trung vao truc quan co che B-cos, khong thay the toan bo pipeline train/inference cua model day du.
- Ban co the mo rong bang cach nap weight that cua `BcosConv2d` tu checkpoint va hien thi theo tung block.
