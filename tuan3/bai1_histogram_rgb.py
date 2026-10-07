"""Bài 1, ý a-b: Histogram ảnh RGB anh_mauRBG.png."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "anh_mauRBG.png"
OUTPUT = ROOT / "output_bai1_rgb"
BINS = LEVELS = (16, 64, 256)


def rgb_sang_xam(anh_rgb: np.ndarray) -> np.ndarray:
    """Chuyển RGB sang xám: Y = 0.299R + 0.587G + 0.114B."""
    return np.clip(np.rint(anh_rgb @ [0.299, 0.587, 0.114]), 0, 255).astype(np.uint8)


def histogram(anh: np.ndarray, bins: int = 256) -> np.ndarray:
    """h[k] là số pixel thuộc bin k; không dùng hàm Histogram có sẵn."""
    chi_so = np.minimum(anh.astype(np.uint32).ravel() * bins // 256, bins - 1)
    return np.bincount(chi_so, minlength=bins)


def luong_tu_hoa(anh_xam: np.ndarray, L: int) -> np.ndarray:
    """Giảm ảnh về L mức xám rồi trải lại 0..255 để dễ quan sát."""
    muc = np.minimum(anh_xam.astype(np.uint32) * L // 256, L - 1)
    return np.rint(muc * 255 / (L - 1)).astype(np.uint8)


def ve_hist(ax, anh: np.ndarray, bins: int, mau: str) -> None:
    hist = histogram(anh, bins)
    x = np.linspace(0, 256, bins, endpoint=False)
    ax.bar(x, hist, width=256 / bins, align="edge", color=mau)
    ax.set(xlim=(0, 256), xlabel="Mức cường độ", ylabel="Số pixel")
    ax.grid(axis="y", alpha=0.2)


def ve_y_a(anh_rgb: np.ndarray, anh_xam: np.ndarray) -> None:
    """Ý a: hiển thị ảnh và Histogram xám/R/G/B."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].imshow(anh_rgb)
    ax[0, 0].set_title("Ảnh màu RGB ban đầu")
    ax[0, 1].imshow(anh_xam, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title("Ảnh chuyển sang mức xám")
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")

    ve_hist(ax[1, 0], anh_xam, 256, "black")
    ax[1, 0].set_title("Histogram ảnh xám")
    for kenh, mau, nhan in zip(range(3), ("red", "green", "blue"), ("R", "G", "B")):
        ax[1, 1].plot(histogram(anh_rgb[:, :, kenh]), color=mau, label=nhan)
    ax[1, 1].set(xlim=(0, 255), xlabel="Mức cường độ", ylabel="Số pixel",
                     title="Histogram ba kênh RGB")
    ax[1, 1].legend()
    ax[1, 1].grid(alpha=0.2)
    fig.suptitle("Ý a — Histogram trên ảnh xám và ảnh màu RGB", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT / "a_histogram_xam_rgb.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def ve_y_b(anh_xam: np.ndarray) -> None:
    """Ý b: khảo sát ảnh hưởng của số bins và số mức xám L."""
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for i, bins in enumerate(BINS):
        ve_hist(ax[0, i], anh_xam, bins, "#455a64")
        ax[0, i].set_title(f"Histogram: bins = {bins}")
    for i, L in enumerate(LEVELS):
        ax[1, i].imshow(luong_tu_hoa(anh_xam, L), cmap="gray", vmin=0, vmax=255)
        ax[1, i].set_title(f"Ảnh lượng tử hóa: L = {L}")
        ax[1, i].axis("off")
    fig.suptitle("Ý b — Ảnh hưởng của bins và số mức xám L", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT / "b_anh_huong_bins_L.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def ve_luoc_do() -> None:
    """Tạo lược đồ tóm tắt các bước xử lý của ý a và ý b."""
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def hop(x, y, w, h, text, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor="#263238", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    def mui_ten(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#37474f"))

    hop(0.3, 2.3, 2.0, 1.1, "Ảnh màu RGB", "#e3f2fd")
    hop(3.0, 2.3, 2.3, 1.1, "Đọc ảnh\nTách R, G, B", "#e8f5e9")
    hop(6.2, 3.8, 3.0, 1.3, "Ý a\nChuyển ảnh xám\nTính Histogram xám/RGB", "#fff3e0")
    hop(6.2, 0.7, 3.0, 1.3, "Ý b\nBins: 16, 64, 256\nL: 16, 64, 256", "#f3e5f5")
    hop(10.3, 2.3, 2.2, 1.1, "Vẽ biểu đồ\nvà so sánh", "#e0f7fa")
    hop(13.1, 2.3, 1.6, 1.1, "Lưu ảnh\nPNG", "#fce4ec")
    mui_ten(2.3, 2.85, 3.0, 2.85)
    mui_ten(5.3, 2.85, 6.2, 4.45)
    mui_ten(5.3, 2.85, 6.2, 1.35)
    mui_ten(9.2, 4.45, 10.3, 3.1)
    mui_ten(9.2, 1.35, 10.3, 2.6)
    mui_ten(12.5, 2.85, 13.1, 2.85)
    ax.set_title("Lược đồ xử lý Histogram — Bài 1, ý a và ý b", fontsize=17, pad=15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "luoc_do_xu_ly.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not INPUT.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {INPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    anh_rgb = np.asarray(Image.open(INPUT).convert("RGB"), dtype=np.uint8)
    anh_xam = rgb_sang_xam(anh_rgb)
    Image.fromarray(anh_xam).save(OUTPUT / "anh_xam.png")
    ve_y_a(anh_rgb, anh_xam)
    ve_y_b(anh_xam)
    ve_luoc_do()
    print(f"Đã lưu kết quả ý a, b tại: {OUTPUT}")


if __name__ == "__main__":
    main()
