"""Bài 1: tự xây dựng Histogram và Histogram Equalization bằng NumPy."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUTPUT = ROOT / "output_bai1"
IMAGES = {"sang": PROJECT / "sang.png", "toi": PROJECT / "toi.png"}
BINS = LEVELS = (16, 64, 256)


def rgb_sang_xam(anh: np.ndarray) -> np.ndarray:
    """Độ chói Y = 0.299R + 0.587G + 0.114B."""
    return np.clip(np.rint(anh @ [0.299, 0.587, 0.114]), 0, 255).astype(np.uint8)


def histogram(anh: np.ndarray, bins: int = 256) -> np.ndarray:
    """Đếm số pixel trong từng bin, không dùng cv2.calcHist."""
    chi_so = np.minimum(anh.astype(np.uint32).ravel() * bins // 256, bins - 1)
    return np.bincount(chi_so, minlength=bins)


def luong_tu_hoa(anh: np.ndarray, L: int) -> np.ndarray:
    """Đưa ảnh 8-bit về L mức xám rồi trải lại miền 0..255 để hiển thị."""
    muc = np.minimum(anh.astype(np.uint32) * L // 256, L - 1)
    return np.rint(muc * 255 / (L - 1)).astype(np.uint8)


def can_bang_histogram(anh: np.ndarray, L: int = 256) -> np.ndarray:
    """HE toàn cục: s(k) = (L-1) * CDF(k)."""
    cdf = np.cumsum(histogram(anh)) / anh.size
    cdf_min = cdf[cdf > 0][0]
    if np.isclose(cdf_min, 1):
        return anh.copy()
    bang_tra = np.clip(np.rint((cdf - cdf_min) / (1 - cdf_min) * (L - 1)), 0, L - 1)
    ket_qua = bang_tra[anh]
    return np.rint(ket_qua * 255 / (L - 1)).astype(np.uint8)


def can_bang_cuc_bo(anh: np.ndarray, k: int) -> np.ndarray:
    """AHE: dùng CDF của cửa sổ k×k để ánh xạ pixel ở tâm."""
    cua_so = sliding_window_view(np.pad(anh, k // 2, mode="reflect"), (k, k))
    hang = np.count_nonzero(cua_so <= anh[:, :, None, None], axis=(-2, -1))
    return np.rint(255 * hang / (k * k)).astype(np.uint8)


def ve_hist(ax, anh: np.ndarray, bins: int = 256, mau: str = "black") -> None:
    """Vẽ histogram lên một ô biểu đồ."""
    hist = histogram(anh, bins)
    x = np.linspace(0, 256, bins, endpoint=False)
    ax.bar(x, hist, width=256 / bins, align="edge", color=mau)
    ax.set(xlim=(0, 256), xlabel="Mức xám", ylabel="Số pixel")
    ax.grid(axis="y", alpha=0.2)


def ve_anh_va_hist(cac_anh, tieu_de, ten_hinh: str, duong_dan: Path) -> None:
    """Hàm chung để so sánh nhiều ảnh và histogram tương ứng."""
    fig, axes = plt.subplots(2, len(cac_anh), figsize=(5.5 * len(cac_anh), 9))
    for i, (anh, ten) in enumerate(zip(cac_anh, tieu_de)):
        axes[0, i].imshow(anh, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(ten)
        axes[0, i].axis("off")
        ve_hist(axes[1, i], anh, mau=("gray", "#1976d2", "#ef6c00")[i % 3])
        axes[1, i].set_title(f"Histogram — {ten}")
    fig.suptitle(ten_hinh, fontsize=16)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ve_y_a(ten: str, rgb: np.ndarray, xam: np.ndarray, thu_muc: Path) -> None:
    """Ý a: ảnh RGB, ảnh xám và histogram của từng kênh."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title(f"Ảnh RGB: {ten}.png")
    ax[0, 1].imshow(xam, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title("Ảnh mức xám")
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ve_hist(ax[1, 0], xam)
    ax[1, 0].set_title("Histogram ảnh xám")
    for kenh, mau, nhan in zip(range(3), "rgb", "RGB"):
        ax[1, 1].plot(histogram(rgb[:, :, kenh]), color=mau, label=nhan)
    ax[1, 1].set(xlim=(0, 255), xlabel="Mức cường độ", ylabel="Số pixel",
                     title="Histogram các kênh RGB")
    ax[1, 1].legend()
    ax[1, 1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(thu_muc / "a_histogram_xam_rgb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def ve_y_b(ten: str, xam: np.ndarray, thu_muc: Path) -> None:
    """Ý b: so sánh số bin Histogram và số mức xám L."""
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for i, bins in enumerate(BINS):
        ve_hist(ax[0, i], xam, bins, "#455a64")
        ax[0, i].set_title(f"Histogram với bins = {bins}")
    for i, L in enumerate(LEVELS):
        ax[1, i].imshow(luong_tu_hoa(xam, L), cmap="gray", vmin=0, vmax=255)
        ax[1, i].set_title(f"L = {L} mức xám")
        ax[1, i].axis("off")
    fig.suptitle(f"Ảnh hưởng của bins và L: {ten}.png", fontsize=16)
    fig.tight_layout()
    fig.savefig(thu_muc / "b_anh_huong_bins_L.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def xu_ly_anh(ten: str, duong_dan: Path) -> None:
    print(f"Đang xử lý {duong_dan.name}...")
    thu_muc = OUTPUT / ten
    thu_muc.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(duong_dan).convert("RGB"), dtype=np.uint8)
    xam = rgb_sang_xam(rgb)
    he = can_bang_histogram(xam)
    ahe3, ahe5 = can_bang_cuc_bo(xam, 3), can_bang_cuc_bo(xam, 5)

    for file, anh in {
        "anh_xam.png": xam,
        "anh_he_toan_cuc.png": he,
        "anh_ahe_3x3.png": ahe3,
        "anh_ahe_5x5.png": ahe5,
    }.items():
        Image.fromarray(anh).save(thu_muc / file)

    ve_y_a(ten, rgb, xam, thu_muc)
    ve_y_b(ten, xam, thu_muc)
    ve_anh_va_hist((xam, he), ("Ảnh xám ban đầu", "Ảnh sau HE"),
                    "Histogram Equalization toàn cục", thu_muc / "c_he_toan_cuc.png")
    he_theo_L = tuple(can_bang_histogram(xam, L) for L in LEVELS)
    ve_anh_va_hist(he_theo_L, tuple(f"HE với L = {L}" for L in LEVELS),
                    "Ảnh hưởng của L đến HE", thu_muc / "c_he_thay_doi_L.png")
    ve_anh_va_hist((xam, ahe3, ahe5), ("Ảnh xám", "AHE 3×3", "AHE 5×5"),
                    "Cân bằng Histogram cục bộ", thu_muc / "c_ahe_3x3_5x5.png")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    thieu = [str(p) for p in IMAGES.values() if not p.exists()]
    if thieu:
        raise FileNotFoundError("Không tìm thấy: " + ", ".join(thieu))
    for ten, duong_dan in IMAGES.items():
        xu_ly_anh(ten, duong_dan)
    print(f"Đã lưu toàn bộ ảnh kết quả tại: {OUTPUT}")


if __name__ == "__main__":
    main()
