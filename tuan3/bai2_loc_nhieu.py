"""Bài 2: tự xây dựng bộ lọc Median và Gaussian bằng NumPy."""

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
OUTPUT = ROOT / "output_bai2"
CAT, HOUSE = PROJECT / "cat.jpg", PROJECT / "house.jpg"


def cac_cua_so(anh: np.ndarray, k: int) -> np.ndarray:
    """Tạo toàn bộ vùng lân cận k×k, dùng đệm phản xạ ở biên ảnh."""
    anh_dem = np.pad(anh, ((k // 2, k // 2), (k // 2, k // 2), (0, 0)), mode="reflect")
    return sliding_window_view(anh_dem, (k, k), axis=(0, 1))


def loc_median(anh: np.ndarray, k: int) -> np.ndarray:
    """Thay mỗi pixel bằng trung vị của cửa sổ k×k."""
    return np.median(cac_cua_so(anh, k), axis=(-2, -1)).astype(np.uint8)


def kernel_gaussian(k: int, sigma: float) -> np.ndarray:
    """G(x,y) = exp(-(x²+y²)/(2σ²)), sau đó chuẩn hóa tổng bằng 1."""
    toa_do = np.arange(-(k // 2), k // 2 + 1, dtype=float)
    x, y = np.meshgrid(toa_do, toa_do)
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def loc_gaussian(anh: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Tích chập từng kênh màu với kernel Gaussian tự tạo."""
    ket_qua = np.einsum("abcij,ij->abc", cac_cua_so(anh, k), kernel_gaussian(k, sigma))
    return np.clip(np.rint(ket_qua), 0, 255).astype(np.uint8)


def ve_luoi(cac_anh, tieu_de, so_hang: int, ten_hinh: str, duong_dan: Path) -> None:
    """Vẽ và lưu một bảng ảnh so sánh."""
    so_cot = len(cac_anh) // so_hang
    fig, axes = plt.subplots(so_hang, so_cot, figsize=(5.5 * so_cot, 5 * so_hang))
    for ax, anh, ten in zip(np.asarray(axes).reshape(-1), cac_anh, tieu_de):
        ax.imshow(anh)
        ax.set_title(ten)
        ax.axis("off")
    fig.suptitle(ten_hinh, fontsize=17)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for duong_dan in (CAT, HOUSE):
        if not duong_dan.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {duong_dan}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cat = np.asarray(Image.open(CAT).convert("RGB"), dtype=np.uint8)
    house = np.asarray(Image.open(HOUSE).convert("RGB"), dtype=np.uint8)
    median3, median5 = loc_median(cat, 3), loc_median(cat, 5)
    gauss3, gauss5 = loc_gaussian(house, 3, 1.0), loc_gaussian(house, 5, 1.4)

    ket_qua = {
        "cat_median_3x3.png": median3,
        "cat_median_5x5.png": median5,
        "house_gaussian_3x3.png": gauss3,
        "house_gaussian_5x5.png": gauss5,
    }
    for ten, anh in ket_qua.items():
        Image.fromarray(anh).save(OUTPUT / ten)

    ve_luoi((cat, median3, median5), ("Ảnh nhiễu", "Median 3×3", "Median 5×5"), 1,
            "Khử nhiễu muối–tiêu", OUTPUT / "so_sanh_median.png")
    ve_luoi((house, gauss3, gauss5), ("Ảnh nhiễu", "Gaussian 3×3", "Gaussian 5×5"), 1,
            "Khử nhiễu Gauss", OUTPUT / "so_sanh_gaussian.png")
    ve_luoi((cat, median3, median5, house, gauss3, gauss5),
            ("Nhiễu muối–tiêu", "Median 3×3", "Median 5×5",
             "Nhiễu Gauss", "Gaussian 3×3", "Gaussian 5×5"), 2,
            "Bài 2 — Lọc Median và Gaussian", OUTPUT / "so_sanh_tong_hop.png")
    print(f"Đã lưu toàn bộ ảnh kết quả tại: {OUTPUT}")


if __name__ == "__main__":
    main()
