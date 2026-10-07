"""Bài 3: tự xây dựng Sobel và Laplace trong miền không gian."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT.parent / "noisy_image.jpg"
SOBEL_WEIGHTS = (1, 2, 4)
GAUSSIAN_3X3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16
LAPLACE_MASKS = {
    "laplace_4_lang_gieng": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "laplace_8_lang_gieng": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], float),
    "laplace_co_trong_so": np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]], float),
}


def rgb_sang_xam(anh: np.ndarray) -> np.ndarray:
    """Độ chói Y = 0.299R + 0.587G + 0.114B."""
    return np.clip(np.rint(anh @ [0.299, 0.587, 0.114]), 0, 255).astype(np.uint8)


def tich_chap(anh: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Trượt kernel trên ảnh; vùng biên được đệm theo kiểu phản xạ."""
    py, px = kernel.shape[0] // 2, kernel.shape[1] // 2
    anh_dem = np.pad(anh.astype(float), ((py, py), (px, px)), mode="reflect")
    cua_so = sliding_window_view(anh_dem, kernel.shape)
    return np.einsum("ijkl,kl->ij", cua_so, kernel, optimize=True)


def chuan_hoa_hien_thi(dap_ung: np.ndarray) -> np.ndarray:
    """Đưa 99,5% đáp ứng về 0..255 để cạnh dễ quan sát."""
    tri_tuyet_doi = np.abs(dap_ung)
    moc = np.percentile(tri_tuyet_doi, 99.5)
    if moc == 0:
        return np.zeros(dap_ung.shape, dtype=np.uint8)
    return np.clip(np.rint(tri_tuyet_doi * 255 / moc), 0, 255).astype(np.uint8)


def sobel(anh: np.ndarray, w: int, nguong: int) -> dict[str, np.ndarray]:
    """Tính Gx, Gy, độ lớn gradient và ảnh cạnh nhị phân."""
    gx_mask = np.array([[-1, 0, 1], [-w, 0, w], [-1, 0, 1]], float)
    gy_mask = np.array([[-1, -w, -1], [0, 0, 0], [1, w, 1]], float)
    gx, gy = tich_chap(anh, gx_mask), tich_chap(anh, gy_mask)
    do_lon = np.hypot(gx, gy)

    # Chuẩn hóa theo đáp ứng cực đại lý thuyết của mặt nạ Sobel.
    do_lon_chuan = np.clip(np.rint(do_lon / (np.sqrt(2) * (w + 2))), 0, 255)
    nhi_phan = np.where(do_lon_chuan >= nguong, 255, 0).astype(np.uint8)
    return {
        "gx": chuan_hoa_hien_thi(gx),
        "gy": chuan_hoa_hien_thi(gy),
        "magnitude": chuan_hoa_hien_thi(do_lon),
        "binary": nhi_phan,
    }


def laplace(anh: np.ndarray, mask: np.ndarray, nguong: int) -> dict[str, np.ndarray]:
    """Tính trị tuyệt đối đạo hàm bậc hai và phân ngưỡng tạo cạnh."""
    dap_ung = tich_chap(anh, mask)
    chuan = np.clip(np.rint(np.abs(dap_ung) / mask[mask > 0].sum()), 0, 255)
    return {
        "response": chuan_hoa_hien_thi(dap_ung),
        "binary": np.where(chuan >= nguong, 255, 0).astype(np.uint8),
    }


def luu_anh(anh: np.ndarray, duong_dan: Path) -> None:
    Image.fromarray(anh.astype(np.uint8), mode="L").save(duong_dan)


def ve_luoi(cac_anh, tieu_de, kich_thuoc, ten_hinh: str, duong_dan: Path) -> None:
    """Hàm chung để lưu các bảng ảnh so sánh."""
    fig, axes = plt.subplots(*kich_thuoc, figsize=(4.2 * kich_thuoc[1], 4.5 * kich_thuoc[0]))
    for ax, anh, ten in zip(np.asarray(axes).reshape(-1), cac_anh, tieu_de):
        ax.imshow(anh, cmap="gray", vmin=0, vmax=255)
        ax.set_title(ten)
        ax.axis("off")
    fig.suptitle(ten_hinh, fontsize=17)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tách cạnh Sobel và Laplace")
    parser.add_argument("--input", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sobel-threshold", type=int, default=40)
    parser.add_argument("--laplace-threshold", type=int, default=15)
    args = parser.parse_args()
    if not 0 <= args.sobel_threshold <= 255 or not 0 <= args.laplace_threshold <= 255:
        parser.error("Ngưỡng phải nằm trong khoảng 0..255")

    dau_vao = args.input.resolve()
    if not dau_vao.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {dau_vao}")
    if args.output:
        output = args.output.resolve()
    elif dau_vao == DEFAULT_IMAGE.resolve():
        output = ROOT / "output_bai3"
    else:
        output = ROOT / f"output_bai3_{dau_vao.stem}"
    output.mkdir(parents=True, exist_ok=True)

    xam = rgb_sang_xam(np.asarray(Image.open(dau_vao).convert("RGB")))
    # Tiền lọc để Laplace bớt nhạy với nhiễu.
    tien_loc = np.clip(np.rint(tich_chap(xam, GAUSSIAN_3X3)), 0, 255).astype(np.uint8)
    luu_anh(xam, output / "anh_xam_dau_vao.png")
    luu_anh(tien_loc, output / "anh_tien_loc_gaussian_3x3.png")

    sobels = {w: sobel(tien_loc, w, args.sobel_threshold) for w in SOBEL_WEIGHTS}
    for w, ket_qua in sobels.items():
        luu_anh(ket_qua["magnitude"], output / f"sobel_magnitude_w{w}.png")
        luu_anh(ket_qua["binary"], output / f"sobel_binary_w{w}.png")

    laplaces = {ten: laplace(tien_loc, mask, args.laplace_threshold)
                for ten, mask in LAPLACE_MASKS.items()}
    for ten, ket_qua in laplaces.items():
        luu_anh(ket_qua["response"], output / f"{ten}_response.png")
        luu_anh(ket_qua["binary"], output / f"{ten}_binary.png")

    chuan = sobels[2]
    ve_luoi((tien_loc, chuan["gx"], chuan["gy"], chuan["magnitude"], chuan["binary"]),
            ("Ảnh đã tiền lọc", "|Gx|", "|Gy|", "Sobel magnitude", "Sobel binary"), (1, 5),
            "Toán tử Sobel chuẩn — w = 2", output / "sobel_chuan_gx_gy.png")

    ve_luoi(tuple(sobels[w][loai] for loai in ("magnitude", "binary") for w in SOBEL_WEIGHTS),
            tuple(f"{loai.title()} — w={w}" for loai in ("magnitude", "binary") for w in SOBEL_WEIGHTS),
            (2, 3), "Ảnh hưởng của trọng số Sobel", output / "so_sanh_sobel_trong_so.png")

    ten_laplace = ("4-láng giềng", "8-láng giềng", "Có trọng số")
    ve_luoi(tuple(laplaces[ten][loai] for loai in ("response", "binary") for ten in LAPLACE_MASKS),
            tuple(f"{loai.title()} — {nhan}" for loai in ("response", "binary") for nhan in ten_laplace),
            (2, 3), "Ảnh hưởng của mặt nạ Laplace", output / "so_sanh_laplace_mat_na.png")

    ve_luoi((tien_loc, chuan["magnitude"], chuan["binary"],
             laplaces["laplace_4_lang_gieng"]["response"],
             laplaces["laplace_8_lang_gieng"]["response"]),
            ("Ảnh đã tiền lọc", "Sobel w=2", "Cạnh Sobel",
             "Laplace 4-láng giềng", "Laplace 8-láng giềng"), (1, 5),
            "So sánh tách cạnh Sobel và Laplace", output / "so_sanh_tong_hop.png")
    print(f"Ảnh đầu vào: {dau_vao}\nĐã lưu toàn bộ ảnh kết quả tại: {output}")


if __name__ == "__main__":
    main()
