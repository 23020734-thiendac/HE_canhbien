# Xử lý ảnh: Histogram, lọc nhiễu và tách biên

Repository gồm mã nguồn Python và ảnh đầu vào cho ba bài thực hành:

1. Xây dựng Histogram và Histogram Equalization (HE).
2. Lọc nhiễu bằng Median và Gaussian.
3. Tách cạnh bằng Sobel và Laplace trong miền không gian.

Các thuật toán chính được xây dựng bằng NumPy để thể hiện rõ công thức và quá trình xử lý. Chương trình không gọi trực tiếp các hàm `calcHist`, `equalizeHist`, `medianBlur`, `GaussianBlur`, `Sobel` hoặc `Laplacian` của OpenCV.

## Cấu trúc thư mục

```text
HE_canhbien/
├── README.md
├── sang.png                    # Ảnh sáng dùng cho Bài 1
├── toi.png                     # Ảnh tối dùng cho Bài 1
├── anh_mauRBG.png              # Ảnh màu hai con vẹt dùng cho Bài 1 ý a-b
├── cat.jpg                     # Ảnh nhiễu muối tiêu dùng cho Bài 2
├── house.jpg                   # Ảnh nhiễu Gauss dùng cho Bài 2
├── noisy_image.jpg             # Ảnh đầu vào dùng cho Bài 3
└── tuan3/
    ├── bai1_histogram.py
    ├── bai1_histogram_rgb.py
    ├── bai2_loc_nhieu.py
    └── bai3_tach_canh.py
```

Repository chỉ lưu mã nguồn và ảnh đầu vào. Ảnh kết quả và tài liệu báo cáo không được đưa lên GitHub.
Khi chạy, các chương trình chỉ tạo ảnh kết quả định dạng PNG, không tạo bảng CSV hay Excel.

## Yêu cầu

- Python 3.10 trở lên.
- NumPy.
- Pillow.
- Matplotlib.

Cài thư viện bằng lệnh:

```bash
python -m pip install numpy pillow matplotlib
```

Sau khi tải repository, mở Terminal tại thư mục `HE_canhbien` rồi thực hiện các lệnh bên dưới.

## Bài 1 – Histogram và Histogram Equalization

Chạy chương trình:

```bash
python tuan3/bai1_histogram.py
```

Chương trình xử lý hai ảnh `sang.png` và `toi.png`, bao gồm:

- Chuyển ảnh RGB sang ảnh xám.
- Tự tính Histogram với số `bins` bằng 16, 64 và 256.
- Khảo sát số mức xám `L` bằng 16, 64 và 256.
- Tính PDF và CDF.
- Cân bằng Histogram toàn cục.
- Cân bằng Histogram cục bộ với cửa sổ 3×3 và 5×5.

Histogram tại mức xám `k` được tính theo:

```text
h(k) = tổng của δ(I(x,y) - k)
```

Trong đó `δ = 1` nếu điểm ảnh có mức xám bằng `k`, ngược lại `δ = 0`.

Kết quả được tạo tự động trong thư mục `tuan3/output_bai1/`.

### Bài 1 với ảnh màu RGB – ý a và ý b

Chạy chương trình:

```bash
python tuan3/bai1_histogram_rgb.py
```

Chương trình xử lý ảnh `anh_mauRBG.png`, bao gồm:

- Chuyển ảnh RGB sang ảnh xám theo công thức độ chói.
- Tự tính Histogram ảnh xám và Histogram riêng của ba kênh R, G, B.
- So sánh Histogram với `bins = 16, 64, 256`.
- So sánh ảnh lượng tử hóa với `L = 16, 64, 256`.
- Tạo lược đồ tóm tắt các bước xử lý.

Kết quả được tạo trong thư mục `tuan3/output_bai1_rgb/`.

## Bài 2 – Lọc nhiễu Median và Gaussian

Chạy chương trình:

```bash
python tuan3/bai2_loc_nhieu.py
```

Chương trình thực hiện:

- Median 3×3 và 5×5 trên ảnh `cat.jpg` chứa nhiễu muối tiêu.
- Gaussian 3×3 (`sigma = 1.0`) và 5×5 (`sigma = 1.4`) trên ảnh `house.jpg` chứa nhiễu Gauss.
- Xuất các ảnh riêng và ảnh tổng hợp để so sánh kết quả.

Kết quả được tạo tự động trong thư mục `tuan3/output_bai2/`.

## Bài 3 – Tách cạnh Sobel và Laplace

### Chạy với ảnh mặc định

```bash
python tuan3/bai3_tach_canh.py
```

Ảnh đầu vào mặc định là `noisy_image.jpg` tại thư mục gốc của repository.

Chương trình thực hiện:

- Tiền lọc Gaussian 3×3 để giảm nhiễu.
- Sobel với các trọng số `w = 1, 2, 4`.
- Xuất ảnh Sobel magnitude và Sobel binary.
- Laplace với mặt nạ 4-láng giềng, 8-láng giềng và mặt nạ có trọng số.
- Tạo ảnh cạnh nhị phân từ kết quả Sobel và Laplace.

Có thể chỉ định thư mục kết quả:

```bash
python tuan3/bai3_tach_canh.py --output ket_qua_bai3
```

Nếu không truyền `--output`, chương trình tự tạo thư mục kết quả theo tên ảnh đầu vào.
