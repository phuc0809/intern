import os
import pandas as pd
from PIL import Image
import pytesseract
import ast
import cv2
import numpy as np

# Cấu hình pytesseract trên Windows
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Đường dẫn
csv_path = r"D:/intern/extracted_data.csv"
crop_dir = r"E:/intern/crop_cell_dpi300"
output_csv = r"D:/intern/cell_with_ocr.csv"

# Đọc dữ liệu
df = pd.read_csv(csv_path)

# Thêm chỉ số cell theo từng bảng để khớp tên file ảnh
df["cell_index_in_table"] = df.groupby("table_ID").cumcount()

# Hàm tiền xử lý ảnh
def preprocess_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize nếu quá nhỏ
    h, w = thresh.shape
    if h < 100 or w < 100:
        thresh = cv2.resize(thresh, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    return Image.fromarray(thresh)

ocr_results = []

# Duyệt từng dòng
for i, row in df.iterrows():
    bbox = row.get("boundingbox", None)

    if pd.isna(bbox) or bbox.strip() == "":
        print(f"[{i}] ❌ Bỏ qua vì thiếu bbox")
        ocr_results.append("")
        continue

    try:
        # Tạo tên ảnh đúng với logic crop cell
        path = row["path"].replace("/", "_").replace(".pdf", "")
        table_id = str(row["table_ID"])
        cell_index = row["cell_index_in_table"]
        filename = f"{path}_table_{table_id}_cell_{cell_index}.jpg"
        image_path = os.path.join(crop_dir, filename)

        if os.path.exists(image_path):
            print(f"[{i}] 🔍 Đang OCR: {filename}")
            img = Image.open(image_path)
            img = preprocess_image(img)

            text = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 7").strip()
            print(f"[{i}] ✅ Kết quả: {text}")
            ocr_results.append(text)
        else:
            print(f"[{i}] ❌ Không tìm thấy ảnh: {filename}")
            ocr_results.append("[NOT FOUND]")

    except Exception as e:
        print(f"[{i}] ❌ Lỗi: {e}")
        ocr_results.append(f"[ERROR: {e}]")

# Ghi kết quả ra file
df["pytesseract_OCR"] = ocr_results
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print("✅ Đã lưu kết quả vào:", output_csv)





