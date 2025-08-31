import pandas as pd
import os
import time
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ----- Giảm tải CPU -----
torch.set_num_threads(2)

# ----- Cấu hình -----
csv_path = r"E:/intern/cell_with_easyocr6.csv"
crop_dir = r"E:/intern/crop_cell_dpi300"
output_csv = r"E:/intern/cell_with_trocr_small.csv"
force_start_from = 354095  # 🔥 Bắt đầu từ dòng này

# ----- Load TrOCR nhỏ -----
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed", use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----- Đọc CSV gốc -----
df = pd.read_csv(csv_path, encoding='cp1252')

# ----- Khôi phục nếu đã có file lưu -----
if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv, encoding='cp1252')

    if "trocr_text" in df_existing.columns:
        df["trocr_text"] = None
        min_len = min(len(df_existing), len(df))
        df.loc[:min_len - 1, "trocr_text"] = df_existing.loc[:min_len - 1, "trocr_text"].values

        # Số dòng đã có kết quả OCR (bỏ qua rỗng và [NOT FOUND])
        done_mask = df["trocr_text"].notna() & (df["trocr_text"] != "[NOT FOUND]")
        num_done = done_mask.sum()
        print(f"🔁 File output đã có khoảng {num_done} dòng OCR.")

    else:
        df["trocr_text"] = None
        num_done = 0
else:
    df["trocr_text"] = None
    num_done = 0

# ----- Ghi đè bắt đầu từ dòng mong muốn -----
num_done = force_start_from
print(f"🚀 Bắt đầu xử lý từ dòng {num_done}")

# ----- Hàm OCR -----
def ocr_task(i):
    row = df.iloc[i]
    path = str(row["path"]).replace("/", "_").replace(".pdf", "")
    table_id = str(row["table_ID"])
    cell_index = row["cell_index_in_table"]
    filename = f"{path}_table_{table_id}_cell_{cell_index}.jpg"
    image_path = os.path.join(crop_dir, filename)

    if not os.path.exists(image_path):
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((768, 384), Image.LANCZOS)

        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        torch.cuda.empty_cache()

        if not ocr_text:
            ocr_text = "[NOT FOUND]"
        return i, ocr_text, f"[✅ OCR] {filename}: {ocr_text}"
    except Exception as e:
        return i, "[NOT FOUND]", f"⚠️ Lỗi khi OCR: {filename} - {e}"

# ----- Ghi an toàn -----
def save_safely(df, path):
    temp_path = path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8', newline='') as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)

# ----- Chạy hàng loạt -----
batch_size = 100
count = 0

for i in range(num_done, len(df)):
    result = ocr_task(i)
    if result is None:
        continue

    idx, ocr_text, log = result
    df.at[idx, "trocr_text"] = ocr_text
    print(log)
    time.sleep(0.05)

    count += 1
    if count % batch_size == 0:
        save_safely(df, output_csv)
        print(f"[💾 SAFE SAVE] Đã lưu an toàn sau {count + num_done} dòng.")

# ----- Lưu cuối cùng -----
save_safely(df, output_csv)
print(f"[✅ DONE] Đã hoàn thành. Tổng dòng OCR: {count + num_done}")
print("\n📄 Kết quả cột 'trocr_text':")
print(df[["trocr_text"]])





