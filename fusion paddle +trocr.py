import os
import re
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ====== CẤU HÌNH ======
csv_path       = r"E:/intern/cell_with_trocr_small.csv"
crop_dir       = r"E:/intern/crop_cell_dpi300"
output_csv     = r"E:/intern/cell_with_paddle_trocr_token.csv"
force_start_from = 0
min_conf       = 0.9          # ngưỡng confidence Paddle
batch_size     = 100

torch.set_num_threads(2)

# ====== Khởi tạo OCR ======
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    ocr_version='PP-OCRv4',
    use_space_char=True,
    show_log=False,
    use_gpu=False,
    enable_mkldnn=True,
    det_db_unclip_ratio=2.3,
    det_db_box_thresh=0.3
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").eval()
device = torch.device("cpu")
model.to(device)

# ====== Đọc CSV ======
df = pd.read_csv(csv_path, encoding="cp1252")
df["ocr_text"] = None
df["method"]   = None
df["status"]   = None   # thêm cột mới để đánh dấu

if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv, encoding="cp1252")
    min_len = min(len(df_existing), len(df))
    df.loc[:min_len - 1, "ocr_text"] = df_existing.loc[:min_len - 1, "ocr_text"].values
    df.loc[:min_len - 1, "method"]   = df_existing.loc[:min_len - 1, "method"].values
    if "status" in df_existing.columns:
        df.loc[:min_len - 1, "status"] = df_existing.loc[:min_len - 1, "status"].values
    done_mask = (df["ocr_text"].notna()) & (df["ocr_text"] != "[NOT FOUND]")
    num_done = done_mask.sum()
    print(f"🔁 Đã OCR khoảng {num_done} dòng.")
else:
    num_done = 0

num_done = force_start_from
print(f"🚀 Bắt đầu từ dòng {num_done}")

# ====== Hàm chạy TrOCR ======
def run_trocr(image):
    img = image.copy()
    img.thumbnail((768, 384), Image.LANCZOS)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=256)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ====== Tokenizer chung ======
def tokenize(text: str):
    return re.findall(r'\d+|[A-Za-z]+|[^\w\s]', text, re.UNICODE)

# ====== Fusion OCR ======
def fusion_ocr(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        np_img = np.asarray(img)

        # --- Paddle OCR ---
        result = ocr.ocr(np_img, cls=False)
        paddle_text = ""
        scores = []
        if result and result[0]:
            texts  = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]
            if texts:
                paddle_text = " ".join(texts).strip()

        # 👉 Nếu có ký hiệu $, % hoặc - → bắt buộc dùng TrOCR full
        if any(sym in paddle_text for sym in ["$", "%", "-"]):
            trocr_text = run_trocr(img)
            if trocr_text:
                return trocr_text.strip(), "TrOCR", "OK"
            else:
                return "[NOT FOUND]", "TrOCR", "Mismatch"

        # --- Nếu không có các ký hiệu đặc biệt thì dùng fusion ---
        if paddle_text and all(s >= min_conf for s in scores):
            tokens = tokenize(paddle_text)
            trocr_text = run_trocr(img)
            trocr_tokens = tokenize(trocr_text)

            fused_text = paddle_text
            for j, t in enumerate(tokens):
                if j < len(trocr_tokens):
                    if re.match(r'^\d+$', t):  # chỉ thay số
                        fused_text = re.sub(r'\b{}\b'.format(re.escape(t)), trocr_tokens[j], fused_text, count=1)

            return fused_text.strip(), "Paddle+TrOCR", "OK"

        # --- Fallback full TrOCR ---
        trocr_text = run_trocr(img)
        if trocr_text:
            return trocr_text.strip(), "TrOCR", "OK"
        else:
            return "[NOT FOUND]", "TrOCR", "Mismatch"

    except Exception as e:
        return "[NOT FOUND]", f"ERROR: {e}", "Error"

# ====== Hàm lưu an toàn ======
def save_safely(df, path):
    temp_path = path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8', newline='') as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)

# ====== Vòng lặp OCR ======
count = 0
for i in range(num_done, len(df)):
    row = df.iloc[i]
    path_name  = str(row["path"]).replace("/", "_").replace(".pdf", "")
    table_id   = str(row["table_ID"])
    cell_index = row["cell_index_in_table"]
    filename   = f"{path_name}_table_{table_id}_cell_{cell_index}.jpg"
    image_path = os.path.join(crop_dir, filename)

    if not os.path.exists(image_path):
        continue

    ocr_text, method, status = fusion_ocr(image_path)

    df.at[i, "ocr_text"] = ocr_text
    df.at[i, "method"]   = method
    df.at[i, "status"]   = status

    print(f"[{method}] {filename}: {ocr_text} ({status})")
    time.sleep(0.05)
    count += 1

    if count % batch_size == 0:
        save_safely(df, output_csv)
        print(f"[💾 SAVE] Đã lưu sau {count + num_done} dòng.")

save_safely(df, output_csv)
print(f"[✅ DONE] Tổng dòng OCR: {count + num_done}")






