import pandas as pd
import Levenshtein
import re

# ----- 1. Đọc file -----
file_path = r"D:\intern\cell_with_ocr.csv"
df = pd.read_csv(file_path)
df = df[['text_GT', 'pytesseract_OCR']].dropna()

# ----- 2. Chuẩn hóa chuỗi -----
def normalize(text):
    text = str(text)
    text = text.lower()                          # chữ thường
    text = re.sub(r'\s+', '', text)              # bỏ khoảng trắng
    text = re.sub(r'[^\w]', '', text)            # bỏ dấu câu, giữ chữ & số
    return text

# Chuẩn hóa dữ liệu
df['norm_GT'] = df['text_GT'].apply(normalize)
df['norm_OCR'] = df['pytesseract_OCR'].apply(normalize)

# ----- 3. So sánh Exact Match (gốc) -----
strict_match = df['text_GT'].str.strip() == df['pytesseract_OCR'].str.strip()
strict_match_accuracy = strict_match.sum() / len(df) * 100

# ----- 4. So sánh Normalized Match -----
normalized_match = df['norm_GT'] == df['norm_OCR']
normalized_match_accuracy = normalized_match.sum() / len(df) * 100

# ----- 5. Tính Levenshtein Accuracy -----
def normalized_lev(gt, ocr):
    dist = Levenshtein.distance(gt, ocr)
    max_len = max(len(gt), len(ocr))
    return 1 - dist / max_len if max_len > 0 else 1

df['lev_accuracy'] = df.apply(lambda row: normalized_lev(row['norm_GT'], row['norm_OCR']), axis=1)
average_lev_accuracy = df['lev_accuracy'].mean() * 100

# ----- 6. Levenshtein fuzzy match (>= 95%) -----
lev_threshold = 0.95
fuzzy_match = df['lev_accuracy'] >= lev_threshold
fuzzy_match_accuracy = fuzzy_match.sum() / len(df) * 100

# ----- 7. In kết quả -----
print(f"📌 Strict Exact Match Accuracy: {strict_match_accuracy:.2f}%")
print(f"📌 Normalized Match Accuracy  : {normalized_match_accuracy:.2f}%")
print(f"📌 Levenshtein Avg Accuracy   : {average_lev_accuracy:.2f}%")
print(f"📌 Fuzzy Match Accuracy (>=95%): {fuzzy_match_accuracy:.2f}%")
