from pdf2image import convert_from_path
import os
import re

# 📂 Thư mục gốc chứa các file PDF của FinTabNet
root_pdf_dir = r"D:/intern/fintabnet/fintabnet/pdf"
# 📂 Nơi lưu ảnh đầu ra
output_dir = r"E:/intern/images_dpi300"
os.makedirs(output_dir, exist_ok=True)

# 🌀 Duyệt toàn bộ cây thư mục PDF
for root, dirs, files in os.walk(root_pdf_dir):
    for file in files:
        if file.endswith(".pdf"):
            match = re.search(r"page_(\d+)\.pdf", file)
            if not match:
                continue
            page_num = match.group(1)
            pdf_path = os.path.join(root, file)

            # 🔍 Tách thông tin BEN và 2012 từ đường dẫn
            parts = os.path.normpath(pdf_path).split(os.sep)
            try:
                source = parts[-3]  # VD: BEN
                year = parts[-2]    # VD: 2012
            except IndexError:
                print(f"⚠️ Không đủ cấp thư mục cho file: {pdf_path}")
                continue

            output_filename = f"{source}_{year}_page_{page_num}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            try:
                images = convert_from_path(pdf_path, dpi=300)
                if images:
                    images[0].save(output_path, "JPEG")
                    print(f"✅ Saved: {output_path}")
            except Exception as e:
                print(f"❌ Error with {pdf_path}: {e}")







