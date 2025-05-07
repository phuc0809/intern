import os
import json
from PIL import Image

# === Config ===
jsonl_path = r"D:/intern/fintabnet/fintabnet/FinTabNet_1.0.0_cell_train.jsonl"
image_dir = r"E:/intern/images_dpi300"
output_dir = r"E:/intern/crop_cell_dpi300"
os.makedirs(output_dir, exist_ok=True)

original_dpi = 72
target_dpi = 300
scale = target_dpi / original_dpi  # pixels

padding_top = 15  # pixel padding (ở ảnh 300dpi)

def count_header_rows(tokens):
    row_count = 0
    inside_row = False
    is_header = True

    for token in tokens:
        if token == "<tr>":
            inside_row = True
            is_header = False
        elif token == "</tr>":
            if is_header:
                row_count += 1
            inside_row = False
        elif token == "<th>":
            is_header = True
    return row_count

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        image_name = data["filename"].replace("/", "_").replace(".pdf", "") + ".jpg"
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy ảnh: {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        tokens = data["html"]["structure"]["tokens"]
        header_rows = count_header_rows(tokens)

        num_columns = 0
        for token in tokens:
            if token == "<td>" or token == "<th>":
                num_columns += 1
            elif token == "</tr>":
                break

        table_id = data.get("table_id", 0)  # 👈 Thêm table_id nếu có

        for idx, cell in enumerate(data["html"]["cells"]):
            if "bbox" not in cell:
                continue

            try:
                x0, y0, x1, y1 = [x * scale for x in cell["bbox"]]

                # Đảo trục Y vì gốc tọa độ nằm dưới
                y0_flipped = h - y1
                y1_flipped = h - y0

                # Thêm padding top
                y0_flipped = max(0, y0_flipped - padding_top)

                cropped = img.crop((int(x0), int(y0_flipped), int(x1), int(y1_flipped)))

                # 👉 Dùng table_id để tránh trùng tên
                out_name = f"{os.path.splitext(image_name)[0]}_table_{table_id}_cell_{idx}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cropped.save(out_path)

                print(f"✅ Đã lưu: {out_path}")

            except Exception as e:
                print(f"❌ Lỗi tại cell {idx} trong {image_name}: {e}")



