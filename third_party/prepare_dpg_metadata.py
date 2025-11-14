import json
import os

input_dir = "third_party/dpg_bench/prompts"
output_path = "third_party/dpg_bench/dpg_metadata.jsonl"

with open(output_path, "w", encoding="utf-8") as out_f:
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            item_id = os.path.splitext(filename)[0]

            item = {
                "item_id": item_id,
                "text": text
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"finish writing {output_path}, len={len(os.listdir(input_dir))}")