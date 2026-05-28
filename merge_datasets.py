import csv, os
from collections import Counter

GDRBENCH_ROOT = "data/gdrbench/images"
OUTPUT_CSV    = "data/merged.csv"

os.makedirs(GDRBENCH_ROOT, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

LABEL_MAP = {
    "nodr": 0, "mild_npdr": 1,
    "moderate_npdr": 2, "severe_npdr": 3, "pdr": 4
}

SOURCES = ["APTOS", "DeepDR", "IDRiD", "RLDR", "DDR", "EYEPACS"]

rows = []

for source in SOURCES:
    source_dir = os.path.join(GDRBENCH_ROOT, source)
    if not os.path.isdir(source_dir):
        print(f"  Skipping {source} — not found at {source_dir}")
        continue
    for grade_folder, grade_label in LABEL_MAP.items():
        folder = os.path.join(source_dir, grade_folder)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.abspath(os.path.join(folder, fname))
                rows.append([full_path, grade_label, source.lower()])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "grade", "source"])
    writer.writerows(rows)

grades  = [r[1] for r in rows]
sources = [r[2] for r in rows]
print(f"\nTotal: {len(rows)} images")
print("By grade:",  dict(sorted(Counter(grades).items())))
print("By source:", dict(sorted(Counter(sources).items())))
