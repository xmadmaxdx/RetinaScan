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

# Search recursively for source folders (zip may have inner root dir)
found_any = False
rows = []

for root, dirs, files in os.walk(GDRBENCH_ROOT):
    for source in SOURCES:
        if source in dirs:
            source_dir = os.path.join(root, source)
            found_any = True
            for grade_folder, grade_label in LABEL_MAP.items():
                folder = os.path.join(source_dir, grade_folder)
                if not os.path.isdir(folder):
                    continue
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        full_path = os.path.abspath(os.path.join(folder, fname))
                        rows.append([full_path, grade_label, source.lower()])

if not found_any:
    print(f"  No source datasets found under {GDRBENCH_ROOT}")
    print(f"  Expected folders: {SOURCES}")
    try:
        actual = [d for d in os.listdir(GDRBENCH_ROOT) if os.path.isdir(os.path.join(GDRBENCH_ROOT, d))]
        print(f"  Actual subdirs: {actual}")
    except FileNotFoundError:
        print(f"  {GDRBENCH_ROOT} does not exist — run unzip first")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "grade", "source"])
    writer.writerows(rows)

grades  = [r[1] for r in rows]
sources = [r[2] for r in rows]
print(f"\nTotal: {len(rows)} images")
print("By grade:",  dict(sorted(Counter(grades).items())))
print("By source:", dict(sorted(Counter(sources).items())))
