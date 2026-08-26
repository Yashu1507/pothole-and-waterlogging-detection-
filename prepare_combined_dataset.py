from pathlib import Path
import shutil

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\Users\M.N THUSHAR")

POTHOLE = BASE / "pothole"
WATER = BASE / "waterlogging"

OUTPUT = WATER / "combined_dataset"

# ============================================================
# DATA SPLITS
# ============================================================

SPLITS = {
    "train": "train",
    "val": "valid",
    "test": "test",
}

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in SPLITS:
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# COPY POTHOLE DATA
# POTHOLE CLASS 0 -> COMBINED CLASS 0
# ============================================================

def copy_potholes():
    print("\n========== COPYING POTHOLES ==========")

    for split, source_split in SPLITS.items():

        image_dir = POTHOLE / source_split / "images"
        label_dir = POTHOLE / source_split / "labels"

        output_images = OUTPUT / "images" / split
        output_labels = OUTPUT / "labels" / split

        images = list(image_dir.glob("*"))

        count = 0

        for image in images:

            if not image.is_file():
                continue

            label = label_dir / f"{image.stem}.txt"

            if not label.exists():
                print("WARNING: Missing label:", image.name)
                continue

            # Prefix prevents filename collisions
            new_name = f"pothole_{image.name}"
            new_label = f"pothole_{image.stem}.txt"

            shutil.copy2(image, output_images / new_name)

            # Pothole dataset already uses class 0.
            shutil.copy2(label, output_labels / new_label)

            count += 1

        print(split, "pothole images:", count)


# ============================================================
# COPY WATERLOGGING DATA
#
# ONLY CLASS 2 IS USED.
#
# CLASS 2 -> COMBINED CLASS 1
#
# Classes 0 and 1 are deliberately ignored because they are
# inconsistently represented in train/validation/test.
# ============================================================

def copy_waterlogging():
    print("\n========== COPYING WATERLOGGING ==========")

    for split, source_split in SPLITS.items():

        image_dir = WATER / source_split / "images"
        label_dir = WATER / source_split / "labels"

        output_images = OUTPUT / "images" / split
        output_labels = OUTPUT / "labels" / split

        images = list(image_dir.glob("*"))

        count = 0

        for image in images:

            if not image.is_file():
                continue

            label = label_dir / f"{image.stem}.txt"

            if not label.exists():
                print("WARNING: Missing label:", image.name)
                continue

            new_lines = []

            for line in label.read_text().splitlines():

                parts = line.split()

                if not parts:
                    continue

                class_id = parts[0]

                # ONLY waterlogging class
                if class_id == "2":

                    # Change waterlogging:
                    # old class 2 -> new class 1
                    parts[0] = "1"

                    new_lines.append(" ".join(parts))

            # Skip images that don't contain waterlogging
            if not new_lines:
                continue

            new_name = f"water_{image.name}"
            new_label = f"water_{image.stem}.txt"

            shutil.copy2(image, output_images / new_name)

            (output_labels / new_label).write_text(
                "\n".join(new_lines) + "\n"
            )

            count += 1

        print(split, "waterlogging images:", count)


# ============================================================
# RUN
# ============================================================

copy_potholes()
copy_waterlogging()

print("\n============================================")
print("COMBINED DATASET CREATED")
print("============================================")
print("Location:")
print(OUTPUT)

print("\nClasses:")
print("0 = pothole")
print("1 = waterlogging")