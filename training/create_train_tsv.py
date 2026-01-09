import os
import csv
import random
import argparse


def create_train_tsv_advanced(root_folder, output_file, include_folder):
    """
    Create train.tsv file from image folders

    Args:
        root_folder: Root directory containing emotion subfolders
        output_file: Output TSV file name
        include_folder: Whether to include folder name in file path
    """
    # Emotion folders to look for
    emotion_folders = [
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
    ]

    # Supported image formats
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

    # Prepare data
    rows = []

    # Header
    header = ["file", "emotion", "valence", "arousal"]

    print(f"🔍 Looking for images in: {root_folder}")

    # Check if root folder exists
    if not os.path.exists(root_folder):
        print(f"❌ Error: The folder '{root_folder}' does not exist.")
        return False

    # Process each emotion folder
    for emotion in emotion_folders:
        emotion_path = os.path.join(root_folder, emotion)

        if not os.path.isdir(emotion_path):
            print(f"  ⚠️  Skipping '{emotion}' - folder not found")
            continue

        # Find all image files
        image_files = []
        for file in sorted(os.listdir(emotion_path)):  # Sort for consistent order
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                image_files.append(file)

        if not image_files:
            print(f"  ℹ️  No images found in '{emotion}' folder")
            continue

        print(f"  ✅ Found {len(image_files)} images in '{emotion}'")

        # Create entries for each image
        for image_file in image_files:
            # Generate random values
            valence = random.uniform(-1, 1)
            arousal = random.uniform(0, 1)

            # Format to 2 decimal places
            valence = round(valence, 2)
            arousal = round(arousal, 2)

            # Create file path
            if include_folder:
                file_path = f"{emotion}/{image_file}"
            else:
                file_path = image_file

            rows.append([file_path, emotion, valence, arousal])

    # Check if we have data
    if not rows:
        print("\n❌ No data to save. Please check:")
        print(f"  1. The '{root_folder}' folder exists")
        print(f"  2. It contains subfolders like: {', '.join(emotion_folders[:3])}...")
        print(f"  3. Those subfolders contain image files")
        return False

    # Write to TSV file
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n✅ Successfully created '{output_file}'")
    print(f"   Total entries: {len(rows)}")

    # Show statistics
    emotion_counts = {}
    for row in rows:
        emotion = row[1]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print(f"\n📊 Statistics by emotion:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"   {emotion}: {count} images")

    print(
        f"\n📁 File path format: {'emotion/filename' if include_folder else 'filename only'}"
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create train.tsv file from image folders"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="processed_images/",
        help="Input folder containing emotion subfolders (default: images/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="train.tsv",
        help="Output TSV file name (default: train.tsv)",
    )
    parser.add_argument(
        "--no-folder",
        action="store_true",
        help="Don't include folder name in file path",
    )

    args = parser.parse_args()

    create_train_tsv_advanced(
        root_folder=args.input,
        output_file=args.output,
        include_folder=not args.no_folder,
    )


if __name__ == "__main__":
    main()
