import os
from pathlib import Path
from tqdm import tqdm


def count_folders_with_log_files(target_dir: Path):
    """
    Counts the number of subdirectories in a target directory that contain at least one .log file.

    Args:
        target_dir (Path): The directory to scan.
    """
    if not target_dir.is_dir():
        print(f"Error: Directory not found at '{target_dir}'")
        return

    print(f"Scanning subdirectories in: {target_dir}\n")

    # Get a list of all items in the target directory that are themselves directories
    subdirectories = [d for d in target_dir.iterdir() if d.is_dir()]

    if not subdirectories:
        print("No subdirectories found.")
        return

    count = 0
    folders_with_logs = []

    # Use tqdm for a progress bar
    for subdir in tqdm(subdirectories, desc="Checking folders"):
        # Check if any file inside the subdirectory ends with .log
        has_log_file = any(file.suffix == '.log' for file in subdir.iterdir() if file.is_file())

        if has_log_file:
            count += 1
            folders_with_logs.append(subdir.name)

    print("\n--- Scan Complete ---")
    print(f"Total number of subdirectories checked: {len(subdirectories)}")
    print(f"Number of subdirectories containing at least one .log file: {count}")

    # Optionally print the list of folders found
    if folders_with_logs:
        print("\nList of folders with .log files (first 20):")
        for folder_name in folders_with_logs[:20]:
            print(f"- {folder_name}")
        if len(folders_with_logs) > 20:
            print(f"... and {len(folders_with_logs) - 20} more.")


if __name__ == "__main__":
    # Define the target directory
    PREDOCKING_DIR = Path("/home/zdy/Project2/data/processed_data/predocking/")

    count_folders_with_log_files(PREDOCKING_DIR)