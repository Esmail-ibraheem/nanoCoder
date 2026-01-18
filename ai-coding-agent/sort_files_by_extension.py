#!/usr/bin/env python3
"""
A Python script to sort files by their extensions.
This script organizes files in a directory into subdirectories based on their file extensions.
"""

import os
import shutil
import argparse
from pathlib import Path


def sort_files_by_extension(directory: str, exclude_dirs: list = None) -> None:
    """
    Sort files in the specified directory by their extensions.
    
    Args:
        directory (str): The directory path to sort files in.
        exclude_dirs (list): List of directory names to exclude from processing.
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    # Convert to Path object for easier manipulation
    dir_path = Path(directory)
    
    # Verify the directory exists
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    if not dir_path.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        return
    
    print(f"Sorting files in: {dir_path.absolute()}")
    
    # Process each item in the directory
    for item in dir_path.iterdir():
        # Skip directories (unless they're in our processing path)
        if item.is_dir():
            if item.name in exclude_dirs:
                print(f"Skipping excluded directory: {item.name}")
                continue
            # Recursively process subdirectories
            sort_files_by_extension(str(item), exclude_dirs)
            continue
        
        # Skip the script itself
        if item.name == __file__:
            continue
        
        # Get file extension
        file_ext = item.suffix.lower()
        
        # Handle files without extensions
        if not file_ext:
            target_dir = dir_path / "no_extension"
            file_ext = "no_extension"
        else:
            # Clean extension (remove the dot)
            file_ext = file_ext[1:]
            target_dir = dir_path / file_ext
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(exist_ok=True)
        
        # Move the file
        target_file = target_dir / item.name
        
        # Handle potential filename conflicts
        counter = 1
        while target_file.exists():
            target_file = target_dir / f"{item.stem}_{counter}{item.suffix}"
            counter += 1
        
        try:
            shutil.move(str(item), str(target_file))
            print(f"Moved: {item.name} -> {file_ext}/{target_file.name}")
        except Exception as e:
            print(f"Error moving {item.name}: {e}")


def main():
    """
    Main function to handle command line arguments and execute the sorting.
    """
    parser = argparse.ArgumentParser(
        description="Sort files by extension into subdirectories"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to sort (default: current directory)"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Directories to exclude from processing"
    )
    
    args = parser.parse_args()
    
    # Add common directories to exclude by default
    default_excludes = [".git", "__pycache__", "node_modules", "venv", "env"]
    exclude_dirs = list(set(default_excludes + args.exclude))
    
    print(f"Sorting files in: {args.directory}")
    print(f"Excluding directories: {', '.join(exclude_dirs) if exclude_dirs else 'None'}")
    print("-" * 50)
    
    sort_files_by_extension(args.directory, exclude_dirs)
    
    print("\nFile sorting complete!")


if __name__ == "__main__":
    main()