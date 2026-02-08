import os
import sys

def rename_pdfs_to_mmds(directory='.'):
    """
    Renames all files with a .pdf extension in the specified directory
    to have a .mmd extension. This function handles filenames that contain spaces.

    Args:
        directory (str): The path to the directory where the files are located.
                         Defaults to the current working directory ('.').
    """
    # Resolve the absolute path of the directory for clear feedback
    abs_directory = os.path.abspath(directory)

    # Check if the specified directory exists
    if not os.path.isdir(abs_directory):
        print(f"Error: Directory '{abs_directory}' not found. Please check the path.", file=sys.stderr)
        return

    print(f"Searching for .pdf files in: '{abs_directory}'")

    renamed_count = 0
    # Iterate over all entries (files and subdirectories) in the specified directory
    for filename in os.listdir(abs_directory):
        # Construct the full path to the current file
        old_filepath = os.path.join(abs_directory, filename)

        # Check if the current entry is actually a file and its name ends with .pdf
        # We convert to lowercase for case-insensitive matching (.pdf, .PDF, etc.)
        if os.path.isfile(old_filepath) and filename.lower().endswith('.pdf'):
            # Split the filename into its base name and extension
            # Example: "My Document.pdf" -> ("My Document", ".pdf")
            base_name, _ = os.path.splitext(filename)

            # Construct the new filename with the .mmd extension
            new_filename = base_name + '.mmd'
            new_filepath = os.path.join(abs_directory, new_filename)

            try:
                # Perform the rename operation
                os.rename(old_filepath, new_filepath)
                print(f"  Renamed: '{filename}' -> '{new_filename}'")
                renamed_count += 1
            except OSError as e:
                # Catch specific OS errors (e.g., permissions, file in use)
                print(f"  Error renaming '{filename}': {e}", file=sys.stderr)
            except Exception as e:
                # Catch any other unexpected errors
                print(f"  An unexpected error occurred with '{filename}': {e}", file=sys.stderr)

    # Provide a summary of the operation
    if renamed_count == 0:
        print("\nNo .pdf files found to rename in this directory.")
    else:
        print(f"\nFinished! Successfully renamed {renamed_count} file(s).")

if __name__ == "__main__":
    # If the script is run directly, process the current directory by default.
    # You can also pass a directory path as a command-line argument.
    # Example: python your_script_name.py "path/to/your/files"
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = '.' # Current directory

    rename_pdfs_to_mmds(target_directory)
