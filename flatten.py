#! /usr/bin/env python3
import os
import sys

def consolidate_py_files(output_filename="all_py_files.txt"):
    """
    Traverses the current directory and all subdirectories, finds all .py files,
    and writes their contents to a single output text file.
    """
    # 1. Open the output file for writing
    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            print(f"Starting consolidation. All contents will be written to **{output_filename}**.")

            # 2. Walk through the directory tree
            # os.walk yields (dirpath, dirnames, filenames)
            for dirpath, dirnames, filenames in os.walk('.'):
                # 3. Process each file in the current directory
                for filename in filenames:
                    # Check if the file is a Python file
                    if filename.endswith('.py'):
                        # Construct the full relative path
                        full_path = os.path.join(dirpath, filename)

                        # Skip the consolidation script itself if it's found
                        if os.path.abspath(full_path) == os.path.abspath(sys.argv[0]):
                            print(f"Skipping the script file itself: {full_path}")
                            continue

                        print(f"Processing: {full_path}")

                        # 4. Write the file header to the output file
                        outfile.write(f"{full_path}:\n")
                        
                        # 5. Read the content of the .py file and write it to the output
                        try:
                            # Use 'r' for reading
                            with open(full_path, 'r', encoding='utf-8') as infile:
                                content = infile.read()
                                outfile.write(content)
                            
                            # Add a separator for clarity between files
                            outfile.write("\n\n" + "-"*80 + "\n\n")

                        except IOError as e:
                            # Handle cases where the file might be unreadable (e.g., permissions)
                            print(f"**WARNING:** Could not read file {full_path}. Error: {e}", file=sys.stderr)
                            outfile.write(f"**[ERROR: COULD NOT READ FILE CONTENT: {e}]**\n\n")

    except IOError as e:
        print(f"**FATAL ERROR:** Could not open or write to the output file {output_filename}. Error: {e}", file=sys.stderr)
        return

    print(f"\n✅ **Consolidation complete!** Output file is **{output_filename}**.")

if __name__ == "__main__":
    consolidate_py_files()
