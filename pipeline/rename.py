import os

def rename_files(directory):
    files = os.listdir(directory)
    files.sort()
    
    for i, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):
            new_filename = f"{i}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)

# Example usage
directory_path = '../../frame_extraction/test/polyps/'
rename_files(directory_path)

