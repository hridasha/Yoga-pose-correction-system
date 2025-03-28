import os
import shutil

# Root directory of your project
root_dir = os.getcwd()

# Walk through the directory tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'migrations' in dirnames:
        folder_path = os.path.join(dirpath, 'migrations')
        print(f"Deleting: {folder_path}")
        shutil.rmtree(folder_path)
        
print("All migrations folders deleted!")
