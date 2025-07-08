# unzip.py
# This script unpacks the dataset from the downloaded .zip files.

import os
import zipfile
import shutil

# --- Configuration ---
# Set the path to the downloaded main zip file
# Example: "C:\\Users\\YourUser\\Downloads\\1512427.zip"
DOWNLOADED_ZIP_PATH = r"C:\Users\uysal\Downloads\1512427.zip"

# Set the destination folder where you want your project to live
# Example: "C:\\Users\\YourUser\\Desktop\\Brain_Tumour_Project"
PROJECT_FOLDER = r"C:\Users\uysal\OneDrive\Masaüstü\bt_v3"

# --- Script ---
def main():
    """Main function to orchestrate the unzipping process."""
    
    # Define key folder paths
    data_raw_folder = os.path.join(PROJECT_FOLDER, "data_raw")
    
    # Create the project and data_raw folders if they don't exist
    os.makedirs(data_raw_folder, exist_ok=True)
    
    print(f"1. Extracting main zip file: {DOWNLOADED_ZIP_PATH}")
    # Extract the main zip file (which contains other zips)
    with zipfile.ZipFile(DOWNLOADED_ZIP_PATH, 'r') as zf:
        zf.extractall(PROJECT_FOLDER)
        
    print("2. Extracting nested zip files into 'data_raw'...")
    # Find all the zip files that were just extracted
    for item in os.listdir(PROJECT_FOLDER):
        if item.endswith(".zip"):
            zip_path = os.path.join(PROJECT_FOLDER, item)
            print(f"   - Unzipping {item}...")
            # Extract each nested zip into the data_raw folder
            with zipfile.ZipFile(zip_path, 'r') as nested_zf:
                nested_zf.extractall(data_raw_folder)
            # Remove the nested zip file after extracting
            os.remove(zip_path)
            
    print("3. Moving 'cvind.mat' to the project root...")
    # The cvind.mat file contains the fold assignments for cross-validation
    cvind_source = os.path.join(data_raw_folder, "cvind.mat")
    cvind_dest = os.path.join(PROJECT_FOLDER, "cvind.mat")
    
    if os.path.exists(cvind_source):
        shutil.move(cvind_source, cvind_dest)
        
    print("\n✅ Done! Your dataset is ready in the 'data_raw' folder.")

# Run the script
if __name__ == "__main__":
    main()