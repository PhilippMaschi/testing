import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import glob

def convert_latest_numpy_to_csv(base_path):
    """
    Finds the sample_data directory in the given path, identifies the 
    epoch folder with the highest number, loads the numpy file inside it,
    and saves it as a CSV in the same location.
    
    Args:
        base_path (str or Path): Path to the directory to search in
    
    Returns:
        tuple: (path to CSV file, DataFrame) or (None, None) if files not found
    """
    base_path = Path(base_path)
    print(f"Searching for sample_data in: {base_path}")
    
    # Find sample_data directory
    sample_dir = base_path / "sample_data"
    if not sample_dir.exists():
        # Try to find it in subdirectories
        sample_dirs = list(base_path.glob("**/sample_data"))
        if not sample_dirs:
            print("No sample_data directory found!")
            return None, None
        sample_dir = sample_dirs[0]
    
    print(f"Found sample_data directory: {sample_dir}")
    
    # Find all epoch directories
    epoch_dirs = list(sample_dir.glob("epoch_*"))
    if not epoch_dirs:
        print("No epoch directories found in sample_data!")
        return None, None
    
    # Extract numbers from directory names and find the highest
    def extract_epoch_number(path):
        match = re.search(r'epoch_(\d+)', str(path.name))
        return int(match.group(1)) if match else -1
    
    epoch_dirs_with_numbers = [(dir_path, extract_epoch_number(dir_path)) for dir_path in epoch_dirs]
    epoch_dirs_with_numbers = [pair for pair in epoch_dirs_with_numbers if pair[1] >= 0]
    
    if not epoch_dirs_with_numbers:
        print("No valid epoch directories found!")
        return None, None
    
    # Sort by epoch number and get the highest
    latest_epoch_dir, epoch_num = max(epoch_dirs_with_numbers, key=lambda x: x[1])
    print(f"Found latest epoch directory: {latest_epoch_dir} (epoch {epoch_num})")
    
    # Find numpy file in the latest epoch directory
    numpy_files = list(latest_epoch_dir.glob("*.npy"))
    if not numpy_files:
        print(f"No numpy files found in {latest_epoch_dir}!")
        return None, None
    
    # Load numpy file
    numpy_file = numpy_files[0]
    print(f"Found numpy file: {numpy_file}")
    
    try:
        data = np.load(numpy_file, allow_pickle=True)
        print(f"Loaded numpy data with shape: {data.shape}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Use parent folder name for CSV file
        parent_folder_name = base_path.parent.name
        csv_file = latest_epoch_dir / f"{parent_folder_name}.csv"
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        print(f"Saved data to CSV: {csv_file}")
        
        return csv_file, df
    
    except Exception as e:
        print(f"Error loading/converting numpy file: {e}")
        return None, None

def find_all_sample_data_directories(base_path):
    """
    Find all sample_data directories in the given path and its subdirectories.
    
    Args:
        base_path (str or Path): Path to the directory to search in
        
    Returns:
        list: List of Path objects to sample_data directories
    """
    base_path = Path(base_path)
    return list(base_path.glob("**/sample_data"))

if __name__ == "__main__":
    
    # Use provided path or current directory
    base_path = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/GAN/runs/WGAN_final_EV+PV/fallen-dragon-385_2025-05-17-101846469")
    
    print(f"Starting conversion for path: {base_path}")
    csv_file, df = convert_latest_numpy_to_csv(base_path)
    
    if csv_file:
        print(f"Conversion successful! CSV file saved at: {csv_file}")
        print(f"Data shape: {df.shape}")
    else:
        print("Conversion failed. Check the error messages above.")
