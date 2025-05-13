import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import csv


def get_sep(path):
    """Determines the separator used in a CSV file.

    Args:
        path (str): Path to the file.

    Returns:
        str: The separator.
    """
    with open(path, newline = '') as file:
        sep = csv.Sniffer().sniff(file.read()).delimiter
        return sep





def select_windows_and_columns(original_df, synthetic_df, window_size=168, num_windows=6, columns_per_window=3):
    """
    Select evenly spaced windows and random columns from both dataframes.
    
    Args:
        original_df: Original dataframe
        synthetic_df: Synthetic dataframe
        window_size: Size of each window (default: 168, representing one week)
        num_windows: Number of windows to select (default: 6)
        columns_per_window: Number of random columns to select for each window (default: 3)
    
    Returns:
        dict: Dictionary containing selected windows and columns for both dataframes
    """
    # Ensure both dataframes have the same dimension
    assert original_df.shape[0] == synthetic_df.shape[0], "Dataframes must have the same number of rows"
    
    # Get total length of dataframes
    total_length = original_df.shape[0]
    
    # Calculate evenly spaced window start points
    window_starts = np.linspace(0, total_length - window_size, num_windows, dtype=int)
    
    # Determine how many columns to sample
    num_columns = min(original_df.shape[1], synthetic_df.shape[1])
    total_sample_columns = min(num_columns, num_windows * columns_per_window)
    
    # Randomly select columns (same for both dataframes)
    selected_columns = np.random.choice(num_columns, total_sample_columns, replace=False)
    
    # Split selected columns into groups for each window
    column_groups = np.array_split(selected_columns, num_windows)
    
    # Create result dictionary
    result = {
        "window_indices": [],
        "column_indices": [],
        "original_data": [],
        "synthetic_data": []
    }
    
    # Extract windows and columns
    for i, start_idx in enumerate(window_starts):
        end_idx = start_idx + window_size
        window_cols = column_groups[i]
        
        result["window_indices"].append((start_idx, end_idx))
        result["column_indices"].append(window_cols)
        
        # Extract data for this window
        orig_window_data = original_df.iloc[start_idx:end_idx, window_cols]
        synth_window_data = synthetic_df.iloc[start_idx:end_idx, window_cols]
        
        result["original_data"].append(orig_window_data)
        result["synthetic_data"].append(synth_window_data)
    
    return result

def plot_selected_windows(selections):
    """
    Plot selected windows from both dataframes for comparison.
    
    Args:
        selections: Dictionary from select_windows_and_columns
    """
    num_windows = len(selections["window_indices"])
    
    fig, axes = plt.subplots(num_windows, 1, figsize=(12, 4 * num_windows))
    if num_windows == 1:
        axes = [axes]  # Make it iterable if there's only one window
    
    for i in range(num_windows):
        ax = axes[i]
        start_idx, end_idx = selections["window_indices"][i]
        orig_data = selections["original_data"][i]
        synth_data = selections["synthetic_data"][i]
        
        # Create a simple x-axis with the window size
        x_values = np.arange(len(orig_data))
        
        # Plot original data (solid lines)
        for j, col in enumerate(orig_data.columns):
            ax.plot(x_values, orig_data.iloc[:, j], label=f"Original {col}", alpha=0.7)
        
        # Plot synthetic data (dashed lines)
        for j, col in enumerate(synth_data.columns):
            ax.plot(x_values, synth_data.iloc[:, j], label=f"Synthetic {col}", 
                   linestyle='--', alpha=0.7)
        
        # Set x-axis ticks for hours
        hours_ticks = np.arange(0, len(orig_data)+1, 24)  # Every 24 hours
        ax.set_xticks(hours_ticks)
        ax.set_xticklabels([f"{h}h" for h in hours_ticks])
        
        ax.set_title(f"Window {i+1}: Time indices {start_idx}-{end_idx}")
        ax.set_xlabel("Hours")
        ax.set_ylabel("electricity consumption in kW")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_gif_from_pngs(directory_path, output_gif_path, pattern="hourly_trend.png", duration=0.5):
    """
    Create a GIF from all .png files matching a pattern in subdirectories.
    
    Args:
        directory_path: Path to the parent directory
        output_gif_path: Path where the output GIF will be saved
        pattern: Filename pattern to match (default: "hourly_trend.png")
        duration: Duration for each frame in seconds (default: 0.5)
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio")
        return
        
    # Find all PNG files matching the pattern in subdirectories
    image_paths = []
    subdirs = []
    
    # Collect all subdirectories
    for subdir in directory_path.glob("*"):
        if subdir.is_dir() and subdir.name.startswith("epoch_"):
            subdirs.append(subdir)
    
    # Sort subdirectories by epoch number
    def get_epoch_number(path):
        # Extract the epoch number from the folder name (epoch_X)
        try:
            return int(path.name.split("_")[1])
        except (IndexError, ValueError):
            return float('inf')  # Put folders that don't match the pattern at the end
    
    subdirs.sort(key=get_epoch_number)
    
    # Collect images in order
    for subdir in subdirs:
        image_path = subdir / pattern
        if image_path.exists():
            image_paths.append(image_path)
            print(f"Found image in {subdir.name}")
    
    if not image_paths:
        print(f"No images matching pattern '{pattern}' found in subdirectories of {directory_path}")
        return
    
    print(f"Found {len(image_paths)} images. Creating GIF...")
    print(f"Order of epochs: {[get_epoch_number(path.parent) for path in image_paths]}")
    
    # Read images and create a GIF
    images = [imageio.imread(str(img_path)) for img_path in image_paths]
    
    # Add frame numbers to help track progress
    for i, img in enumerate(images):
        frame_num = f"Frame {i+1}/{len(images)}"
        imageio.imwrite('temp.png', img)
        img_with_text = plt.imread('temp.png')
        fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100))
        ax.imshow(img_with_text)
        ax.text(10, 30, frame_num, fontsize=12, color='red', 
                bbox=dict(facecolor='white', alpha=0.7))
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
        images[i] = imageio.imread('temp.png')
        plt.close()
    
    # Create GIF
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_gif_path), images, duration=duration)
    
    # Clean up temporary file
    if Path('temp.png').exists():
        Path('temp.png').unlink()
    
    print(f"GIF created successfully: {output_gif_path}")



if __name__ == "__main__":
    # Usage example
    # Set random seed for reproducibility
    np.random.seed(42)

    # INPUT_PATH = Path(__file__).parent.parent.parent / "GAN" / 'data' / "raw_data" / "enercoop" / "ENERCOOP_1year_filtered.csv"
    # inputFile = pd.read_csv(INPUT_PATH, sep = get_sep(INPUT_PATH))
    # inputFile = inputFile.set_index(inputFile.columns[0])
    # inputFile.index = pd.to_datetime(inputFile.index, format = 'mixed')

    # epoch = 964
    # synth_Path = Path(__file__).parent.parent.parent  / "GAN" / "runs" / "ENERCOOP_GAN" / "gute_ergbnisse_bis_1000_epochen" / "sample_data" / f"epoch_{epoch}" / "example_synth_profiles.npy"
    # synth_profiles = pd.DataFrame(np.load(synth_Path, allow_pickle = True))

    # # Select windows and columns
    # selections = select_windows_and_columns(inputFile, synth_profiles)

    # # Plot the selected data
    # plot_selected_windows(selections)


    plot_path = Path(__file__).parent.parent.parent / "GAN" / "runs" / "ENERCOOP_GAN" / "gute_ergbnisse_bis_1000_epochen" / "plots" 

    # Create a GIF from all hourly_trend.png files
    gif_output_path = plot_path / "hourly_trends_animation.gif"

    create_gif_from_pngs(plot_path, gif_output_path, duration=250)













