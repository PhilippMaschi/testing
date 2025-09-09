import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from matplotlib.path import Path as mpl_Path

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

def visualize_pca_comparison(original_df, synthetic_df, n_components=2):
    """
    Perform PCA reduction to 2 dimensions on both dataframes and visualize the results.
    
    Args:
        original_df: Original dataframe
        synthetic_df: Synthetic dataframe
        n_components: Number of PCA components (default: 2)
    """
    # Make sure we're working with numeric data only
    # Reset index if it's a datetime to avoid issues with StandardScaler
    if isinstance(original_df.index, pd.DatetimeIndex):
        original_df_numeric = original_df.reset_index(drop=True)
    else:
        original_df_numeric = original_df
        
    # Handle synthetic data similarly
    if isinstance(synthetic_df.index, pd.DatetimeIndex):
        synthetic_df_numeric = synthetic_df.reset_index(drop=True)
    else:
        synthetic_df_numeric = synthetic_df
    
    # Make sure both dataframes have the same number of features
    min_cols = min(original_df_numeric.shape[1], synthetic_df_numeric.shape[1])
    original_data = original_df_numeric.iloc[:, :min_cols].values
    synthetic_data = synthetic_df_numeric.iloc[:, :min_cols].values
    
    # Standardize the data
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original_data)
    # Use the same scaler for synthetic data to ensure consistency
    synthetic_scaled = scaler.transform(synthetic_data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    original_pca = pca.fit_transform(original_scaled)
    # Use the same PCA model for synthetic data
    synthetic_pca = pca.transform(synthetic_scaled)
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Plot the original data
    plt.scatter(original_pca[:, 0], original_pca[:, 1], alpha=0.5, label='Original', color='blue')
    
    # Plot the synthetic data
    plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.5, label='Synthetic', color='red')
    
    # Plot centroids
    original_centroid = np.mean(original_pca, axis=0)
    synthetic_centroid = np.mean(synthetic_pca, axis=0)
    
    plt.scatter(original_centroid[0], original_centroid[1], color='navy', 
                s=200, marker='*', label='Original Centroid')
    plt.scatter(synthetic_centroid[0], synthetic_centroid[1], color='darkred', 
                s=200, marker='*', label='Synthetic Centroid')
    
    # Draw convex hulls
    def plot_hull(points, color, alpha=0.2):
        if len(points) > 3:  # Need at least 4 points for ConvexHull
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=alpha, color=color)
    
    plot_hull(original_pca, 'blue')
    plot_hull(synthetic_pca, 'red')
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variance = np.sum(explained_variance)
    
    # Add labels and title
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)', fontsize=14)
    plt.title(f'PCA Comparison of Original vs Synthetic Data\nTotal Explained Variance: {total_variance*100:.2f}%', 
              fontsize=16)
    
    # Add a legend
    plt.legend(fontsize=14, loc='best')
    
    # Add a grid for better readability
    plt.grid(alpha=0.3)
    
    # Calculate and display distance between centroids
    centroid_distance = np.linalg.norm(original_centroid - synthetic_centroid)
    plt.annotate(f'Centroid Distance: {centroid_distance:.4f}',
                 xy=(0.05, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                 fontsize=12)
    
    # Calculate and display overlap percentage
    # (approximated by the ratio of points from one dataset that fall within the hull of the other)
    try:
        def points_in_hull(points, hull_points):
            hull_path = mpl_Path(hull_points)
            return hull_path.contains_points(points).sum() / len(points)
        
        original_hull = ConvexHull(original_pca)
        synthetic_hull = ConvexHull(synthetic_pca)
        
        original_in_synthetic = points_in_hull(original_pca, synthetic_pca[synthetic_hull.vertices])
        synthetic_in_original = points_in_hull(synthetic_pca, original_pca[original_hull.vertices])
        
        overlap = (original_in_synthetic + synthetic_in_original) / 2
        
        plt.annotate(f'Approx. Overlap: {overlap*100:.2f}%',
                     xy=(0.05, 0.10), xycoords='axes fraction',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                     fontsize=12)
    except Exception as e:
        print(f"Could not calculate overlap: {e}")
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('pca_comparison.png', dpi=300, bbox_inches='tight')
    
    return plt

def plot_cdf_comparison(original_df, synthetic_df, num_points=1000, save_path=None):
    """
    Calculate and plot the Cumulative Distribution Function (CDF) for both datasets.
    
    Args:
        original_df: Original dataframe
        synthetic_df: Synthetic dataframe
        num_points: Number of points to evaluate the CDF (default: 1000)
        save_path: Path to save the plot (default: None)
    
    Returns:
        plt: Matplotlib figure with the CDF comparison
    """
    # Make sure we're working with numeric data only
    # Convert columns to numeric, errors='coerce' will convert non-numeric values to NaN
    original_numeric = original_df.apply(pd.to_numeric, errors='coerce')
    synthetic_numeric = synthetic_df.apply(pd.to_numeric, errors='coerce')
    
    # Flatten the data to 1D arrays
    original_flat = original_numeric.values.flatten()
    synthetic_flat = synthetic_numeric.values.flatten()
    
    # Remove NaN values
    original_flat = original_flat[~np.isnan(original_flat)]
    synthetic_flat = synthetic_flat[~np.isnan(synthetic_flat)]
    
    # Make sure we have values to work with
    if len(original_flat) == 0 or len(synthetic_flat) == 0:
        print("Warning: One or both datasets have no valid numeric values after conversion")
        return None
    
    # Calculate min and max values across both datasets
    min_val = min(np.min(original_flat), np.min(synthetic_flat))
    max_val = max(np.max(original_flat), np.max(synthetic_flat))
    
    # Create evenly spaced points to evaluate the CDF
    x_points = np.linspace(min_val, max_val, num_points)
    
    # Calculate CDFs
    original_cdf = np.array([np.sum(original_flat <= x) / len(original_flat) for x in x_points])
    synthetic_cdf = np.array([np.sum(synthetic_flat <= x) / len(synthetic_flat) for x in x_points])
    
    # Plot the CDFs
    plt.figure(figsize=(12, 8))
    plt.plot(x_points, original_cdf, label='Original Data', color='blue', linewidth=2)
    plt.plot(x_points, synthetic_cdf, label='Synthetic Data', color='red', linewidth=2)
    
    # Calculate and plot the absolute difference
    cdf_diff = np.abs(original_cdf - synthetic_cdf)
    plt.plot(x_points, cdf_diff, label='Absolute Difference', color='green', linestyle='--', alpha=0.7)
    
    # Calculate KS statistic (maximum difference)
    ks_stat = np.max(cdf_diff)
    
    # Add labels and title
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.title('Cumulative Distribution Function Comparison', fontsize=16)
    
    # Add grid and legend
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)
    
    # Add KS statistic annotation
    plt.annotate(f'KS Statistic: {ks_stat:.4f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                 fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def calculate_percentiles(original_df, synthetic_df, percentiles=None):
    """
    Calculate and compare percentiles between original and synthetic data.
    
    Args:
        original_df: Original dataframe
        synthetic_df: Synthetic dataframe
        percentiles: List of percentiles to calculate (default: None, uses standard percentiles)
    
    Returns:
        pd.DataFrame: Dataframe with percentile comparisons
    """
    if percentiles is None:
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    
    # Convert to numeric values, coercing non-numeric values to NaN
    original_numeric = original_df.apply(pd.to_numeric, errors='coerce')
    synthetic_numeric = synthetic_df.apply(pd.to_numeric, errors='coerce')
    
    # Flatten the data
    original_flat = original_numeric.values.flatten()
    synthetic_flat = synthetic_numeric.values.flatten()
    
    # Remove NaN values
    original_flat = original_flat[~np.isnan(original_flat)]
    synthetic_flat = synthetic_flat[~np.isnan(synthetic_flat)]
    
    # Check if we have data to work with
    if len(original_flat) == 0 or len(synthetic_flat) == 0:
        print("Warning: One or both datasets have no valid numeric values after conversion")
        return None
    
    # Calculate percentiles
    original_percentiles = np.percentile(original_flat, percentiles)
    synthetic_percentiles = np.percentile(synthetic_flat, percentiles)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Percentile': percentiles,
        'Original': original_percentiles,
        'Synthetic': synthetic_percentiles,
        'Absolute Difference': np.abs(original_percentiles - synthetic_percentiles),
        'Relative Difference (%)': np.abs(original_percentiles - synthetic_percentiles) / np.abs(original_percentiles) * 100
    })
    
    # Format the table
    comparison = comparison.round(4)
    
    return comparison

if __name__ == "__main__":
    # Load the data
    INPUT_PATH = Path(__file__).parent.parent.parent / "GAN" / 'data' / "raw_data" / "enercoop" / "ENERCOOP_1year_filtered.csv"
    inputFile = pd.read_csv(INPUT_PATH, sep=get_sep(INPUT_PATH))
    inputFile = inputFile.set_index(inputFile.columns[0])
    inputFile.index = pd.to_datetime(inputFile.index, format='mixed')

    epoch = 964
    synth_Path = Path(__file__).parent.parent.parent / "GAN" / "runs" / "ENERCOOP_GAN" / "gute_ergbnisse_bis_1000_epochen" / "sample_data" / f"epoch_{epoch}" / "example_synth_profiles.npy"
    synth_profiles = pd.DataFrame(np.load(synth_Path, allow_pickle=True))
    synth_profiles = synth_profiles.set_index(synth_profiles.columns[0])
    synth_profiles.index = pd.to_datetime(inputFile.index, format='mixed')
    
    # Visualize PCA
    print("Creating PCA visualization...")
    pca_plot = visualize_pca_comparison(inputFile, synth_profiles)
    plt.show()
    
    # Plot CDF comparison
    print("Creating CDF comparison...")
    cdf_plot = plot_cdf_comparison(inputFile, synth_profiles, save_path='cdf_comparison.png')
    plt.show()
    
    # Calculate and display percentile comparison
    print("Calculating percentile comparison...")
    percentile_comparison = calculate_percentiles(inputFile, synth_profiles)
    print(percentile_comparison) 