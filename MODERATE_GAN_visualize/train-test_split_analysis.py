from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_overall_metrics(real_data, synthetic_data):
    """Calculate overall distribution metrics between real and synthetic data."""
    metrics = {}
    
    # Flatten all values into single arrays
    real_values = real_data.values.flatten()
    synth_values = synthetic_data.values.flatten()
    
    # Calculate basic statistics
    metrics['real_mean'] = np.mean(real_values)
    metrics['real_std'] = np.std(real_values)
    metrics['synth_mean'] = np.mean(synth_values)
    metrics['synth_std'] = np.std(synth_values)
    
    # Calculate Wasserstein distance on flattened data
    metrics['wasserstein_dist'] = stats.wasserstein_distance(real_values, synth_values)
    
    return metrics

def plot_distribution_comparison(test_data, synthetic_datasets, save_path=None):
    """Create distribution comparison plots with histograms and CDFs."""
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Flatten test data
    test_values = test_data.values.flatten()
    
    # Create figure for histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data=test_values, label='Test Data (size 261)', alpha=0.6, stat='density')
    
    # Plot histograms for each synthetic dataset
    for size, synthetic_data in synthetic_datasets.items():
        if size in [261, 261*2, 261*4, 4000]:
            synth_values = synthetic_data.values.flatten()
            sns.histplot(data=synth_values, label=f'Synthetic Data (size {size})', alpha=0.3, stat='density')
    
    plt.xlabel('electricity consumption (kWh/h)')
    plt.ylabel('probability density')
    plt.legend()
    plt.xlim(0, 2)  
    plt.ylim(0, 3)   
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()
    plt.close()
    
    # Create separate figure for CDF
    plt.figure(figsize=(8, 6))
    
    # Plot test data CDF
    x = np.sort(test_values)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label='Test Data', linewidth=2)
    
    # Plot CDFs for each synthetic dataset
    for size, synthetic_data in synthetic_datasets.items():
        if size in [261, 261*2, 261*4, 4000]:
            synth_values = synthetic_data.values.flatten()
            x = np.sort(synth_values)
            y = np.arange(1, len(x) + 1) / len(x)
            plt.plot(x, y, label=f'Synthetic Data (size {size})', alpha=0.7)
    
    plt.xlabel('electricity consumption (kWh/h)')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.xlim(0, 2.5)  # Zoom in on x-axis
    
    plt.tight_layout()
    if save_path:
        cdf_save_path = str(save_path).replace('.png', '_cdf.png')
        plt.savefig(cdf_save_path, dpi=900)
    plt.show()
    plt.close()

def plot_train_test_comparison(train_data, test_data, save_path=None):
    """Create histogram comparison plot for training and test data."""
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Flatten data
    train_values = train_data.values.flatten()
    test_values = test_data.values.flatten()
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot histograms
    sns.histplot(data=train_values, label='Training data', alpha=0.6, stat='density')
    sns.histplot(data=test_values, label='Test data', alpha=0.6, stat='density')
    
    plt.xlabel('electricity consumption (kWh/h)')
    plt.ylabel('probability density')
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0, 3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()
    plt.close()

def analyze_profile_patterns(real_data, synthetic_data, save_path=None):
    """Analyze profile-specific patterns and variations."""
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Calculate profile-specific statistics
    real_stats = pd.DataFrame({
        'mean': real_data.mean(axis=0),
        'std': real_data.std(axis=0),
        'skew': real_data.skew(axis=0),
        'kurtosis': real_data.kurtosis(axis=0)
    })
    
    synth_stats = pd.DataFrame({
        'mean': synthetic_data.mean(axis=0),
        'std': synthetic_data.std(axis=0),
        'skew': synthetic_data.skew(axis=0),
        'kurtosis': synthetic_data.kurtosis(axis=0)
    })
    
    # Create comparison plots for each statistic
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Profile-specific Statistical Properties Comparison')
    
    # Mean comparison
    sns.scatterplot(data=real_stats, x='mean', y='std', label='Real', alpha=0.6, ax=axes[0,0])
    sns.scatterplot(data=synth_stats, x='mean', y='std', label='Synthetic', alpha=0.6, ax=axes[0,0])
    axes[0,0].set_title('Mean vs Standard Deviation')
    axes[0,0].set_xlabel('Mean Consumption')
    axes[0,0].set_ylabel('Standard Deviation')
    
    # Skewness comparison
    sns.scatterplot(data=real_stats, x='skew', y='kurtosis', label='Real', alpha=0.6, ax=axes[0,1])
    sns.scatterplot(data=synth_stats, x='skew', y='kurtosis', label='Synthetic', alpha=0.6, ax=axes[0,1])
    axes[0,1].set_title('Skewness vs Kurtosis')
    axes[0,1].set_xlabel('Skewness')
    axes[0,1].set_ylabel('Kurtosis')
    
    # Distribution of means
    sns.histplot(data=real_stats['mean'], label='Real', alpha=0.6, ax=axes[1,0])
    sns.histplot(data=synth_stats['mean'], label='Synthetic', alpha=0.6, ax=axes[1,0])
    axes[1,0].set_title('Distribution of Profile Means')
    axes[1,0].set_xlabel('Mean Consumption')
    axes[1,0].set_ylabel('Count')
    
    # Distribution of standard deviations
    sns.histplot(data=real_stats['std'], label='Real', alpha=0.6, ax=axes[1,1])
    sns.histplot(data=synth_stats['std'], label='Synthetic', alpha=0.6, ax=axes[1,1])
    axes[1,1].set_title('Distribution of Profile Standard Deviations')
    axes[1,1].set_xlabel('Standard Deviation')
    axes[1,1].set_ylabel('Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()
    plt.close()

def analyze_temporal_patterns(real_data, synthetic_data, save_path=None):
    """Analyze temporal patterns and seasonality."""
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Convert string indices to datetime
    real_data.index = pd.to_datetime(real_data.index)
    synthetic_data.index = pd.to_datetime(synthetic_data.index)
    
    # Calculate daily patterns
    real_daily = real_data.groupby(real_data.index.hour).mean()
    synth_daily = synthetic_data.groupby(synthetic_data.index.hour).mean()
    
    # Calculate weekly patterns
    real_weekly = real_data.groupby(real_data.index.dayofweek).mean()
    synth_weekly = synthetic_data.groupby(synthetic_data.index.dayofweek).mean()
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Temporal Pattern Comparison')
    
    # Daily patterns
    sns.lineplot(data=real_daily, label='Real', ax=axes[0])
    sns.lineplot(data=synth_daily, label='Synthetic', ax=axes[0])
    axes[0].set_title('Average Daily Pattern')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Average Consumption')
    axes[0].set_xticks(range(0, 24, 2))  # Show every other hour for clarity
    
    # Weekly patterns
    sns.lineplot(data=real_weekly, label='Real', ax=axes[1])
    sns.lineplot(data=synth_weekly, label='Synthetic', ax=axes[1])
    axes[1].set_title('Average Weekly Pattern')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Average Consumption')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()
    plt.close()

def analyze_clustering_behavior(real_data, synthetic_data, save_path=None):
    """Analyze clustering behavior and pattern recognition."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Perform PCA
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_data.T)
    synth_pca = pca.transform(synthetic_data.T)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    real_clusters = kmeans.fit_predict(real_pca)
    synth_clusters = kmeans.predict(synth_pca)
    
    # Create comparison plot
    plt.figure(figsize=(10, 8))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], c=real_clusters, label='Real', alpha=0.6)
    plt.scatter(synth_pca[:, 0], synth_pca[:, 1], c=synth_clusters, label='Synthetic', alpha=0.6, marker='x')
    plt.title('PCA and Clustering Comparison')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()
    plt.close()

def main():
    data_dir = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/GAN/data/Fluvius_processed/")
    train_file = data_dir / "train_data.csv"
    test_file = data_dir / "test_data.csv"
    
    # Load data
    train_df = pd.read_csv(train_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)
    
    # Create output directory for plots
    output_dir = Path(__file__).parent / "distribution_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Create train-test comparison plot
    train_test_plot_path = output_dir / "train_test_comparison.png"
    plot_train_test_comparison(train_df, test_df, train_test_plot_path)
    
    # Load synthetic datasets of different sizes
    synthetic_sizes = [261, 261*2, 261*4, 1500, 2000, 2500, 3000, 3500, 4000]
    synthetic_datasets = {}
    
    for size in synthetic_sizes:
        synthetic_file = data_dir / f"{size}_profiles.csv"
        if synthetic_file.exists():
            synthetic_datasets[size] = pd.read_csv(synthetic_file, index_col=0)
    
    # Analyze overall statistics for each synthetic dataset size
    overall_metrics = {}
    for size, synthetic_data in synthetic_datasets.items():
        metrics = calculate_overall_metrics(test_df, synthetic_data)
        overall_metrics[size] = metrics
        
        # Add new analyses for each synthetic dataset
        profile_patterns_path = output_dir / f"profile_patterns_{size}.png"
        analyze_profile_patterns(test_df, synthetic_data, profile_patterns_path)
        
        temporal_patterns_path = output_dir / f"temporal_patterns_{size}.png"
        analyze_temporal_patterns(test_df, synthetic_data, temporal_patterns_path)
        
        clustering_path = output_dir / f"clustering_behavior_{size}.png"
        analyze_clustering_behavior(test_df, synthetic_data, clustering_path)
    
    # Create distribution comparison plot
    plot_path = output_dir / "overall_distribution_comparison.png"
    plot_distribution_comparison(test_df, synthetic_datasets, plot_path)
    
    # Create summary plot of Wasserstein distances
    plt.figure(figsize=(10, 6))
    sizes = list(overall_metrics.keys())
    wasserstein_dists = [metrics['wasserstein_dist'] for metrics in overall_metrics.values()]
    
    plt.plot(sizes, wasserstein_dists, marker='o')
    plt.xlabel('Synthetic dataset size')
    plt.ylabel('Wasserstein Distance')
    plt.tight_layout()
    plt.savefig(output_dir / "wasserstein_by_size.png", dpi=900)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()


