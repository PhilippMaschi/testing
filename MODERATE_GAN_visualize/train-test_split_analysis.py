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

def analyze_profile_patterns(real_data, synthetic_data):
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
    return fig

def analyze_pca_comparison(train_data, test_data, synthetic_data):
    """Analyze and compare PCA projections of training, test and synthetic data."""
    from sklearn.decomposition import PCA
    
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
    # Fit PCA on training data and transform all datasets
    train_pca = pca.fit_transform(train_data.T)
    test_pca = pca.transform(test_data.T)
    synth_pca = pca.transform(synthetic_data.T)
    
    # Create comparison plot
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(train_pca[:, 0], train_pca[:, 1], c='blue', label='Training Data', alpha=0.6)
    plt.scatter(test_pca[:, 0], test_pca[:, 1], c='red', label='Test Data', alpha=0.6)
    plt.scatter(synth_pca[:, 0], synth_pca[:, 1], c='green', label='Synthetic Data', alpha=0.6, marker='x')
    
    plt.title('PCA Comparison of Training, Test and Synthetic Data')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    
    plt.tight_layout()
    return fig

def plot_dataset_distributions(train_data, test_data, synthetic_data, save_path=None):
    """Create distribution comparison plots with histograms for training, test and synthetic data."""
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Flatten all datasets
    train_values = train_data.values.flatten()
    test_values = test_data.values.flatten()
    synth_values = synthetic_data.values.flatten()
    
    # Create figure for histogram
    fig = plt.figure(figsize=(8, 6))
    
    
    # Plot histograms with step style to ensure visibility
    sns.histplot(data=train_values, label='Training Data', 
                stat='density', color='blue', 
                common_norm=False, element='step', fill=False, linewidth=2)
    sns.histplot(data=test_values, label='Test Data', 
                stat='density', color='red', 
                common_norm=False, element='step', fill=False, linewidth=2)
    sns.histplot(data=synth_values, label='Synthetic Data', 
                stat='density', color='green', 
                common_norm=False, element='step', fill=False, linewidth=2)
    
    plt.xlabel('electricity consumption (kWh/h)')
    plt.ylabel('probability density')
    plt.legend()
    plt.xlim(0, 2)  
    plt.ylim(0, 3)   
    plt.tight_layout()
    return fig

def plot_cdf(train_data, test_data, synthetic_data, save_path=None):
    train_values = train_data.values.flatten()
    test_values = test_data.values.flatten()
    synth_values = synthetic_data.values.flatten()
    # Create separate figure for CDF
    fig = plt.figure(figsize=(8, 6))
    
    # Plot CDFs for all datasets
    x = np.sort(train_values)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label='Training Data', linewidth=2, color='blue')
    
    x = np.sort(test_values)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label='Test Data', linewidth=2, color='red')
    
    x = np.sort(synth_values)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label='Synthetic Data', linewidth=2, color='green')
    
    plt.xlabel('electricity consumption (kWh/h)')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.xlim(0, 2.5)  # Zoom in on x-axis
    
    plt.tight_layout()
    return fig

def plot_wasserstein_by_size(overall_metrics):
    """Create a plot of Wasserstein distances by synthetic dataset size."""
    fig = plt.figure(figsize=(10, 6))
    sizes = list(overall_metrics.keys())
    wasserstein_dists = [metrics['wasserstein_dist'] for metrics in overall_metrics.values()]
    
    plt.plot(sizes, wasserstein_dists, marker='o')
    plt.xlabel('Synthetic dataset size')
    plt.ylabel('Wasserstein Distance')
    plt.tight_layout()
    return fig


def train_test_comparison():
    data_dir = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/GAN/data/Fluvius_processed/")
    train_file = data_dir / "train_data.csv"
    test_file = data_dir / "test_data.csv"
    
    # Load data
    train_df = pd.read_csv(train_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)
    
    # Create output directory for plots
    output_dir = Path(__file__).parent / "distribution_analysis"
    output_dir.mkdir(exist_ok=True)
        
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
        
    # Create profile patterns plot
    profile_patterns_path = output_dir / f"profile_patterns_{261}.png"
    profile_patterns_fig = analyze_profile_patterns(test_df, synthetic_datasets[261])
    profile_patterns_fig.savefig(profile_patterns_path, dpi=600)
    plt.close(profile_patterns_fig)
    
    # Create PCA plot
    pca_comparison_path = output_dir / f"pca_comparison_{261}.png"
    pca_comparison_fig = analyze_pca_comparison(train_df, test_df, synthetic_datasets[261])
    pca_comparison_fig.savefig(pca_comparison_path, dpi=600)
    plt.close(pca_comparison_fig)
    
    # Create distribution comparison plot
    synthetic_df = synthetic_datasets[261]
    plot_path = output_dir / "dataset_distributions.png"
    hist_plot = plot_dataset_distributions(train_df, test_df, synthetic_df, plot_path)
    hist_plot.savefig(plot_path, dpi=600)
    plt.close(hist_plot)
    
    # Create CDF plot
    cdf_plot_path = output_dir / "cdf_plot.png"
    cdf_plot = plot_cdf(train_df, test_df, synthetic_df, cdf_plot_path)
    cdf_plot.savefig(cdf_plot_path, dpi=600)
    plt.close(cdf_plot)

    # Create Wasserstein plot
    wasserstein_plot_path = output_dir / "wasserstein_by_size.png"
    wasserstein_plot = plot_wasserstein_by_size(overall_metrics)
    wasserstein_plot.savefig(wasserstein_plot_path, dpi=600)
    plt.close(wasserstein_plot)


if __name__ == "__main__":
    train_test_comparison()


