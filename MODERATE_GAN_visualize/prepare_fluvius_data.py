import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

translation_dict = {
    "Warmtepomp_Indicator": "heat pump",
    "Elektrisch_Voertuig_Indicator": "electric vehicle",
    "PV-Installatie_Indicator": "PV",
}

def preprocess_fluvius_data(df):
    """
    Preprocess Fluvius data:
    1. Convert Datum_Startuur to datetime and set as index
    2. Group by EAN_ID to maintain separate time series for each meter
    3. Resample each group to hourly frequency, summing energy volumes
    4. Keep other columns (string/boolean) by using first value in each hour
    
    Args:
        df: Raw Fluvius DataFrame
        
    Returns:
        Preprocessed hourly DataFrame
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert datetime column to proper datetime format
    df_processed['Date'] = pd.to_datetime(df_processed['Datum_Startuur'])
    df_processed.drop(columns=['Datum_Startuur', 'Datum'], inplace=True)
    df_processed.set_index('Date', inplace=True)
    
    # Identify numeric columns to sum and non-numeric columns to keep first value
    numeric_cols = ['Volume_Afname_kWh', 'Volume_Injectie_kWh']
    
    # Identify non-numeric columns (keeping first value)
    non_numeric_cols = [col for col in df_processed.columns if col not in numeric_cols]
    
    # Create aggregation dictionary
    agg_dict = {col: 'sum' for col in numeric_cols}
    agg_dict.update({col: 'first' for col in non_numeric_cols})
    
    # Process each meter ID separately
    df_hourly_list = []
    
    for ean_id, group in df_processed.groupby('EAN_ID'):
        # Resample to hourly frequency with appropriate aggregation
        group_hourly = group.resample('h').agg(agg_dict)
        df_hourly_list.append(group_hourly)
    
    # Combine all hourly dataframes
    df_hourly = pd.concat(df_hourly_list)
    
    # Sort by EAN_ID and timestamp for better organization
    df_hourly = df_hourly.sort_values(by=['EAN_ID'])
    # Also sort by timestamp
    df_hourly = df_hourly.sort_index(level=0)
    
    print(f"Original data shape: {df.shape}")
    print(f"Hourly resampled data shape: {df_hourly.shape}")
    print(f"Number of unique EAN_IDs: {df_hourly['EAN_ID'].nunique()}")
    
    return df_hourly


def save_processed_data(df, output_path):
    """Save processed data to CSV and pickle formats"""
    # Save as CSV
    df.to_csv(output_path.with_suffix('.csv'))
    print(f"Data saved to {output_path.with_suffix('.csv')}")

def create_indicator_df(df):
    """
    Create a DataFrame that correlates EAN_ID with their indicators.
    
    Args:
        df: Processed Fluvius DataFrame with indicator columns
        
    Returns:
        DataFrame with EAN_ID and their corresponding labels
    """
    # Get unique EAN_IDs
    unique_eans = df['EAN_ID'].unique()
    
    # Create empty DataFrame
    indicators_df = pd.DataFrame(index=unique_eans, columns=['label'])
    indicators_df.index.name = 'EAN_ID'
    
    # For each EAN_ID, check indicators and create labels
    for ean_id in unique_eans:
        # Get first row for this EAN_ID (assuming indicators are the same for a meter)
        meter_data = df[df['EAN_ID'] == ean_id].iloc[0]
        
        # Initialize empty label
        label = []
        
        # Check each indicator and add corresponding label
        if meter_data.get('Warmtepomp_Indicator', 0) == 1:
            label.append("heat pump")
        
        if meter_data.get('Elektrisch_Voertuig_Indicator', 0) == 1:
            label.append("EV")
            
        if meter_data.get('PV-Installatie_Indicator', 0) == 1:
            label.append("PV")
        
        # If no indicators are true, label as 'standard'
        if not label:
            label = ["standard"]
            
        # Join labels with '+'
        indicators_df.loc[ean_id, 'label'] = '+'.join(label)
    
    return indicators_df

def plot_label_distribution(indicators_df):
    """
    Plot the distribution of indicator labels as a bar plot.
    
    Args:
        indicators_df: DataFrame with EAN_ID as index and 'label' column
        
    Returns:
        matplotlib figure
    """
    # Get label counts
    label_counts = indicators_df['label'].value_counts()
    
    # Sort by count (optional)
    label_counts = label_counts.sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    bars = ax.bar(label_counts.index, label_counts.values, color='skyblue', edgecolor='navy')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels inside bars
    total = label_counts.sum()
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height / total * 100
        if height > 0:  # Only add text if the bar has height
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{percentage:.1f}%', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    # Set title and labels
    ax.set_title('Distribution of Energy Profile Types', fontsize=16)
    ax.set_xlabel('Profile Type', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def reshape_to_wide_format(df, value_column='Volume_Afname_kWh', id_column='EAN_ID'):
    """
    Reshape the DataFrame to a wide format where:
    - Each EAN_ID is a column
    - Datetime remains as the index
    - Values are from the specified value column
    
    Args:
        df: Preprocessed hourly DataFrame
        value_column: Column containing the values to use (default: 'Volume_Afname_kWh')
        id_column: Column containing the identifiers (default: 'EAN_ID')
        
    Returns:
        Wide-format DataFrame
    """
    # Keep only necessary columns
    df_subset = df[[id_column, value_column]].copy()
   
    # Pivot the table
    df_wide = df_subset.pivot(columns=id_column, values=value_column)
    
    # Rename the index
    df_wide.index.name = 'datetime'
    
    return df_wide

def main():
    # Load data
    path = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/GAN/data/Fluvius_original/") / "P6269_1_50_DMK_Sample_Elek.csv"
    df = pd.read_csv(path, sep=";")
    
    # Preprocess data
    df_hourly = preprocess_fluvius_data(df)
    
    # Create indicators DataFrame
    indicators_df = create_indicator_df(df_hourly)
    print("\nIndicator labels:")
    print(indicators_df.head())
    
    # Count of each label type
    print("\nCounts of each profile type:")
    print(indicators_df['label'].value_counts())
    
    # Save indicator DataFrame
    output_dir = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/GAN/data/Fluvius_processed/")
    output_dir.mkdir(parents=True, exist_ok=True)
    indicators_df.to_csv(output_dir / "fluvius_indicators.csv")
    
    # Plot label distribution
    label_fig = plot_label_distribution(indicators_df)
    label_fig.savefig(output_dir / "label_distribution.svg", dpi=300, bbox_inches='tight')
    plt.close(label_fig)
    
    # Convert to wide format (profiles as columns)
    print("\nReshaping to wide format...")
    df_wide = reshape_to_wide_format(df_hourly)
    print(f"Wide format shape: {df_wide.shape}")
    print("First few rows and columns:")
    print(df_wide.iloc[:5, :5])
    
    # Check for completeness
    missing_values = df_wide.isna().sum().sum()
    total_values = df_wide.size
    print(f"Missing values: {missing_values} ({missing_values/total_values*100:.2f}% of total)")
    
    # Save processed data
    save_processed_data(df_wide, output_dir / "fluvius_wide_format")
    # Also save original format if needed
    save_processed_data(df_hourly, output_dir / "fluvius_hourly")
    print("Processing complete!")

if __name__ == "__main__":
    main()