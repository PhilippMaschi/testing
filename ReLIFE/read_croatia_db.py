import pandas as pd
import pyodbc
print(pyodbc.drivers())
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
from pathlib import Path
import sqlalchemy
from typing import List
from ydata_profiling import ProfileReport
import matplotlib

column_translation = {
    "ISGE šifra objekta": "ISGE object code",
    "Vrsta objekta": "Type of object",
    "Godina": "Year",
    "Mjesec": "Month",
    "Energent": "Energy source",
    "Količina": "Quantity",
    "kWh": "kWh",  # No translation needed
    "Cijena u eurima": "Price in euros",
    "Cijena s PDV-om u eurima": "Price with VAT in euros",
    "CO2": "CO2",  # No translation needed
    "Primarna energija": "Primary energy",
    "Grad/mjesto": "City/Location",
    "Ploština korisne površine Ak": "Useful surface area Ak",
    "Mjerna jedinica": "Unit of measurement"
}

translation_dict = {
    # Energy sources
    "Električna energija": "Electric energy",
    "Voda": "Water",
    "Loživo ulje ekstra lako": "Extra light fuel oil",
    "Prirodni plin": "Natural gas",
    "Drvna sječka": "Wood chips",
    "Toplina": "Heat",
    "UNP": "LPG (Liquefied Petroleum Gas)",
    "Peleti": "Pellets",
    "Drvo za ogrjev": "Firewood",
    "Loživo ulje srednje": "Medium fuel oil",
    "Para": "Steam",
    "Plin u boci": "Bottled gas",
    "Briket": "Briquettes",
    "Loživo ulje lako": "Light fuel oil",
    "Dizel": "Diesel",

    # Building types
    "Fakultetska zgrada": "University building",
    "Muzeji i knjižnice": "Museums and libraries",
    "Osnovna škola": "Primary school",
    "Uredska zgrada": "Office building",
    "Zgrade za kulturno-umjetničku djelatnost i zabavu": "Buildings for cultural and artistic activities and entertainment",
    "Zatvori, kaznionice i popravni centri": "Prisons, correctional facilities, and rehabilitation centers",
    "Bolnica": "Hospital",
    "Srednja škola": "Secondary school",
    "Umirovljenički dom": "Retirement home",
    "Zgrade za trgovinu na veliko i malo": "Buildings for wholesale and retail trade",
    "Ambulanta": "Clinic",
    "Đački i studentski dom": "Student dormitory",
    "Dječji vrtić": "Kindergarten",
    "Dječji dom": "Children's home",
    "Dom (općenito)": "Home (general)",
    "Stambene zgrade s jednim stanom": "Residential buildings with one apartment",
    "Ostale građevine (osim zgrada), drugdje neklasificirane": "Other structures (except buildings), not classified elsewhere",
    "Hoteli i moteli": "Hotels and motels",
    "Garaže": "Garages",
    "Vojarna": "Barracks",
    "Sportska dvorana": "Sports hall",
    "Zgrade željezničkog, cestovnog, zračnog i vodenog prometa": "Buildings for railway, road, air, and water transport",
    "Stambena zgrada s više od 3 stana": "Residential building with more than 3 apartments",
    "Ostale zgrade, drugdje neklasificirane": "Other buildings, not classified elsewhere",
    "Nefizički objekt (neaktivan)": "Non-physical object (inactive)",
    "Zgrade za znanstvenoistraživačku djelatnost": "Buildings for scientific research activities",
    "Stambene zgrade s dva stana": "Residential buildings with two apartments",
    "Ostale građevine na grobljima": "Other structures in cemeteries",
    "Industrijske zgrade": "Industrial buildings",
    "Ostale zgrade za kratkotrajni boravak": "Other buildings for short-term accommodation",
    "Povijesni ili zaštićeni spomenici": "Historical or protected monuments",
    "Zgrade za televizijsko i radijsko emitiranje": "Buildings for television and radio broadcasting",
    "Zgrade na grobljima": "Buildings in cemeteries",
    "Restorani, barovi i slične ugostiteljske zgrade": "Restaurants, bars, and similar hospitality buildings",
    "Zatvorena skladišta": "Closed warehouses",
    "Zgrade pošta i telekomunikacija": "Buildings for post and telecommunications",
    "Staje za stoku i peradarnici": "Barns for livestock and poultry houses",
    "Natkrivena skladišta": "Covered warehouses",
    "Zgrade za uzgoj, proizvodnju i smještaj poljoprivrednih proizvoda": "Buildings for cultivation, production, and storage of agricultural products",
    "Zgrade za obavljanje vjerskih obreda": "Buildings for religious ceremonies",
    "Ostale zgrade za promet i komunikacije": "Other buildings for transport and communications"
}

class SQLITE_DB:
    def __init__(self, path: Path):
        self.engine = sqlalchemy.create_engine(f'sqlite:///{path}')
        self.metadata = sqlalchemy.MetaData()
        self.metadata.reflect(bind=self.engine)

    def get_table_names(self) -> List[str]:
        """Returns a list of all table names in the database."""
        return list(self.metadata.tables.keys())

    def write_dataframe(
        self,
        table_name: str,
        data_frame: pd.DataFrame,
        data_types: dict = None,
        if_exists="replace",
    ):  # if_exists: {'replace', 'fail', 'append'}
        data_frame.to_sql(
            table_name,
            self.engine,
            index=False,
            dtype=data_types,
            if_exists=if_exists,
            chunksize=10_000,
        )

    def read_dataframe(self, table_name: str, filter: dict = None, column_names: List[str] = None) -> pd.DataFrame:
        """Reads data from a database table with optional filtering and column selection.

                Args:
                    table_name (str): Name of the table to query.
                    filter (dict, optional): Dictionary with {column_name: value} to filter the data.
                    column_names (list of str, optional): List of column names to extract.

                Returns:
                    pd.DataFrame: Resulting dataframe.
                """
        table = self.metadata.tables[table_name]

        if column_names:
            columns = [table.columns[name] for name in column_names]
            query = sqlalchemy.select(*columns)
        else:
            query = sqlalchemy.select(table)

        if filter:
            for key, value in filter.items():
                query = query.where(table.columns[key] == value)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)
            

def read_accdb_database(path2db: str) -> pd.DataFrame:
    # Connection string
    conn_str = (
        r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
        f"DBQ={path2db};"
    )
    # Establish connection
    conn = pyodbc.connect(conn_str)
    table_name = "TAN_ANALYSIS" 
    query = f"SELECT * FROM [{table_name}]"
    df = pd.read_sql(query, conn)
    # Close the connection
    conn.close()
    return df

def translate_db(df: pd.DataFrame) -> pd.DataFrame:
    # translate db to englisch
    df.rename(columns=column_translation, inplace=True)
    df["Type of object"].replace(translation_dict, inplace=True)
    df["Energy source"].replace(translation_dict, inplace=True)
    return df

def save_to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str, if_exists: str = "replace"):
    """
    Save a pandas DataFrame to an SQLite database.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        db_filename (Path): The path the SQLite database file.
        table_name (str): The name of the table where the data will be stored.
        if_exists (str): Behavior when the table exists ('fail', 'replace', 'append'). Default is 'replace'.
    """
    try:
        # Connect to SQLite database (creates file if it does not exist)
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        print(f"Data successfully saved to {db_path} in table '{table_name}'.")
    except Exception as e:
        print(f"Error saving DataFrame to SQLite: {e}")


def show_monthly_total_consumption(df):
    import matplotlib.pyplot as plt
    import pandas as pd
    grouped = df.groupby(["Year", "Month", "Energy source"])["kWh"].sum().reset_index()
    pd.options.display.float_format = '{:,.0f}'.format

    grouped = grouped.sort_values(by=['Energy source', 'Year', 'Month'])
    grouped['Date'] = pd.to_datetime(grouped['Year'].astype(str) + '-' + grouped['Month'].astype(str) + '-01')

    # Plot
    plt.figure(figsize=(14, 8))

    for energy_source, group in grouped.groupby('Energy source'):
        plt.plot(group['Date'], group['kWh'], label=energy_source)

    plt.xlabel('Date')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title('Monthly Energy Consumption by Source')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # Legend outside the plot
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def convert_floating_point_commas(val):
    if isinstance(val, str):
        if "," in val and "." in val:
            # If both present, decide based on last position
            if val.rfind(",") > val.rfind("."):
                # Comma is decimal
                val = val.replace(".", "").replace(",", ".")
            else:
                # Dot is decimal
                val = val.replace(",", "")
        elif "," in val:
            # Assume comma is decimal
            val = val.replace(".", "").replace(",", ".")
        elif "." in val:
            # Assume dot is decimal
            val = val.replace(",", "")
    try:
        return float(val)
    except ValueError:
        return None  # or np.nan


def read_sqlite_db(db_path: Path):
    print(f"Attempting to open database at: {db_path}")
    db = SQLITE_DB(path=db_path)
    table_names = db.get_table_names()
    print(f"Found tables: {table_names}")
    
    if not table_names:
        print("No tables found in the database!")
        return
        
    df = db.read_dataframe(table_name=table_names[0])
    print(f"Successfully read table: {table_names[0]}")
    print(f"DataFrame shape: {df.shape}")
    return df

def mark_outliers_from_group(group, col):
    Q1 = group[col].quantile(0.25)
    Q3 = group[col].quantile(0.75)
    IQR = Q3 - Q1
    if "area" in col.lower():
        lower = 0
    else:
        lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return group.loc[(group[col] < lower) | (group[col] > upper), [col, "ISGE object code", "Energy source", "year_month"]]
     

def count_outliers(df_original, df_cleaned, columns):
    """
    Count the number of outliers marked in each column.
    
    Parameters:
        df_original (pd.DataFrame): Original DataFrame before outlier removal
        df_cleaned (pd.DataFrame): DataFrame after outlier removal
        columns (list): List of columns to check for outliers
        
    Returns:
        dict: Dictionary with column names as keys and number of outliers as values
    """
    outlier_counts = {}
    for col in columns:
        original_na = df_original[col].isna().sum()
        cleaned_na = df_cleaned[col].isna().sum()
        outlier_counts[col] = cleaned_na - original_na
    
    return outlier_counts

def analyze_building_outliers(df_original, normalized_cols, building_id_col="ISGE object code"):
    """
    Analyze how buildings' outlier status changes across different months.
    
    Parameters:
        df_original (pd.DataFrame): Original DataFrame before outlier removal
        df_cleaned (pd.DataFrame): DataFrame after outlier removal
        building_id_col (str): Name of the column containing building IDs
        
    Returns:
        tuple: (total_outlier_buildings, monthly_outlier_buildings, building_outlier_frequency)
    """
    outlier_df = df_original.loc[df_original["outlier"]==True, normalized_cols + [building_id_col] + ["Energy source"] + ["year_month"]]
    
    for i in normalized_cols:
        sns.boxplot(
            data=outlier_df,
            x="year_month",
            y=i,

        )
        plt.show()

    # Filter out rows where only one of normalized Quantity or kWh is NA
    quantity_kwh_mask = (
        (outlier_buildings["normalized Quantity"].isna() & outlier_buildings["normalized kWh"].notna()) |
        (outlier_buildings["normalized Quantity"].notna() & outlier_buildings["normalized kWh"].isna())
    )
    outlier_buildings = outlier_buildings[~quantity_kwh_mask]

    for year_month, group in outlier_buildings.groupby(["year_month", building_id_col]):
        if len(group)>1:
            break
    
    # Remove the last digits after the last "-" and get unique values
    outlier_buildings_short = list(set(["-".join(building.split("-")[:-1]) for building in outlier_buildings[building_id_col]]))
    total_outlier_buildings = len(outlier_buildings_short)
    
    # Count how many months each building was an outlier in each energy column
    building_outlier_frequency = {}
    
#     for col in normalized_cols:
#         building_outlier_frequency[col] = {}
#         for building in outlier_buildings_short:
#             # Match buildings with the same prefix (ignoring the last digits)
#             building_prefix = building
#             building_data = df_cleaned[df_cleaned[building_id_col].str.startswith(building_prefix)]
#             # Count months where this building was an outlier for this column
#             months_as_outlier = building_data[outlier_mask[col]].sum()
#             building_outlier_frequency[col][building] = months_as_outlier
    
#     # Count how many buildings were outliers in each month
#     monthly_outlier_buildings = {}
#     for year_month in df_cleaned['year_month'].unique():
#         month_data = df_cleaned[df_cleaned['year_month'] == year_month]
#         month_mask = outlier_mask.loc[month_data.index]
#         # Get unique building prefixes for this month
#         month_outliers = month_data.loc[month_mask.any(axis=1), building_id_col].unique()
#         month_outliers = ["-".join(building.split("-")[:-1]) for building in month_outliers]
#         monthly_outlier_buildings[year_month] = len(set(month_outliers))
    
#     return total_outlier_buildings, monthly_outlier_buildings, building_outlier_frequency

# def plot_outliers(df: pd.DataFrame, columns: list, percentage: dict):
#     for col in columns:
#         df["is_outlier"] = df[col].isna()
#         outlier = df["is_outlier"].value_counts(normalize=True).get(True, 0)
#         percentage[col] = outlier * 100

#     outlier_df = pd.DataFrame.from_dict(percentage, orient="index", columns=["Outlier (%)"]).reset_index()
#     plt.figure(figsize=(6,8))
#     sns.barplot(
#         data=outlier_df,
#         x="index",
#         y="Outlier (%)",
#         hue="index"
#     )

#     plt.xlabel("")
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.show()

def plot_outliers(out_df: pd.DataFrame, col: str):
    sns.scatterplot(
        data=out_df,
        x="year_month",
        y=col

    )

    plt.tight_layout()
    plt.show()

def remove_zero_area(df: pd.DataFrame):
    # drop all buildigns where the area is 0:
    df_without_zero = df.loc[df["Useful surface area Ak"]!=0, :]
    # drop all rows where the Useful surface area is not available:
    df_without_zero = df_without_zero.dropna(subset=["Useful surface area Ak"])
    return df_without_zero

def normalize_columns(df: pd.DataFrame):
    numeric_cols = [i for i in df.select_dtypes(include='number').columns if not "area" in i]
    for col in numeric_cols:
        if col != "Month" and col != "Year" and col != "Useful surface area Ak":
            df[f"{col}"] = df[col] / df["Useful surface area Ak"]
    return df

def clean_dataset(df: pd.DataFrame):
    # before removing outliers, the data has to be normalized using the floor area to make it comparable
    numeric_cols = [i for i in df.select_dtypes(include='number').columns if not "area" in i]
    new_df = df.copy()

    df_without_zero = remove_zero_area(new_df)
    df_without_zero["outlier"] = False
    df_without_zero["year_month"] = df_without_zero['Year'].astype(str) + '-' + df_without_zero['Month'].astype(str).str.zfill(2)
    # now remove outliers of the Usefule surface area:
    area_outlier_df = mark_outliers_from_group(df_without_zero_no_na, col="Useful surface area Ak")
    print(f"{round(len(area_outlier_df) / len(df_without_zero_no_na)  * 100, 2)}% are outliers based on the floor area")
    plot_outliers(area_outlier_df, col="Useful surface area Ak")

    df_normalized = df_without_zero_no_na.loc[~df_without_zero_no_na.index.isin(area_outlier_df.index), : ].copy()

    normalized_cols = []
    for col in numeric_cols:
        if col != "Month" and col != "Year":
            df_normalized[f"normalized {col}"] = df_normalized[col] / df_normalized["Useful surface area Ak"]
            normalized_cols.append(f"normalized {col}")
    
    df_norm = df_normalized.copy()
    outlier_dfs = {}
    for col in normalized_cols:
        outlier_dfs[col] = df_norm.groupby("year_month", group_keys=False).apply(
            lambda group: mark_outliers_from_group(group, col)
        )
    for col, df in outlier_dfs.items():
        plot_outliers(df, col)
        # mark the outliers in the orig dataset
        df_norm[f"{col} outlier"] = False
        df_norm.loc[df_norm.index.isin(df.index), f"{col} outlier"] = True
        
    

    
    # Analyze building outlier patterns
    total_outliers, monthly_outliers, building_frequency = analyze_building_outliers(df_norm, normalized_cols)
    print(f"\nOutlier Analysis:")
    print(f"Total number of unique buildings that were outliers: {total_outliers}")
    print("\nBuildings that were outliers in multiple months:")
    for building, frequency in sorted(building_frequency.items(), key=lambda x: x[1], reverse=True):
        if frequency > 1:
            print(f"Building {building}: {frequency} months as outlier")
    
    return 


def create_stackplot_positive_negative_y(df, word: str):
    df["year_month"] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    energy = df.groupby(["year_month", "Energy source"])["kWh"].sum().reset_index()
    energy["kWh"] = energy["kWh"] / 1_000 / 1_000 / 1_000 # TWh
    positive_df = energy.loc[~energy["Energy source"].isin(["Water","Electric energy", "Heat"]), :]
    negative_df = energy.loc[energy["Energy source"].isin(["Heat"]), :].reset_index(drop=True)
    
    pivot_postive = positive_df.pivot(index="year_month", columns="Energy source", values=["kWh"]).fillna(0)
    pivot_postive.columns = pivot_postive.columns.get_level_values(1)
    pivot_postive.index.name = None
    pivot_negative = negative_df.pivot(index="year_month", columns="Energy source", values=["kWh"]).fillna(0)
    pivot_negative.columns = pivot_negative.columns.get_level_values(1)
    pivot_negative.index.name = None
    
    x_axis = pivot_postive.index

    palette = sns.color_palette("tab20", n_colors=20)
    columns = [col for col in pivot_postive.columns if not "year" in col]
    colors = {col: palette[i % len(palette)] for i, col in enumerate(columns)}
    sns.set_theme(font_scale=1.6)
    figure = plt.figure(figsize=(14, 10))
    plt.stackplot(
        x_axis,                 # x-axis (year)
        [pivot_postive[col] for col in pivot_postive.columns if not "year" in col], 
        labels=pivot_postive.columns,
        colors=[colors[col] for col in columns]         
    )
    plt.stackplot(
        x_axis,
        [-pivot_negative[col] for col in pivot_negative.columns if not "year" in col], 
        labels=pivot_negative.columns,
        colors=[colors[col] for col in columns]         
    )
    plt.xticks(rotation=90)
    plt.xlabel('date')

    plt.legend(loc='upper right', ncol=2)
    plt.ylabel(f'primary energy (TWh)')
    plt.xlim(0, len(x_axis)-1)
    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER / f"Primary_energy_heating_over_time_{word}.svg")
    plt.show()
    plt.close()


def create_stackplot(df: pd.DataFrame, y_axis: str):
    if y_axis == "Water":
        column_name = "Quantity"
    else:
        column_name = "kWh"

    energy = df.groupby(["Energy source", "Month", "Year"])[column_name].sum().reset_index()

    energy["year_month"] = energy['Year'].astype(str) + '-' + energy['Month'].astype(str).str.zfill(2)
    if y_axis == "Water":
        energy[column_name] = energy[column_name] / 1_000 / 1_000   # mio cubic meter
    else:
        energy[column_name] = energy[column_name] / 1_000 / 1_000 / 1_000  # TWh

    pivot_energy = energy.pivot(index="year_month", columns="Energy source", values=column_name).reset_index()
    
    x_axis = pivot_energy["year_month"]
    pivot_energy.drop(columns="year_month", inplace=True)
    pivot_energy = pivot_energy.apply(pd.to_numeric, errors='coerce')
    pivot_energy = pivot_energy.fillna(0)
    
    
    sns.set_theme(font_scale=1.6)
    figure = plt.figure(figsize=(14, 10))
    plt.stackplot(
        x_axis,                 # x-axis (year)
        [pivot_energy[col] for col in pivot_energy.columns if not "year" in col],  # y-values for each category
        labels=pivot_energy.columns          # labels for legend
    )
    plt.xticks(rotation=90)
    plt.xlabel('date')
    if y_axis =="Water":
        plt.ylabel("water (mio. m$^3$)")
        plt.legend(loc='upper right', ncol=1)

    else:
        plt.legend(loc='upper right', ncol=2)

        plt.ylabel(f'primary energy (TWh)')
    plt.xlim(0, len(x_axis)-1)
    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER / f"Primary_energy_{y_axis}_over_time.svg")
    plt.show()
    plt.close()

def plot_energy_source_consumption(df):
    """Create time series plots of mean and median consumption for each energy source.
    
    Args:
        df: DataFrame containing columns 'Energy source', 'kWh', 'Quantity', 'Year', and 'Month'
    """
    # Set font sizes
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8
    })
    
    # Get unique energy sources
    energy_sources = df['Energy source'].unique()
    
    # Calculate number of rows and columns for subplot grid
    n_sources = len(energy_sources)
    n_cols = 3
    n_rows = (n_sources + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    # Create time series for each energy source
    for idx, source in enumerate(energy_sources):
        # Filter data for current energy source
        source_data = df[df['Energy source'] == source].copy()
        
        # Create datetime index
        source_data['Date'] = pd.to_datetime(source_data[['Year', 'Month']].assign(day=1))
        
        # Choose column based on energy source
        value_column = 'Quantity' if source == 'Water' else 'kWh'
        
        # Calculate mean and median for each month
        monthly_stats = source_data.groupby('Date').agg({
            value_column: ['mean', 'median']
        }).reset_index()
        
        # Plot
        ax = axes[idx]
        ax.plot(monthly_stats['Date'], monthly_stats[(value_column, 'mean')], 
                label='Mean', color='blue', linewidth=1.5)
        ax.plot(monthly_stats['Date'], monthly_stats[(value_column, 'median')], 
                label='Median', color='red', linewidth=1.5)
        
        # Customize plot
        ax.set_title(f'{source}', pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity' if source == 'Water' else 'kWh')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis to show values in thousands
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
    
    # Remove empty subplots
    for idx in range(len(energy_sources), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig

def plot_statistics(db_path: Path):

    df = read_sqlite_db(db_path=db_path)
    figure = plot_energy_source_consumption(df)
    figure.savefig(FIGURE_FOLDER / "energy_source_consumption_raw.svg",)# dpi=300)
    plt.close(figure)

    figure = plot_energy_source_consumption(remove_zero_area(df))
    figure.savefig(FIGURE_FOLDER / "energy_source_consumption_no_zero_area.svg",)# dpi=300)
    plt.close(figure)

    df_normalized = normalize_columns(remove_zero_area(df))
    figure = plot_energy_source_consumption(df_normalized)
    figure.savefig(FIGURE_FOLDER / "energy_source_consumption_normalized.svg",)# dpi=300)
    plt.close(figure)
    # isge_groups = df.groupby(["ISGE object code", "Energy source"])
    # group = isge_groups.get_group(("HR-51410-0018-0", "Electric energy"))
    # profile = ProfileReport(df, title="Profiling Report")
    # output_path = db_path.parent / "data_profile.html"
    # profile.to_file(output_file=output_path)
    # profile.to_widgets()
    # print("saved y data profiling report")

    for source in ["Water", "Electric energy"]:
        filtered = df.loc[df["Energy source"] == source, :]
        create_stackplot(filtered, source)

    # create_stackplot(df.loc[~df["Energy source"].isin(["Water","Electric energy", "Heat"]), :], y_axis="primary energy")
    # create_stackplot(df.loc[df["Energy source"].isin(["Heat"]), :], y_axis="heating energy")

    clean_df, area_outlier = clean_dataset(df)
    # percentage = {"Useful surface area Ak": area_outlier*100}
    columns_2_plot = [i for i in df.select_dtypes(include='number').columns if i not in ["Year", "Month", "Useful surface area Ak"]]
    plot_outliers(clean_df, columns=columns_2_plot, percentage={})

    create_stackplot_positive_negative_y(df, "original")
    create_stackplot_positive_negative_y(clean_df, "clean")





    for (year, month), group in df.groupby(["Year", "Month"]):
        energy = group.groupby(["Energy source", "Month", "Year"])["kWh"].sum().reset_index()
        energy["year_month"] = energy['Year'].astype(str) + '-' + energy['Month'].astype(str).str.zfill(2)
        pivot_energy = energy.pivot(index="year_month", columns="Energy source", values="kWh")


        plt.show()


        # fig, ax = plt.subplots(figsize=(8,8))
        # wedges, texts, autotexts = ax.pie(
        #     energy["kWh"], labels=None, autopct='%1.1f%%', startangle=90
        # )
        # ax.legend(wedges, energy["Energy source"], title="Energy Source", loc="center left", bbox_to_anchor=(1, 0.5))
        # plt.tight_layout()
        # plt.show()

        # # Calculate total energy consumption
        # total_kWh = energy["kWh"].sum()

        # # Define threshold for 5%
        # threshold = 0.05 * total_kWh

        # # Separate categories above and below the threshold
        # above_threshold = energy[energy["kWh"] >= threshold].copy()
        # below_threshold = energy[energy["kWh"] < threshold].copy()

        # # Sum all small categories into one
        # if not below_threshold.empty:
        #     other_sum = below_threshold["kWh"].sum()
        #     above_threshold = above_threshold.append(
        #         {"Energy source": "Other", "kWh": other_sum}, ignore_index=True
        #     )


        # # Plot new pie chart
        # fig, ax = plt.subplots(figsize=(8, 8))
        # wedges, texts, autotexts = ax.pie(
        #     above_threshold["kWh"],
        #     labels=None,  # No direct labels to avoid clutter
        #     autopct='%1.1f%%',
        #     startangle=90,
        # )

        # # Add external legend
        # ax.legend(wedges, above_threshold["Energy source"], title="Energy Source", loc="center left", bbox_to_anchor=(1, 0.5))

        # # Set title

        # # Show plot
        # plt.show()

    


if __name__ == "__main__":
    db_path = Path(r"/home/users/pmascherbauer/projects4/workspace_nikolausd/ReLIFE/ReLIFE_datasets/Croatia_public_buildings_data.sqlite")
    FIGURE_FOLDER = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/ReLIFE/figures")
    # database_path = r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\ReLIFE\data\ISGE_potrošnja_2020-2024.accdb"
    # df = translate_db(read_accdb_database(path2db=database_path))
    # save_to_sqlite(df=df, 
    #                db_path=db_path / "Croatia_public_buildings_data.db", 
    #                table_name="Energy_data")
    
    plot_statistics(db_path=db_path)
