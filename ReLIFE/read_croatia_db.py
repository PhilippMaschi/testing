import pandas as pd
import pyodbc
print(pyodbc.drivers())
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from pathlib import Path
import sqlalchemy
from typing import Dict, List, Tuple
from ydata_profiling import ProfileReport
import matplotlib
import numpy as np

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
    
    # Get list of all tables
    cursor = conn.cursor()
    tables = cursor.tables(tableType='TABLE')
    print("\nAvailable tables in the database:")
    for table in tables:
        table_name = table.table_name
        # Get row count for each table
        cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        row_count = cursor.fetchone()[0]
        print(f"Table: {table_name}, Row count: {row_count:,}")
    
    table_name = "TAN_ANALYSIS" 
    query = f"SELECT * FROM [{table_name}]"
    df = pd.read_sql(query, conn)
    
    # Print DataFrame info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
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

def remove_low_area_rows(df: pd.DataFrame, min_area: float = 20.0):
    # drop all buildigns where the area is 0:
    df_without_zero = df.loc[df["Useful surface area Ak"]>=min_area, :]
    # drop all rows where the Useful surface area is not available:
    df_without_zero = df_without_zero.dropna(subset=["Useful surface area Ak"])
    return df_without_zero

def normalize_columns(df: pd.DataFrame):
    numeric_cols = [i for i in df.select_dtypes(include='number').columns if not "area" in i]
    for col in numeric_cols:
        if col != "Month" and col != "Year" and col != "Useful surface area Ak":
            df[f"{col} per m2"] = df[col] / df["Useful surface area Ak"]
    return df

NORMALIZED_COLUMN_MAP: Dict[str, str] = {
    "Quantity": "normalized Quantity",
    "kWh": "normalized kWh",
    "Primary energy": "normalized Primary energy",
}


def remove_based_on_yearly_kWh_consumption_per_square_meter(df: pd.DataFrame, max_value: float = 300, monthly_percentage: float = 0.25) -> pd.DataFrame:
    # now remove outliers based on the kWh consumption per m2 per year: max consumption is 300kWh/m2/year
    # first calculate the kWh consumption per m2 per year by summing up the consumption of every building over all the months in a year
    df = df.copy()
    df['annual_kwh_m2'] = (df['kWh per m2']).groupby([df['ISGE object code'], df['Year']]).transform('sum')
    #  remove rows where the monthly consumption is higher the 25% of the annual consumption
    df = df.loc[df["kWh per m2"] <= df["annual_kwh_m2"] * monthly_percentage, :]

    # re-calculate the annual kWh consumption per m2 per year so a monthly outlier does not kick the whole year
    df.drop(columns=["annual_kwh_m2"], inplace=True)
    df['annual_kwh_m2'] = (df['kWh per m2']).groupby([df['ISGE object code'], df['Year']]).transform('sum')

    # drop buildings that have a higher annual kWh consumption per m2 than the max value
    out = df.loc[df["annual_kwh_m2"] < max_value, :].drop(columns=["annual_kwh_m2"]).copy()
    return out

def choose_factor(r, low, high, target):
    cands = np.array([0.001,0.01, 0.1, 1.0, 10.0, 100.0, 1000])
    # r is current ratio = qty/kWh
    # new_ratio = r / factor  (because kWh_new = kWh * factor)
    new = r / cands
    # prefer those inside band; otherwise pick closest to target
    inside = (new >= low) & (new <= high)
    if inside.any():
        # pick the factor that lands closest to target among inside
        idx = np.argmin(np.abs(new[inside] - target))
        return cands[inside][idx]
    # none inside: pick nearest to target
    idx = np.argmin(np.abs(new - target))
    return cands[idx]

def fix_conversion_errors_and_remove_outliers(df: pd.DataFrame, ) -> pd.DataFrame:
    normalized_cols = [col for col in df.columns if "per m2" in col]
    cleaned_df = pd.DataFrame()
    groups = df.groupby("Energy source")

    for source, group in groups:
        # Choose column based on energy source
        value_column = 'Quantity per m2' if source == 'Water' else 'kWh per m2'
        
        # if source is Electric energy remove rows with consumption higher than 500kWh/m2
        if source == "Electric energy":
            group_clean = group.loc[group[value_column] <= 500, :]
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] >= 0, :]
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        # if the source is Water remove rows with consumption higher than 0.5 m3/m2
        elif source == "Water":
            group_clean = group.loc[group[value_column] <= 0.5, :]
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] >= 0, :]
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        # if the source is Extra light fuel oil
        elif source == "Extra light fuel oil":
            # first fix the conversion from quantity (l) into kWh, here we see obvious errors in the data: there are 3 clusters with conversion factors around 0.09 (correct), 1 and 10
            # the conversion factors around 1 and 10 are scaled down as they are most likely wrong by manual calcualtion of the conversion
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] >= 0, :]

            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # physical band and target for extra light fuel oil: 10-10.6 kWH per liter
            low, high = 0.08, 0.11
            target = 0.095
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        elif source == "Natural gas":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] >= 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # natural gas ratio should be aournd 0.09 and 0.11 because 10-11 kWh per m3
            low, high = 0.09, 0.11
            target = 0.01
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))

            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        elif source == "Wood chips":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]

            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # physical band and target for wood chips: 2000-4500 kWh/ton -> ratio: 0.00025-0.0005
            low, high = 0.0002, 0.0005
            target = 0.00035
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)
        
        elif source == "LPG (Liquefied Petroleum Gas)":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for LPG the ratio should be around 13 kWh per kg so target = 0.077 kg/kWh
            low, high = 0.07, 0.085
            target = 0.077
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        elif source == "Pellets":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for pellets the ratio should be around 1t / 4600-5000 kWh 
            high, low = 0.00019, 0.00026
            target = 0.00021
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)


        elif source == "Firewood":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for firewood the ratio should be around 1prm / 1500 - 2100 kWh strongly dependent on type of wood and especially moisture content
            high, low = 0.0004, 0.0008
            target = 0.00058
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)


        elif source == "Medium fuel oil":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for medium fuel oil the ratio should be around 11.4 kWh/kg
            high, low = 0.085, 0.093
            target = 0.088
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        elif source == "Bottled gas":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for bottled gas the ratio should be around 13 kWh per kg
            high, low = 0.07, 0.085
            target = 0.077
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)


        elif source == "Briquettes":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for briquettes the ratio should be around 5 kWh/kg
            high, low = 0.18, 0.21
            target = 0.2
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        elif source == "Diesel":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for diesel the ratio should be around 10.5 kWh/l
            high, low = 0.09, 0.11
            target = 0.95
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        elif source == "Light fuel oil":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for light fuel oil the ratio should be around 10.5 kWh/l
            high, low = 0.09, 0.11
            target = 0.95
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        elif source == "Steam":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                conversion_ratio = pd.to_numeric(group_clean["Quantity"]) / pd.to_numeric(group_clean["kWh"])
            # for steam the ratio should be around 600-700 kWh/ton depending on pressure and temperature (4-11bar). I will use a range of 400 to 800 to be on the safe side
            high, low = 1/720, 1/500
            target = 1/650
            factor = pd.Series(1.0, index=group_clean.index)
            factor = conversion_ratio.apply(choose_factor, args=(low, high, target))
            group_clean.loc[:, ["kWh", "kWh per m2"]] = group_clean.loc[:, ["kWh", "kWh per m2"]].multiply(factor, axis=0)
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)

        elif source == "Heat":
            group_clean = group.copy()
            # remove negative values
            group_clean = group_clean.loc[group_clean[value_column] > 0, :]
            # heat is already provided in kWh - no conversion needed
            # remove outliers based on the yearly kWh consumption per square meter
            group_clean = remove_based_on_yearly_kWh_consumption_per_square_meter(group_clean, max_value=300, monthly_percentage=0.25)
            cleaned_df = pd.concat([cleaned_df, group_clean], ignore_index=True)
     
    return cleaned_df








def save_normalized_comparison_html(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    output_path: Path,
    top_n_sources: int = 8,
) -> None:
    """Persist a lightweight HTML figure that compares summary stats before/after filtering."""

    metrics = [
        (NORMALIZED_COLUMN_MAP["kWh"], "kWh per area"),
        (NORMALIZED_COLUMN_MAP["Quantity"], "Quantity per area"),
        (NORMALIZED_COLUMN_MAP["Primary energy"], "Primary energy per area"),
    ]

    before_norm = add_normalized_columns(before_df)
    after_norm = add_normalized_columns(after_df)

    if top_n_sources:
        top_sources = (
            before_norm["Energy source"]
            .value_counts()
            .head(top_n_sources)
            .index
            .tolist()
        )
    else:
        top_sources = before_norm["Energy source"].dropna().unique().tolist()

    def _summarise_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
        filtered = df[df["Energy source"].isin(top_sources)][["Energy source", metric_col]].copy()
        filtered[metric_col] = filtered[metric_col].replace([np.inf, -np.inf], np.nan)
        filtered = filtered.dropna(subset=[metric_col])

        if filtered.empty:
            return pd.DataFrame(
                columns=["Energy source", "median", "p25", "p75", "p95", "count"]
            )

        def _percentile(series: pd.Series, q: float) -> float:
            arr = series.dropna().values
            if arr.size == 0:
                return float("nan")
            return float(np.percentile(arr, q))

        summary = (
            filtered.groupby("Energy source")[metric_col]
            .agg(
                median=lambda s: float(np.median(s.dropna())) if s.dropna().size else float("nan"),
                p25=lambda s: _percentile(s, 25),
                p75=lambda s: _percentile(s, 75),
                p95=lambda s: _percentile(s, 95),
                count=lambda s: int(s.dropna().size),
            )
            .reset_index()
        )

        if not summary.empty:
            summary["Energy source"] = pd.Categorical(
                summary["Energy source"], categories=top_sources, ordered=True
            )
            summary = summary.sort_values("Energy source").reset_index(drop=True)
        return summary

    subplot_titles = [label for _, label in metrics]
    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
    )

    for row_idx, (metric_col, metric_label) in enumerate(metrics, start=1):
        before_summary = _summarise_metric(before_norm, metric_col)
        after_summary = _summarise_metric(after_norm, metric_col)

        combined = before_summary.merge(
            after_summary,
            on="Energy source",
            how="outer",
            suffixes=("_before", "_after"),
        )

        if combined.empty:
            continue

        combined["Energy source"] = pd.Categorical(
            combined["Energy source"], categories=top_sources, ordered=True
        )
        combined = combined.sort_values("Energy source").reset_index(drop=True)

        combined = combined.fillna({
            "count_before": 0,
            "count_after": 0,
        })

        x_values = combined["Energy source"].astype(str)

        before_custom = combined[["p25_before", "p75_before", "p95_before", "count_before"]].to_numpy()
        after_custom = combined[["p25_after", "p75_after", "p95_after", "count_after"]].to_numpy()

        before_error_upper = np.nan_to_num(
            (combined["p75_before"] - combined["median_before"]).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
        )
        before_error_lower = np.nan_to_num(
            (combined["median_before"] - combined["p25_before"]).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
        )
        after_error_upper = np.nan_to_num(
            (combined["p75_after"] - combined["median_after"]).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
        )
        after_error_lower = np.nan_to_num(
            (combined["median_after"] - combined["p25_after"]).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
        )

        fig.add_trace(
            go.Bar(
                x=x_values,
                y=combined["median_before"],
                name="Before (median)",
                marker_color="#636EFA",
                legendgroup="before",
                offsetgroup="before",
                customdata=before_custom,
                error_y=dict(
                    type="data",
                    array=before_error_upper,
                    arrayminus=before_error_lower,
                    visible=True,
                ),
                hovertemplate=(
                    "Energy source: %{x}<br>" +
                    "Median: %{y:.3f}<br>" +
                    "P25: %{customdata[0]:.3f}<br>" +
                    "P75: %{customdata[1]:.3f}<br>" +
                    "P95: %{customdata[2]:.3f}<br>" +
                    "Count: %{customdata[3]}<extra></extra>"
                ),
                showlegend=row_idx == 1,
            ),
            row=row_idx,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=x_values,
                y=combined["median_after"],
                name="After (median)",
                marker_color="#EF553B",
                legendgroup="after",
                offsetgroup="after",
                customdata=after_custom,
                error_y=dict(
                    type="data",
                    array=after_error_upper,
                    arrayminus=after_error_lower,
                    visible=True,
                ),
                hovertemplate=(
                    "Energy source: %{x}<br>" +
                    "Median: %{y:.3f}<br>" +
                    "P25: %{customdata[0]:.3f}<br>" +
                    "P75: %{customdata[1]:.3f}<br>" +
                    "P95: %{customdata[2]:.3f}<br>" +
                    "Count: %{customdata[3]}<extra></extra>"
                ),
                showlegend=row_idx == 1,
            ),
            row=row_idx,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=combined["p95_before"],
                name="Before p95",
                mode="markers",
                marker=dict(color="#1F77B4", symbol="triangle-up", size=8),
                legendgroup="before",
                hovertemplate="Energy source: %{x}<br>P95: %{y:.3f}<extra></extra>",
                showlegend=row_idx == 1,
            ),
            row=row_idx,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=combined["p95_after"],
                name="After p95",
                mode="markers",
                marker=dict(color="#D62728", symbol="triangle-up", size=8),
                legendgroup="after",
                hovertemplate="Energy source: %{x}<br>P95: %{y:.3f}<extra></extra>",
                showlegend=row_idx == 1,
            ),
            row=row_idx,
            col=1,
        )

        fig.update_yaxes(title_text=metric_label, row=row_idx, col=1)
        fig.update_xaxes(tickangle=-45, row=row_idx, col=1)

    fig.update_layout(
        height=350 * len(metrics),
        width=1200,
        template="plotly_white",
        barmode="group",
        title="Area-normalised energy metrics summary (before vs after filtering)",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


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
    # clean dataset
    # (1) remove buildings with zero area
    df_no_small_area = remove_low_area_rows(df, min_area=20.0)
    # (2) normalize consumption values by area
    df_normalized = normalize_columns(df_no_small_area)
    # (3) clean dataset by removing outliers: first the conversion errors from the measured quantity into kWh are fixed 
    # so not too much of the data is lost and most likely the conversion error occured due to manual inputs. 
    # Then all buildings which have a higher yearly consumption of more than 300 kWh/m2 are removed and monthly values with 
    # more than 25% of the yearly consumption are also removed. Rows where the electricity consumption is higher than 500 kWh/m2 are also removed.
    # If the water consumption is higher than 0.5 m3/m2 per month, the row is also removed. Heat is just removed based on the yearly consumption.
    df_clean = fix_conversion_errors_and_remove_outliers(df_normalized)





    figure = plot_energy_source_consumption(df_clean)
    figure.savefig(FIGURE_FOLDER / "energy_source_consumption_cleaned.svg",)# dpi=300)
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


    create_stackplot_positive_negative_y(df, "original")
    create_stackplot_positive_negative_y(df_clean, "clean")



    


if __name__ == "__main__":
    db_path = Path(r"/home/users/pmascherbauer/projects4/workspace_nikolausd/ReLIFE/ReLIFE_datasets/Croatia_public_buildings_data.sqlite")
    FIGURE_FOLDER = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/ReLIFE/figures")
    # database_path = r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\ReLIFE\data\ISGE_potrošnja_2020-2024.accdb"
    # df = translate_db(read_accdb_database(path2db=database_path))
    # save_to_sqlite(df=df, 
    #                db_path=db_path / "Croatia_public_buildings_data.db", 
    #                table_name="Energy_data")
    
    plot_statistics(db_path=db_path)
