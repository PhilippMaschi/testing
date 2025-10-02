import pandas as pd
import pyodbc
print(pyodbc.drivers())
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from collections import Counter
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

ENERGY_SOURCE_COLORS: Dict[str, str] = {
    "Electric energy": "#FFD54F",  # warm yellow
    "Water": "#1F78B4",  # clear blue
    "Extra light fuel oil": "#D3D3D3",  # light grey
    "Natural gas": "#E6550D",  # deep orange
    "Wood chips": "#2E8B57",  # forest green
    "Heat": "#D73027",  # vivid red
    "LPG (Liquefied Petroleum Gas)": "#F28E2B",  # amber orange
    "Pellets": "#8C6D31",  # warm brown
    "Firewood": "#654321",  # dark brown
    "Medium fuel oil": "#A9A9A9",  # medium grey
    "Steam": "#9467BD",  # soft purple
    "Bottled gas": "#F4A259",  # light orange
    "Briquettes": "#556B2F",  # olive green
    "Light fuel oil": "#BEBEBE",  # silver grey
    "Diesel": "#000000",  # black
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


def remove_based_on_yearly_kWh_consumption_per_square_meter(df: pd.DataFrame, max_value: float = 300, monthly_percentage: float = 0.25) -> pd.DataFrame:
    # now remove outliers based on the kWh consumption per m2 per year: max consumption is 300kWh/m2/year
    # first calculate the kWh consumption per m2 per year by summing up the consumption of every building over all the months in a year
    df = df.copy()
    df['annual_kwh_m2'] = (df['kWh per m2']).groupby([df['ISGE object code'], df['Year']]).transform('sum')
    #  remove rows where the monthly consumption is higher the 25% of the annual consumption
    df_no_outlier = df.loc[df["kWh per m2"] <= df["annual_kwh_m2"] * monthly_percentage, :].copy()

    # re-calculate the annual kWh consumption per m2 per year so a monthly outlier does not kick the whole year
    df_no_outlier.drop(columns=["annual_kwh_m2"], inplace=True)
    df_no_outlier['annual_kwh_m2'] = (df_no_outlier['kWh per m2']).groupby([df_no_outlier['ISGE object code'], df_no_outlier['Year']]).transform('sum')

    # drop buildings that have a higher annual kWh consumption per m2 than the max value
    out = df_no_outlier.loc[df_no_outlier["annual_kwh_m2"] < max_value, :].copy()
    out.drop(columns=["annual_kwh_m2"], inplace=True)
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
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
   # Get unique energy sources
    energy_sources = df['Energy source'].unique()
    
    # Calculate number of rows and columns for subplot grid
    n_sources = len(energy_sources)
    n_cols = 2
    n_rows = (n_sources + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8*n_rows))
    axes = axes.flatten()
    palette = {source: ENERGY_SOURCE_COLORS.get(source, "black") for source in energy_sources}
    # Create time series for each energy source
    for idx, source in enumerate(energy_sources):
        # Filter data for current energy source
        source_data = df[df['Energy source'] == source].copy()
        
        # Create datetime index
        source_data['Date'] = pd.to_datetime(source_data[['Year', 'Month']].assign(day=1))
        
        # Choose column based on energy source
        value_column = 'Quantity per m2' if source == 'Water' else 'kWh per m2'
                
        # Plot
        ax = axes[idx]
        sns.boxplot(data=source_data, x='Date', y=value_column, ax=ax, color=palette[source], showfliers=False)

        # Customize plot
        ax.set_title(f'{source}', pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity per m2' if source == 'Water' else 'kWh per m2')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis to show values in thousands
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    
    # Remove empty subplots
    for idx in range(len(energy_sources), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig

def plot_mean_median_energy_source_consumption(df):
    """Create time series plots of mean and median consumption for each energy source.
    
    Args:
        df: DataFrame containing columns 'Energy source', 'kWh', 'Quantity', 'Year', and 'Month'
    """
    # Set font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
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

def plot_yearly_consumption_per_m2(df: pd.DataFrame):
    plot_df = df.loc[~df["Energy source"].isin(["Electric energy", "Water"]), :].copy()
    available_sources = sorted(plot_df["Energy source"].unique())

    palette: Dict[str, str] = {}
    for idx, source in enumerate(available_sources):
        palette[source] = ENERGY_SOURCE_COLORS.get(source, "black")
    plot_df.loc[plot_df["Energy source"]=="LPG (Liquefied Petroleum Gas)", "Energy source"] = "LPG"
    palette["LPG"] = palette["LPG (Liquefied Petroleum Gas)"]
    del palette["LPG (Liquefied Petroleum Gas)"]


    fig_box, ax_box = plt.subplots(figsize=(12, 8))
    sns.boxplot(
        data=plot_df,
        y="yearly_consumption_per_m2",
        x="Energy source",
        hue="Energy source",
        palette=palette,
        ax=ax_box,
        fliersize=3,
        linewidth=1.2,
    )
    ax_box.set_ylabel("Yearly consumption per m² [kWh]", fontsize=16)
    ax_box.set_xlabel("Energy source", fontsize=16)
    ax_box.tick_params(axis="x", labelsize=14, rotation=45)
    ax_box.tick_params(axis="y", labelsize=14)
    ax_box.grid(axis="x", color="0.85", linestyle="-", linewidth=0.8)

    fig_box.tight_layout()
    fig_box.savefig( FIGURE_FOLDER / "yearly_consumption_per_m2_by_energy_source_boxplot.svg" )
    plt.show()
    plt.close()

def create_sankey_diagram_for_heating_system_switch(switches: dict):
    if not switches:
        raise ValueError("No heating system switches provided.")

    # Collect individual switches as (from_source, to_source) pairs
    flows = []
    for mapping in switches.values():
        if not isinstance(mapping, dict):
            continue
        for source, target in mapping.items():
            if source is None or target is None:
                continue
            flows.append((source, target))

    if not flows:
        raise ValueError("Switch data is empty after filtering.")

    source_counts = Counter(src for src, _ in flows)
    target_counts = Counter(dst for _, dst in flows)
    link_counts = Counter(flows)

    left_nodes = sorted(source_counts.keys())
    right_nodes = sorted(target_counts.keys())

    left_y = [0.5] if len(left_nodes) <= 1 else np.linspace(0.1, 0.9, len(left_nodes)).tolist()
    right_y = [0.5] if len(right_nodes) <= 1 else np.linspace(0.1, 0.9, len(right_nodes)).tolist()

    node_labels: List[str] = []
    node_colors: List[str] = []
    node_x: List[float] = []
    node_y: List[float] = []

    source_indices: Dict[str, int] = {}
    for idx, source in enumerate(left_nodes):
        source_indices[source] = idx
        node_labels.append(f"{source}")
        node_colors.append(ENERGY_SOURCE_COLORS.get(source, "#999999"))
        node_x.append(0.0)
        node_y.append(left_y[idx])

    target_indices: Dict[str, int] = {}
    offset = len(node_labels)
    for idx, target in enumerate(right_nodes):
        target_indices[target] = offset + idx
        node_labels.append(f"{target}")
        node_colors.append(ENERGY_SOURCE_COLORS.get(target, "#999999"))
        node_x.append(1.0)
        node_y.append(right_y[idx])

    def _hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
        hex_value = hex_color.lstrip('#')
        if len(hex_value) != 6:
            return f"rgba(153, 153, 153, {alpha})"
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    link_sources: List[int] = []
    link_targets: List[int] = []
    link_values: List[int] = []
    link_colors: List[str] = []

    for (src, dst), value in link_counts.items():
        if src not in source_indices or dst not in target_indices:
            continue
        link_sources.append(source_indices[src])
        link_targets.append(target_indices[dst])
        link_values.append(value)
        link_colors.append(_hex_to_rgba(ENERGY_SOURCE_COLORS.get(src, "#999999")))

    sankey = go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=20,
            thickness=20,
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y,
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            color=link_colors,
        ),
    )

    fig = go.Figure(sankey)
    fig.update_layout(title_text="Heating System Switches", font_size=12)
    fig.write_image(FIGURE_FOLDER / "heating_system_switches.png")
    fig.write_image(FIGURE_FOLDER / "heating_system_switches.svg")


def check_for_heating_system_switch(df: pd.DataFrame):

    df_energy = df.loc[(~df["Energy source"].isin(["Electric energy", "Water", "Heat", "Steam"])) & (df["kWh"]!=0), :].copy()
    buildings_with_heating_system_switch = []
    heating_system_switch = {}
    for _, group in df_energy.groupby(["ISGE object code"]):
        if len(group["Energy source"].unique()) > 1:
            # heating system has been switched
            buildings_with_heating_system_switch.append(group["ISGE object code"].unique()[0])
            sorted_group = group.sort_values(by=["Year", "Month"])
            sources = sorted_group["Energy source"].unique()
            first_system = sources[0]
            last_system = sources[-1]
            heating_system_switch[group["ISGE object code"].unique()[0]] = {first_system: last_system}

    # check if the buildings do not all belong to the same building cluster (last digit of the ISGE object code)
    overall_building_codes = [b[:-1] for b in buildings_with_heating_system_switch]

    duplicates = {x for x in overall_building_codes if overall_building_codes.count(x) > 1}
    if len(duplicates) > 0:
        if len(duplicates) > 0:
            print("sub buildings are switched which would result in double countiny")
            print(duplicates)
    

    create_sankey_diagram_for_heating_system_switch(heating_system_switch)







def plot_statistics(db_path: Path):

    df_croatia = read_sqlite_db(db_path=db_path)
    # clean dataset
    # (1) remove buildings with zero area
    df_no_small_area = remove_low_area_rows(df_croatia, min_area=20.0)
    # (2) normalize consumption values by area
    df_normalized = normalize_columns(df_no_small_area)
    # (3) clean dataset by removing outliers: first the conversion errors from the measured quantity into kWh are fixed 
    # so not too much of the data is lost and most likely the conversion error occured due to manual inputs. 
    # Then all buildings which have a higher yearly consumption of more than 300 kWh/m2 are removed and monthly values with 
    # more than 25% of the yearly consumption are also removed. Rows where the electricity consumption is higher than 500 kWh/m2 are also removed.
    # If the water consumption is higher than 0.5 m3/m2 per month, the row is also removed. Heat is just removed based on the yearly consumption.
    df_clean = fix_conversion_errors_and_remove_outliers(df_normalized)

    # add yearly consumption per m2 to the whole dataframe
    df_clean["yearly_consumption_per_m2"] = df_clean.groupby(["ISGE object code", "Year", "Energy source"])["kWh per m2"].transform("sum")

    plot_yearly_consumption_per_m2(df_clean)
    # df_normalized["yearly_consumption_per_m2"] = df_clean.groupby(["ISGE object code", "Year", "Energy source"])["kWh per m2"].transform("sum")
    # plot_yearly_consumption_per_m2(df_normalized)

    check_for_heating_system_switch(df_clean)


    # figure = plot_mean_median_energy_source_consumption(df_clean)
    # figure.savefig(FIGURE_FOLDER / "energy_source_consumption_cleaned_mean_median.svg",)# dpi=300)
    # plt.show()
    # plt.close(figure)

    # figure = plot_energy_source_consumption(df_clean)
    # figure.savefig(FIGURE_FOLDER / "energy_source_consumption_cleaned_boxplot.svg",)# dpi=300)
    # plt.show()
    # plt.close(figure)

    # isge_groups = df.groupby(["ISGE object code", "Energy source"])
    # group = isge_groups.get_group(("HR-51410-0018-0", "Electric energy"))
    # profile = ProfileReport(df, title="Profiling Report")
    # output_path = db_path.parent / "data_profile.html"
    # profile.to_file(output_file=output_path)
    # profile.to_widgets()
    # print("saved y data profiling report")

    # for source in ["Water", "Electric energy"]:
    #     filtered = df.loc[df["Energy source"] == source, :]
    #     create_stackplot(filtered, source)

    # create_stackplot(df.loc[~df["Energy source"].isin(["Water","Electric energy", "Heat"]), :], y_axis="primary energy")
    # create_stackplot(df.loc[df["Energy source"].isin(["Heat"]), :], y_axis="heating energy")


    # create_stackplot_positive_negative_y(df, "original")
    # create_stackplot_positive_negative_y(df_clean, "clean")



    


if __name__ == "__main__":
    db_path = Path(r"/home/users/pmascherbauer/projects4/workspace_nikolausd/ReLIFE/ReLIFE_datasets/Croatia_public_buildings_data.sqlite")
    FIGURE_FOLDER = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/ReLIFE/figures")
    # database_path = r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\ReLIFE\data\ISGE_potrošnja_2020-2024.accdb"
    # df = translate_db(read_accdb_database(path2db=database_path))
    # save_to_sqlite(df=df, 
    #                db_path=db_path / "Croatia_public_buildings_data.db", 
    #                table_name="Energy_data")
    
    plot_statistics(db_path=db_path)
