import pandas as pd
import pyodbc
print(pyodbc.drivers())
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
from pathlib import Path

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


def plot_statistics(df: pd.DataFrame):
    for (year, month), group in df.groupby(["Year", "Month"]):
        energy = group.groupby("Energy source")["kWh"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(8,8))
        wedges, texts, autotexts = ax.pie(
            energy["kWh"], labels=None, autopct='%1.1f%%', startangle=90
        )
        ax.legend(wedges, energy["Energy source"], title="Energy Source", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

        # Calculate total energy consumption
        total_kWh = energy["kWh"].sum()

        # Define threshold for 5%
        threshold = 0.05 * total_kWh

        # Separate categories above and below the threshold
        above_threshold = energy[energy["kWh"] >= threshold].copy()
        below_threshold = energy[energy["kWh"] < threshold].copy()

        # Sum all small categories into one
        if not below_threshold.empty:
            other_sum = below_threshold["kWh"].sum()
            above_threshold = above_threshold.append(
                {"Energy source": "Other", "kWh": other_sum}, ignore_index=True
            )


        # Plot new pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            above_threshold["kWh"],
            labels=None,  # No direct labels to avoid clutter
            autopct='%1.1f%%',
            startangle=90,
        )

        # Add external legend
        ax.legend(wedges, above_threshold["Energy source"], title="Energy Source", loc="center left", bbox_to_anchor=(1, 0.5))

        # Set title

        # Show plot
        plt.show()


if __name__ == "__main__":

    database_path = r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\ReLIFE\data\ISGE_potrošnja_2020-2024.accdb"
    df = translate_db(read_accdb_database(path2db=database_path))
    save_to_sqlite(df=df, 
                   db_path=Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\ReLIFE\data") / "Croatia_public_buildings_data.sqlite", 
                   table_name="Energy_data")


