import pandas as pd
import pyodbc
print(pyodbc.drivers())


database_path = r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\ReLIFE\data\ISGE_potrošnja_2020-2024.accdb"

# Connection string
conn_str = (
    r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
    f"DBQ={database_path};"
)

# Establish connection
conn = pyodbc.connect(conn_str)

# List available tables
tables = pd.read_sql("SELECT Name FROM MSysObjects WHERE Type=1 AND Flags=0", conn)
print("Available tables:", tables)

# Specify table to load
table_name = "YourTableName"  # Replace with your table name

# Load data into pandas DataFrame
query = f"SELECT * FROM {table_name}"
df = pd.read_sql(query, conn)

# Close the connection
conn.close()
