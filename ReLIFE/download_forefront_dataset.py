import httpx
import pandas as pd
import sqlalchemy
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict
from access_token import maschi_acceess_token
import tqdm

class SQLITE_DB:
        def __init__(self, path: Path):
            self.engine = sqlalchemy.create_engine(f'sqlite:///{path}')
            self.metadata = sqlalchemy.MetaData()
            self.metadata.reflect(bind=self.engine)

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

        def table_exists_and_has_data(self, table_name: str) -> bool:
            """Check if a table exists and contains data.
            
            Args:
                table_name (str): Name of the table to check.
                
            Returns:
                bool: True if table exists and has at least one row, False otherwise.
            """
            try:
                # Refresh metadata to get current state
                self.metadata.reflect(bind=self.engine)
                
                if table_name not in self.metadata.tables:
                    return False
                
                # Check if table has any data
                with self.engine.connect() as conn:
                    result = conn.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.scalar()
                    return count > 0
            except Exception:
                return False

        def get_existing_ids(self, table_name: str, id_column: str) -> set:
            """Get set of existing IDs from a table column.
            
            Args:
                table_name (str): Name of the table.
                id_column (str): Name of the ID column.
                
            Returns:
                set: Set of existing IDs, empty set if table doesn't exist.
            """
            try:
                if not self.table_exists_and_has_data(table_name):
                    return set()
                
                df = self.read_dataframe(table_name, column_names=[id_column])
                return set(df[id_column].dropna().unique())
            except Exception:
                return set()
            

def fetch_data(endpoint: str, params: dict = None):
    """
    Fetch data from Forefront API and return as JSON.
    :param endpoint: API endpoint (e.g., 'measurements', 'buildings')
    :param params: Optional query parameters
    :return: JSON response
    """
    with httpx.Client() as client:
        response = client.get(f"{API_BASE_URL}{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()  # Raises an error if request fails
        return response.json()

def data_to_dataframe(endpoint: str, params: dict = None):
    """
    Fetch API data and convert it to a Pandas DataFrame.
    :param endpoint: API endpoint
    :param params: Optional query parameters
    :return: Pandas DataFrame
    """
    data = fetch_data(endpoint, params)
    
    # Convert JSON response to DataFrame
    df = pd.DataFrame(data)
    
    return df

API_BASE_URL = "https://forefront-api.build.aau.dk"

# Your access token (replace with actual token)
ACCESS_TOKEN = maschi_acceess_token

# Headers for authentication
HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}


if __name__ == "__main__":

    path_to_sqlite = Path(__file__).parent.parent.parent.parent / r"workspace_nikolausd/ReLIFE/ReLIFE_datasets" / "ForeFront_data.sqlite"
    db = SQLITE_DB(path=path_to_sqlite)

    # Download EPC data if not already present
    if db.table_exists_and_has_data("EPC_data"):
        print("EPC data already exists, skipping download")
        epc_df = db.read_dataframe("EPC_data")
    else:
        print("Downloading EPC data...")
        epc_df = data_to_dataframe(endpoint="/api/v1/epc")
        db.write_dataframe(
            table_name="EPC_data",
            data_frame=epc_df,
            if_exists="replace"
        )
        print("Saved EPC data")
    
    # Download BBR data if not already present
    if db.table_exists_and_has_data("BBR_data"):
        print("BBR data already exists, skipping download")
        bbr_df = db.read_dataframe("BBR_data")
    else:
        print("Downloading BBR data...")
        bbr_df = data_to_dataframe(endpoint="/api/v1/bbr")
        db.write_dataframe(
            table_name="BBR_data",
            data_frame=bbr_df,
            if_exists="replace"
        )
        print("Saved BBR data")

    # Get list of already processed heat meter IDs
    existing_heat_ids = db.get_existing_ids("Heat_meter_data", "heat_meter_id")
    heat_ids_to_process = [heat_id for heat_id in bbr_df["heat_meter_id"] if heat_id is not None and heat_id not in existing_heat_ids]
    
    print(f"Found {len(existing_heat_ids)} existing heat meter records")
    print(f"Need to process {len(heat_ids_to_process)} heat meter IDs")
    
    # Determine if we should replace or append
    if_exist = "replace" if len(existing_heat_ids) == 0 else "append"
    
    for heat_id in tqdm.tqdm(heat_ids_to_process):
        heat_meter_df = data_to_dataframe(endpoint="/api/v1/shmdataid", params={"heat_meter_ids": heat_id})
        if not heat_meter_df.empty:
            db.write_dataframe(
                table_name="Heat_meter_data",
                data_frame=heat_meter_df,
                if_exists=if_exist
            )
            # After first successful write, switch to append mode
            if_exist = "append"
    print("Downloaded heat meter data")

    # Get list of already processed water meter IDs
    existing_water_ids = db.get_existing_ids("Water_meter_data", "water_meter_id")
    water_ids_to_process = [water_id for water_id in bbr_df["water_meter_id"] if water_id is not None and water_id not in existing_water_ids]
    
    print(f"Found {len(existing_water_ids)} existing water meter records")
    print(f"Need to process {len(water_ids_to_process)} water meter IDs")
    
    # Determine if we should replace or append
    if_exist = "replace" if len(existing_water_ids) == 0 else "append"
    
    for water_id in tqdm.tqdm(water_ids_to_process):
        water_df = data_to_dataframe(endpoint="/api/v1/swmdataid", params={"water_meter_ids": water_id})
        if water_df.empty:
            continue
        db.write_dataframe(
            table_name="Water_meter_data",
            data_frame=water_df,
            if_exists=if_exist
        )
        # After first successful write, switch to append mode
        if_exist = "append"
    print("Downloaded water meter data")















