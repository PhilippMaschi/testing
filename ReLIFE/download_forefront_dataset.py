import httpx
import pandas as pd
import sqlalchemy
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict
from access_token import maschi_acceess_token

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

    path_to_sqlite = Path(__file__).parent / "ForeFront_data.sqlite"
    db = SQLITE_DB(path=path_to_sqlite)

    epc_df = data_to_dataframe(endpoint="/api/v1/epc")
    bbr_df = data_to_dataframe(endpoint="/api/v1/bbr")
    db.write_dataframe(
        table_name="EPC_data",
        data_frame=epc_df,
        if_exists="replace"
    )
    db.write_dataframe(
        table_name="BBR_data",
        data_frame=bbr_df,
        if_exists="replace"
    )
    print("saved EPC and BBR data")

    for heat_id in list(bbr_df["heat_meter_id"]):
        if heat_id == None:
            continue

        heat_meter_df = data_to_dataframe(endpoint="/api/v1/shmdataid", params={"heat_meter_ids": heat_id})
        db.write_dataframe(
            table_name="Heat_meter_data",
            data_frame=heat_meter_df,
            if_exists="append"
        )
    print("downloaded heat meter data")

    for water_id in list(bbr_df["water_meter_id"]):
        if water_id == None:
            continue
        water_df = data_to_dataframe(endpoint="/api/v1/swmdataid", params={"water_meter_ids": water_id})
        db.write_dataframe(
            table_name="Water_meter_data",
            data_frame=water_df,
            if_exists="append"
        )
    print("dowloaded water meter data")















