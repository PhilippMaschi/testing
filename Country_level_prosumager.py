import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlalchemy
from typing import List
import random


EUROPEAN_COUNTRIES = {
    'AUT': 'Austria',  #
    'BEL': 'Belgium',  #
    'BGR': 'Bulgaria',  #
    'HRV': 'Croatia',  #
    "CYP": "Cyprus",
    'CZE': 'Czech Republic',
    'DNK': 'Denmark',  #
    'EST': 'Estonia',  #
    'FIN': 'Finland',  #
    'FRA': 'France',  #
    'DEU': 'Germany',  #
    'GRC': 'Greece',  #
    'HUN': 'Hungary',  #
    'IRL': 'Ireland',  #
    'ITA': 'Italy',  #
    'LVA': 'Latvia',  #
    'LTU': 'Lithuania',  #
    'LUX': 'Luxembourg',  #
    'MLT': 'Malta',  #
    'NLD': 'Netherlands',  #
    'POL': 'Poland',  #
    'PRT': 'Portugal',  #
    'ROU': 'Roumania',  #
    'SVK': 'Slovakia',  #
    'SVN': 'Slovenia',  #
    'ESP': 'Spain',  #
    'SWE': 'Sweden'  #
}

# match ID to size
DHWTANK = {
    1: 0,
    2: 300,   # l
    3: 700,
}

BUFFERTANK = {
    1: 0,
    2: 700,   # l
    3: 1500
}

BOILER = {
    1: "air HP",
    2: "ground HP"
}


# no PV
# no battery



class DB:
    def __init__(self, path):
        self.engine = sqlalchemy.create_engine(f'sqlite:///{path}')
        self.metadata = sqlalchemy.MetaData()
        self.metadata.reflect(bind=self.engine)

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
            if len(column_names) > 1:
                query = sqlalchemy.select(*[table.columns[name] for name in column_names])
            else:
                query = sqlalchemy.select([table.columns[name] for name in column_names][0])
        else:
            query = sqlalchemy.select(table)

        if filter:
            for key, value in filter.items():
                query = query.where(table.columns[key] == value)
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)


def read_parquet(table_name: str, scenario_ID: int, folder: Path, column_names: List[str] = None) -> pd.DataFrame:
    """
    Returns: dataframe containing the results for the table name and specific scenario ID
    """
    file_name = f"{table_name}_{scenario_ID}.parquet.gzip"
    path_to_file = folder / file_name
    if column_names:
        df = pd.read_parquet(path=path_to_file, engine="auto", columns=column_names)
    else:
        df = pd.read_parquet(path=path_to_file, engine="auto")
    return df


def load_electricity_demand_profiles(scenario_ids: list, folder: Path) -> (pd.DataFrame, pd.DataFrame):
    opts = []
    simus = []
    for scen_id in scenario_ids:
        opts.append(read_parquet(table_name="OperationResult_OptimizationHour", scenario_ID=scen_id, folder=folder, column_names=["Grid", "ID_Scenario"]))
        simus.append(read_parquet(table_name="OperationResult_ReferenceHour", scenario_ID=scen_id, folder=folder, column_names=["Grid", "ID_Scenario"]))
    
    df_opt = pd.concat(opts)
    df_simus = pd.concat(simus)
    return df_opt, df_simus

def calculate_the_counts_of_each_id(prob_dhw, prob_buffer, num_buildings):
    subcategory_probabilities = [
        (dhw, buffer, prob_dhw * prob_buffer if dhw and buffer else
        prob_dhw * (1 - prob_buffer) if dhw and not buffer else
        (1 - prob_dhw) * prob_buffer if not dhw and buffer else
        (1 - prob_dhw) * (1 - prob_buffer)) 
        for dhw in [False, True]
        for buffer in [False, True]
    ]
        # Scale the probabilities to match the number buildings
    total_prob = sum(prob for _, _, prob in subcategory_probabilities)
    scaled_counts = [int(round(prob / total_prob * num_buildings)) for _, _, prob in subcategory_probabilities]

    # Adjust the counts to ensure the total is exactly 1000
    difference = num_buildings - sum(scaled_counts)
    if difference != 0:
        for i in range(abs(difference)):
            index = i % len(scaled_counts)
            scaled_counts[index] += 1 if difference > 0 else -1
    
    return scaled_counts

def define_tech_scenario(prob_dhw: float, prob_buffer: float, df_scenarios: pd.DataFrame, df_hp: pd.DataFrame):
    """based on the adaption values (0 to 1) the buildings having the technology are randomly selected"""
    random.seed(42)
    results = {  # the ids for the tanks in order in which they come ot of calculate the counts of each id function
        0: {"ID_HotWaterTank": [1], "ID_SpaceHeatingTank": [1]},
        1: {"ID_HotWaterTank": [1], "ID_SpaceHeatingTank": [2, 3]},
        2: {"ID_HotWaterTank": [2, 3], "ID_SpaceHeatingTank": [1]},
        3: {"ID_HotWaterTank": [2, 3], "ID_SpaceHeatingTank": [2, 3]},
    }
    scenarios_with_numbers = []
    for building_id, group in df_scenarios.groupby("ID_Building"):
        building = df_hp[df_hp["ID_Building"]==building_id].copy()
        num_buildings = int(building["number_buildings_heat_pump_air"][0] + building["number_buildings_heat_pump_ground"][0])


        counts = calculate_the_counts_of_each_id(prob_dhw, prob_buffer, num_buildings)
        for index, id_dict in results.items():
            group.loc[(group.loc[:, "ID_HotWaterTank"].isin(id_dict["ID_HotWaterTank"])) & (group.loc[:, "ID_SpaceHeatingTank"].isin(id_dict["ID_SpaceHeatingTank"])), "number_buildings"] = counts[index]

        scenarios_with_numbers.append(group)





def get_sqlite_result_file(folder_name: Path):
    db = DB(path=folder_name / f"{folder_name.name}.sqlite")
    scenario_table = db.read_dataframe(table_name="OperationScenario")

    # filter out only buildings with air HP and ground HP
    building_table = db.read_dataframe(table_name="OperationScenario_Component_Building")
    hp_df = building_table.loc[building_table["number_buildings_electricity"] == 0, :]

    filtered_scenarios = scenario_table.loc[scenario_table["ID_Building"].isin(list(hp_df.ID_Building)), :]

    define_tech_scenario(
        prob_dhw=0.1,
        prob_buffer=0.1,
        df_scenarios=filtered_scenarios,
        df_hp=hp_df
    )



def main():
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/data/output/")
    folder_names = [f"D5.4_{country}_{year}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2020, 2030, 2040, 2050]]
    folder_names_low_scenario = [f"D5.4_low_{country}_{year}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2020, 2030, 2040, 2050]] 


    folder = path_2_model_results / folder_names[0]
    get_sqlite_result_file(folder_name=folder)





if __name__ == "__main__":
    main()
