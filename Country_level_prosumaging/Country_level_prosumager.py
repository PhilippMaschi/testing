import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlalchemy
from typing import List
import random
from country_level_prosumaging_figures import plot_supply_and_demand_matching_over_price
from joblib import Parallel, delayed
import os

CPU_COUNT = os.cpu_count()

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

COUNTRY_CODES = {
    'AUT': "AT",
    'BEL': "BE",
    'BGR': "BG",
    'HRV': "HR",
    "CYP": "CZ",
    'CZE': "CZ",
    'DNK': "DK",
    'EST': "ES",
    'FIN': "FI",
    'FRA': "FR",
    'DEU': "DE",
    'GRC': "GR",
    'HUN': "HU",
    'IRL': "IE",
    'ITA': "IT",
    'LVA': "LV",
    'LTU': "LT",
    'LUX': "LU",
    'MLT': "" ,
    'NLD': "" ,
    'POL': "PL",
    'PRT': "PT",
    'ROU': "RO",
    'SVK': "SK",
    'SVN': "" ,
    'ESP': "ES",
    'SWE': "SE",
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

def flexibility_factor(consumer_profile: np.array, prosumager_profile: np.array, national_demand_profile: np.array) -> float:
    """This ratio is the amount of energy shifted relative to the total energy consumed, indicating the proportion of demand that could be shifted."""
    difference = consumer_profile - prosumager_profile
    difference[difference<0] = 0  # only consider energy that is shifted "away", so the reduced energy

    return  difference.sum() / national_demand_profile.sum() * 100  # %

def flexibility_factor_hourly(consumer_profile: np.array, prosumager_profile: np.array, national_demand_profile: np.array) -> np.array:
    """This ratio is the amount of energy shifted relative to the total energy consumed, indicating the proportion of demand that could be shifted in every hour as percentage"""
    difference = consumer_profile - prosumager_profile
    difference[difference<0] = 0  # only consider energy that is shifted "away", so the reduced energy

    return  difference / national_demand_profile * 100  # %

def supply_and_demand_matching(price_profile: np.array, consumer_profile: np.array, prosumager_profile: np.array) -> dict:
    """set price threshold and if the price is below that threshold the used electricity is “renewable”. Check how much more “renewable” electricity can be used
    returns the differnce betwen prosumager demand at low prices and consumer demand at low prices"""
    match_increase = {}
    match_decrease = {}
    for quantile in np.arange(0, 1.05, 0.05):
        price_threshold = np.quantile(price_profile, quantile)
        consumer_match = 0
        prosumager_match = 0
        for i, price in enumerate(price_profile):
            if price <= price_threshold:
                diff =  prosumager_profile[i] - consumer_profile[i]
                if diff < 0:
                    match_decrease
                
        
        diff = prosumager_match - consumer_match
        diff[diff<0] = 0

    matching = []
    for i, price in enumerate(price_profile):
        diff =  prosumager_profile[i] - consumer_profile[i]
        matching.append(diff)
    x = pd.DataFrame({"price": price_profile, "change in electricity demand": matching})
    x.reset_index(drop=True).to_csv(Path(__file__).parent / "single_country_plots"/"change_demand.csv", index=False, sep=";")
    # match[price_threshold * 1_000] = diff

    return match 

def flexible_storage_efficiency(consumer_profile: np.array, prosumager_profile: np.array) -> float:
    """calculates the upward storage efficiency between 0 and 1"""
    # diff negative is the energy that is taken out of the building during "discharging"
    discharging_energy = prosumager_profile - consumer_profile
    discharging_energy[discharging_energy > 0] = 0

    # diff positve is the energy that is stored in the building during "charging"
    charging_energy = prosumager_profile - consumer_profile
    charging_energy[charging_energy < 0] = 0

    #  eta from https://doi.org/10.1016/j.apenergy.2017.04.061
    eta = 1 - (prosumager_profile - consumer_profile).sum() / charging_energy.sum()
    # if eta is negative than the prosumager uses less energy! which can be the case if there is PV for example
    return eta

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
    def load_data(scen_id):
        """Helper function to load data for a single scenario."""
        opt = read_parquet(
            table_name="OperationResult_OptimizationHour",
            scenario_ID=scen_id,
            folder=folder,
            column_names=["Grid", "ID_Scenario", "Hour"]
        )
        sim = read_parquet(
            table_name="OperationResult_ReferenceHour",
            scenario_ID=scen_id,
            folder=folder,
            column_names=["Grid", "ID_Scenario", "Hour"]
        )
        return opt, sim

    # Parallelize the loading of data
    results = Parallel(n_jobs=max(1, int(CPU_COUNT*0.75)))(delayed(load_data)(scen_id) for scen_id in scenario_ids)
    
    # Unpack results
    opts, simus = zip(*results)
    
    # Concatenate all results
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

def define_tech_scenario(prob_dhw: float, prob_buffer: float, df_scenarios: pd.DataFrame, df_hp: pd.DataFrame) -> pd.DataFrame:
    """based on the adaption values (0 to 1) the buildings having the technology are randomly selected"""
    results = {  # the ids for the tanks in order in which they come ot of calculate the counts of each id function
        0: {"ID_HotWaterTank": [1], "ID_SpaceHeatingTank": [1]},
        1: {"ID_HotWaterTank": [1], "ID_SpaceHeatingTank": [2, 3]},
        2: {"ID_HotWaterTank": [2, 3], "ID_SpaceHeatingTank": [1]},
        3: {"ID_HotWaterTank": [2, 3], "ID_SpaceHeatingTank": [2, 3]},
    }
    scenarios_with_numbers = []
    for building_id, group in df_scenarios.groupby("ID_Building"):
        building = df_hp[df_hp["ID_Building"]==building_id].copy()
        num_buildings = int(building["number_buildings_heat_pump_air"].values[0] + building["number_buildings_heat_pump_ground"].values[0])

        counts = calculate_the_counts_of_each_id(prob_dhw, prob_buffer, num_buildings)
        for index, id_dict in results.items():
            group.loc[(group.loc[:, "ID_HotWaterTank"].isin(id_dict["ID_HotWaterTank"])) & (group.loc[:, "ID_SpaceHeatingTank"].isin(id_dict["ID_SpaceHeatingTank"])), "number_buildings"] = counts[index]

        scenarios_with_numbers.append(group)
    
    scenario_numbers_df = pd.concat(scenarios_with_numbers).reset_index(drop=True)
    return scenario_numbers_df





def get_country_load_profiles(folder_name: Path, percentage_dhw_tanks: float, percentage_buffer_tanks: float):
    db = DB(path=folder_name / f"{folder_name.name}.sqlite")
    scenario_table = db.read_dataframe(table_name="OperationScenario")

    # filter out only buildings with air HP and ground HP
    building_table = db.read_dataframe(table_name="OperationScenario_Component_Building")
    hp_df = building_table.loc[building_table["number_buildings_electricity"] == 0, :]

    filtered_scenarios = scenario_table.loc[scenario_table["ID_Building"].isin(list(hp_df.ID_Building)), :]

    scenario_df = define_tech_scenario(
        prob_dhw=percentage_dhw_tanks,
        prob_buffer=percentage_buffer_tanks,
        df_scenarios=filtered_scenarios,
        df_hp=hp_df
    )

    opt_loads, ref_loads = load_electricity_demand_profiles(scenario_ids=list(scenario_df["ID_Scenario"]), folder=folder_name)

    # merge the numbers of each building scenario with the opt and ref loads:
    building_numbers = scenario_df.loc[:, ['ID_Scenario', "number_buildings"]].copy().set_index("ID_Scenario").sort_index()
    opt_stock = opt_loads.sort_values(by=["ID_Scenario", "Hour"]).set_index("ID_Scenario").join(building_numbers)
    opt_stock["opt_grid_demand_stock_MW"] = opt_stock["Grid"] * opt_stock["number_buildings"] / 1_000_000  # MW
    ref_stock = ref_loads.sort_values(by=["ID_Scenario", "Hour"]).set_index("ID_Scenario").join(building_numbers)
    ref_stock["ref_grid_demand_stock_MW"] = ref_stock["Grid"] * ref_stock["number_buildings"] / 1_000_000  # MW

    country_load_df = pd.concat([opt_stock, ref_stock["ref_grid_demand_stock_MW"]], axis=1)

    return country_load_df


def get_national_demand_profiles():
    gen_file = Path(__file__).parent.parent / "ENTSOE Generation" / "ENTSOE_generation_MWh_2019.csv"
    df = pd.read_csv(gen_file, sep=";")
    return df


def get_price_profile(folder_name: Path):
    db = DB(path=folder_name / f"{folder_name.name}.sqlite")
    price_table = db.read_dataframe(table_name="OperationScenario_EnergyPrice", column_names=["electricity_1"])
    return price_table


def main(percentage_dhw_tanks: float, percentage_buffer_tanks: float):
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects/Philipp/PycharmProjects/data/output/")
    folder_names = [f"D5.4_{country}_{year}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2020, 2030, 2040, 2050]]
    folder_names_low_scenario = [f"D5.4_low_{country}_{year}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2020, 2030, 2040, 2050]] 

    national_demand = get_national_demand_profiles()

    flexiblity_factor = {}
    flexiblity_factor_hourly = {}
    s_and_d_matching_dict = {}
    storage_efficiency = {}

    folder_names = folder_names[:4]
    for folder_name in folder_names:
        folder = path_2_model_results / folder_name
        country = folder.name.split("_")[-2]
        year = folder.name.split("_")[-1]
        print(f"analysing {country} {year}")

        country_loads = get_country_load_profiles(
            folder_name=folder,
            percentage_dhw_tanks=percentage_dhw_tanks,
            percentage_buffer_tanks=percentage_buffer_tanks
        )
        # add the country loads from different building types together:
        consumers_MW = np.array(country_loads.groupby(["Hour"]).sum()["ref_grid_demand_stock_MW"])
        prosumer_MW = np.array(country_loads.groupby(["Hour"]).sum()["opt_grid_demand_stock_MW"])

        price = np.array(get_price_profile(folder).iloc[:, 0])
        # calcualte the factors:
        flexiblity_factor[f"{country}_{year}"] = flexibility_factor(
            consumer_profile=consumers_MW, 
            prosumager_profile=prosumer_MW, 
            national_demand_profile=np.array(national_demand[COUNTRY_CODES[country]])
        )
        flexiblity_factor_hourly[f"{country}_{year}"] = flexibility_factor_hourly(
            consumer_profile=consumers_MW,
            prosumager_profile=prosumer_MW,
            national_demand_profile=np.array(national_demand[COUNTRY_CODES[country]])
        )

        s_and_d_matching_dict[f"{country}_{year}"]  = supply_and_demand_matching(
            price_profile=price,
            consumer_profile=consumers_MW,
            prosumager_profile=prosumer_MW
        )

        storage_efficiency[f"{country}_{year}"] = flexible_storage_efficiency(
            consumer_profile=consumers_MW,
            prosumager_profile=prosumer_MW
        )


        plot_supply_and_demand_matching_over_price(price=price,
                                                  s_d_match=s_and_d_matching_dict[f"{country}_{year}"],
                                                  country=country,
                                                  year=year)



if __name__ == "__main__":
    main(
        percentage_dhw_tanks=0.1, 
        percentage_buffer_tanks=0.1
    )

    # 10 cent/kWh levelized cost for offshore wind, onshore and PV are below that [https://www.ise.fraunhofer.de/en/publications/studies/cost-of-electricity.html]
    # 10 cent/kWh = 0.001 cent/Wh , I use this as my thresholds price 
