import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlalchemy
from typing import List
import random
from joblib import Parallel, delayed
import os




CPU_COUNT = os.cpu_count()

EUROPEAN_COUNTRIES = {
    'AUT': 'Austria',  #
    'BEL': 'Belgium',  #
    'BGR': 'Bulgaria',  #
    'HRV': 'Croatia',  #
    # "CYP": "Cypru#",
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
    # 'MLT': 'Malta',  #
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
    "AT": 'AUT',
    "BE": 'BEL',
    "BG": 'BGR',
    "HR": 'HRV',
    "CY": "CYP",
    "CZ": 'CZE',
    "DK": 'DNK',
    "EE": 'EST',
    "FI": 'FIN',
    "FR": 'FRA',
    "DE": 'DEU',
    "GR": 'GRC',
    "HU": 'HUN',
    "IE": 'IRL',
    "IT": 'ITA',
    "LV": 'LVA',
    "LT": 'LTU',
    "LU": 'LUX',
    "MT": 'MLT' ,
    "NL": 'NLD' ,
    "PL": 'POL',
    "PL": "POL",
    "PT": 'PRT',
    "RO": 'ROU',
    "SK": 'SVK',
    "SI": 'SVN' ,
    "ES": 'ESP',
    "SE": 'SWE',
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


def get_national_demand_profiles():
    secures_path = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/Secures") / "Demand_all_countries.csv"
    if not secures_path.exists():
        demand_dfs = []
        for year in [2030, 2050]:
            for country in COUNTRY_CODES.keys():
                if country == "CY" or country == "MT":
                    continue
                columns_2_extract = ["Exogenous", "Elect2Heat", "Interseasonal storage (PS)", "Elec-demand H2 prod", "Trans-losses", "Intreasonal storage (battery)"]
                gen_file = Path(__file__).parent.parent.parent / "FLEX" / "projects" / "Secures" / "prices" / f"{year}" / f"{country}_DN_{year}_Normal.xlsx"
                df = pd.read_excel(
                    gen_file,
                    engine="openpyxl",                
                    sheet_name=f"EL_dem",
                    header=1
                )[columns_2_extract]
                df_1 = pd.concat([df, df.iloc[-24:, :]], axis=0).reset_index(drop=True)
                df_1["demand MW"] = df_1.sum(axis=1)
                df_1.drop(columns=columns_2_extract, inplace=True)
                df_1["country"] = COUNTRY_CODES[country]
                df_1["year"] = year
                df_1["Hour"] = np.arange(1, 8761)
                demand_dfs.append(df_1)

        big_df = pd.concat(demand_dfs, axis=0)
        big_df.to_csv(secures_path, index=False, sep=";")
    else:
        big_df = pd.read_csv(secures_path, sep=";")
    return big_df

    # entsoe = pd.read_csv(gen_file, sep=";")
    # demand = entsoe.melt(var_name="country", value_name="demand").reset_index(drop=True)
    # demand["year"] = 2020
    # demand["scenario"] = "baseyear"
    # levethian = pd.read_csv(Path(__file__).parent / "AURES_leviathan_demand.csv", sep=",").drop(columns="UNITS").reset_index(drop=True)
    # levethian["scenario"] = "levethian"
    # shiny_happy = pd.read_csv(Path(__file__).parent / "AURES_shiny_demand.csv", sep=",").drop(columns="UNITS").reset_index(drop=True)
    # shiny_happy["scenario"] = "shiny happy"

    # df = pd.concat([demand, shiny_happy, levethian], axis=0, ignore_index=True)
    # df["country"] = df["country"].map(COUNTRY_CODES)


    # # AURES data only contains 8735 values: duplicate the last ones
    # extended_data = []
    # for (country, year, scenario), group in df.groupby(["country", "year", "scenario"]):
    #     if len(group) != 8760:
    #         extended_values = group['demand'].tolist() + group["demand"].iloc[-24:].to_list()
    #         extended_group = pd.DataFrame({
    #             'country': [country] * 8760,
    #             'year': [year] * 8760,
    #             'scenario': [scenario] * 8760,
    #             'demand': extended_values,
    #             "Hour": np.arange(1, 8761)
    #         })
    #         extended_data.append(extended_group)
    #     else:
    #         group["Hour"] = np.arange(1, 8761)
    #         extended_data.append(group)
    # extended_df = pd.concat(extended_data, ignore_index=True)
    
    # replace the 2 digit country code with 3 digits
    assert not extended_df.country.isna().any()
    return extended_df


def flexibility_factor(consumer_profile: np.array, prosumager_profile: np.array, national_demand_profile: np.array) -> float:
    """This ratio is the amount of energy shifted relative to the total energy consumed, indicating the proportion of demand that could be shifted."""
    difference = np.maximum(consumer_profile - prosumager_profile, 0)
    return difference.sum() / national_demand_profile.sum() * 100  # %

def flexibility_factor_hourly(consumer_profile: np.array, prosumager_profile: np.array, national_demand_profile: np.array) -> np.array:
    """This ratio is the amount of energy shifted relative to the total energy consumed, indicating the proportion of demand that could be shifted in every hour as percentage"""
    difference = np.maximum(consumer_profile - prosumager_profile, 0)
    return difference / national_demand_profile * 100  # %

def supply_and_demand_matching(price_profile: np.array, consumer_profile: np.array, prosumager_profile: np.array) -> dict:
    """set price threshold and if the price is below that threshold the used electricity is “renewable”. Check how much more “renewable” electricity can be used
    returns the differnce betwen prosumager demand at low prices and consumer demand at low prices"""
    # match_increase = {}
    # match_decrease = {}
    # for quantile in np.arange(0, 1.05, 0.05):
    #     price_threshold = np.quantile(price_profile, quantile)
    #     consumer_match = 0
    #     prosumager_match = 0
    #     for i, price in enumerate(price_profile):
    #         if price <= price_threshold:
    #             diff =  prosumager_profile[i] - consumer_profile[i]
    #             if diff < 0:
    #                 match_decrease
                
        
    #     diff = prosumager_match - consumer_match
    #     diff[diff<0] = 0

    matching = []
    for i, price in enumerate(price_profile):
        diff =  prosumager_profile[i] - consumer_profile[i]
        matching.append(diff)
    df = pd.DataFrame({"price": price_profile, "change in electricity demand": matching})
    df.reset_index(drop=True, inplace=True)
    # match[price_threshold * 1_000] = diff

    return df 

def flexible_storage_efficiency(consumer_profile: np.array, prosumager_profile: np.array) -> float:
    """calculates the upward storage efficiency between 0 and 1"""
    # diff negative is the energy that is taken out of the building during "discharging"
    discharging_energy = prosumager_profile - consumer_profile
    discharging_energy[discharging_energy > 0] = 0

    # diff positve is the energy that is stored in the building during "charging"
    charging_energy = prosumager_profile - consumer_profile
    charging_energy[charging_energy < 0] = 0

    #  eta from https://doi.org/10.1016/j.apenergy.2017.04.061
    eta = 1 - np.sum(prosumager_profile - consumer_profile) / np.sum(charging_energy)
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
    file_name = f"{table_name}_S{scenario_ID}.parquet.gzip"
    path_to_file = folder / file_name
    return pd.read_parquet(path=path_to_file, columns=column_names) if column_names else pd.read_parquet(path=path_to_file)

def load_electricity_demand_profiles(scenario_ids: list, folder: Path) -> (pd.DataFrame, pd.DataFrame):
    def load_data(scen_id):
        """Helper function to load data for a single scenario."""
        opt = read_parquet(
            table_name=f"OperationResult_OptHour",
            scenario_ID=scen_id,
            folder=folder,
            column_names=["Grid", "ID_Scenario", "Hour", "PV2Grid", "PhotovoltaicProfile", "PV2Load"]
        )
        sim = read_parquet(
            table_name=f"OperationResult_RefHour",
            scenario_ID=scen_id,
            folder=folder,
            column_names=["Grid", "ID_Scenario", "Hour", "PV2Grid", "PhotovoltaicProfile", "PV2Load"]
        )
        return opt, sim

    # Parallelize the loading of data
    results = Parallel(n_jobs=max(1, int(CPU_COUNT * 0.75)))(delayed(load_data)(scen_id) for scen_id in scenario_ids)
    
    # Unpack results
    opts, simus = zip(*results)
    
    # Concatenate all results
    df_opt = pd.concat(opts)
    df_simus = pd.concat(simus)
    
    return df_opt, df_simus

def load_specific_heat_profiles(scenario_ids: list, folder: Path) -> (pd.DataFrame, pd.DataFrame):
    def load_data(scen_id):
        """Helper function to load data for a single scenario."""
        opt = read_parquet(
            table_name=f"OperationResult_OptHour",
            scenario_ID=scen_id,
            folder=folder,
            column_names=["ID_Scenario", "Hour", "Q_HeatingTank_bypass", "Q_HeatingTank_in", "Q_HeatingTank_out", "Q_DHWTank_in", "Q_DHWTank_bypass", "Q_DHWTank_out", "Q_RoomHeating"]
        )
        sim = read_parquet(
            table_name=f"OperationResult_RefHour",
            scenario_ID=scen_id,
            folder=folder,
            column_names=["ID_Scenario", "Hour", "Q_HeatingTank_bypass", "Q_HeatingTank_in", "Q_HeatingTank_out", "Q_DHWTank_in", "Q_DHWTank_bypass", "Q_DHWTank_out", "Q_RoomHeating"]
        )
        return opt, sim

    # Parallelize the loading of data
    results = Parallel(n_jobs=max(1, int(CPU_COUNT * 0.75)))(delayed(load_data)(scen_id) for scen_id in scenario_ids)
    opts, simus = zip(*results)
    df_opt = pd.concat(opts)
    df_simus = pd.concat(simus)
    
    return df_opt, df_simus

def get_shifted_energy_per_appliance_type(folder_name: Path, perc_cooling: float, price_id):
    db = DB(path=folder_name / "output" / f"{folder_name.name}.sqlite")
    scenario_table = db.read_dataframe(table_name="OperationScenario")

    # filter out only buildings with air HP and ground HP
    building_table = db.read_dataframe(table_name="OperationScenario_Component_Building")

    # return 0 for those..
    if building_table.empty:
        return pd.DataFrame.from_dict({"opt_grid_demand_stock_MW": np.zeros(8760), 
                                       "ref_grid_demand_stock_MW": np.zeros(8760),
                                       "Hour": np.arange(1, 8761)},  orient="index").T
    else:

        scenario_df = define_tech_scenario(
            prob_cooling=perc_cooling,
            df_scenarios=scenario_table,
            df_hp=building_table
        )
        scenario_df = scenario_df.loc[scenario_df["ID_EnergyPrice"]==price_id, :]
        opt_loads, ref_loads = load_specific_heat_profiles(scenario_ids=list(scenario_df["ID_Scenario"]), folder=folder_name / "output")

        # merge the numbers of each building scenario with the opt and ref loads:
        building_numbers = scenario_df.loc[:, ["ID_Scenario", "number_of_buildings", "ID_EnergyPrice"]].copy().set_index("ID_Scenario").sort_index()
        opt_stock = opt_loads.sort_values(by=["ID_Scenario", "Hour"]).set_index("ID_Scenario").join(building_numbers)
        cols = ["Q_HeatingTank_bypass", "Q_HeatingTank_in", "Q_HeatingTank_out", "Q_DHWTank_in", "Q_DHWTank_bypass", "Q_DHWTank_out", "Q_RoomHeating"]
        opt_stock[[col + "_stock" for col in cols]] = opt_stock[cols].mul(opt_stock["number_of_buildings"], axis=0)/ 1_000_000  # MW


        ref_stock = ref_loads.sort_values(by=["ID_Scenario", "Hour"]).set_index("ID_Scenario").join(building_numbers)
        ref_stock[[col + "_stock_ref" for col in cols]] = ref_stock[cols].mul(ref_stock["number_of_buildings"], axis=0)/ 1_000_000  # MW
        
        country_load_df = pd.concat([opt_stock, ref_stock[[col + "_stock_ref" for col in cols]]], axis=1)
        return country_load_df
    
def calculate_shifted_energy_per_appliance_type(perc_cooling, price_id):
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/")
    
    folder_names = [f"{country}_{year}_{PROJECT_ACRONYM}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2030, 2050]]
    dfs = []
    for folder_name in folder_names:
        if folder_name in [f"CYP_2020_{PROJECT_ACRONYM}", f"MLT_2020_{PROJECT_ACRONYM}"]:
            continue
        folder = path_2_model_results / folder_name
        country = folder.name.split("_")[0]
        year = folder.name.split("_")[1]
        print(f"calculating shited energy per appliance for {country} {year}")
    
        df = get_shifted_energy_per_appliance_type(folder, perc_cooling, price_id)
        df["country"] = country
        df["year"] = year

        # shifted energy through the thermal mass is the difference in Q_RoomHeating:
        df["thermal_mass_shift"] = df["Q_RoomHeating_stock"] - df["Q_RoomHeating_stock_ref"]
        df["thermal_mass_shift_increase"] = df["thermal_mass_shift"].clip(lower=0)
        df["thermal_mass_shift_decrease"] = df["thermal_mass_shift"].clip(upper=0)

        # shift in DHW is the difference of the DHW Tank input (before losses) and DHW Tank bypass
        df["DHW_shift"] = df["Q_DHWTank_in_stock"] + df["Q_DHWTank_bypass_stock"] - df["Q_DHWTank_in_stock_ref"] - df["Q_DHWTank_bypass_stock_ref"] 
        df["DHW_shift_increase"] = df["DHW_shift"].clip(lower=0)
        df["DHW_shift_decrease"] = df["DHW_shift"].clip(upper=0)

        # shift in Buffer Tank is the energy that goes into the tank as it is not used in the reference:
        df["Buffer_shift"] = df["Q_HeatingTank_in_stock"] + df["Q_HeatingTank_bypass_stock"] - df["Q_HeatingTank_bypass_stock_ref"]
        df["Buffer_shift_increase"] = df["Buffer_shift"].clip(lower=0)
        df["Buffer_shift_decrease"] = df["Buffer_shift"].clip(upper=0)
        cols = ["Q_HeatingTank_bypass", "Q_HeatingTank_in", "Q_HeatingTank_out", "Q_DHWTank_in", "Q_DHWTank_bypass", "Q_DHWTank_out",]# "Q_RoomHeating"]
        cols_opt = [f"{c}_stock" for c in cols]
        cols_ref = [f"{c}_stock_ref" for c in cols]
        df.drop(columns=cols+cols_ref+cols_opt+["number_of_buildings"], inplace=True)

        dfs.append(df)
    
    big_df = pd.concat(dfs, axis=0)
    big_df.columns = [str(col) for col in big_df.columns]
    return big_df

def define_tech_scenario(prob_cooling: float, df_scenarios: pd.DataFrame, df_hp: pd.DataFrame) -> pd.DataFrame:
    """Based on the adaption values (0 to 1) the buildings having the technology are randomly selected"""

    price_dfs = []
    for price_id in df_scenarios["ID_EnergyPrice"].unique():
        scenarios_with_numbers = []

        scenarios_price = df_scenarios.query(f"ID_EnergyPrice=={price_id}")
        # b_ids = scenarios_price["ID_Building"]
        # np.array([df_hp.loc[df_hp["ID_Building"]==i, "number_of_buildings"] for i in b_ids]).flatten().sum() / 2  # kommt was anderes raus als mit der berechnung die hier steht
        for building_id, group in scenarios_price.groupby("ID_Building"):
            # change the number of buildings in the scenario table by multiplying with the cooling proabability
            group.loc[group["ID_SpaceCoolingTechnology"]==1, "number_of_buildings"] = group.loc[group["ID_SpaceCoolingTechnology"]==1, "number_of_buildings"] * (1 - prob_cooling)
            group.loc[group["ID_SpaceCoolingTechnology"]==2, "number_of_buildings"] = group.loc[group["ID_SpaceCoolingTechnology"]==2, "number_of_buildings"] * prob_cooling

            scenarios_with_numbers.append(group)
    
        scenario_numbers_df = pd.concat(scenarios_with_numbers).reset_index(drop=True)
        price_dfs.append(scenario_numbers_df)
    
    total_scenario_df = pd.concat(price_dfs).reset_index(drop=True)
    return total_scenario_df

def get_total_number_of_buildings_with_HP(folder_name: Path, perc_cooling: float):
    db = DB(path=folder_name / "output" / f"{folder_name.name}.sqlite")
    scenario_table = db.read_dataframe(table_name="OperationScenario")
    building_table = db.read_dataframe(table_name="OperationScenario_Component_Building")
    scenario_df = define_tech_scenario(
        prob_cooling=perc_cooling,
        df_scenarios=scenario_table,
        df_hp=building_table
    )
    building_numbers = scenario_df.loc[:, ["ID_Scenario", "number_of_buildings", "ID_EnergyPrice"]].copy()
    df = building_numbers.groupby("ID_EnergyPrice")["number_of_buildings"].sum().reset_index()
    return df

def get_heating_demand(db: DB, include_opt: bool = True):
    if include_opt:
        opt = db.read_dataframe(table_name="OperationResult_OptYear", )
        ref = db.read_dataframe(table_name="OperationResult_RefYear")
        return (ref[["ID_Scenario", "Q_RoomHeating"]], opt[["ID_Scenario", "Q_RoomHeating"]])
    else:
        ref = db.read_dataframe(table_name="OperationResult_RefYear")
        return (ref[["ID_Scenario", "Q_RoomHeating"]], _)
    
def get_country_temperature_profile(folder_name: Path):
    db = DB(path=folder_name / "output" / f"{folder_name.name}.sqlite")
    temp_table = db.read_dataframe(table_name="OperationScenario_RegionWeather")
    return temp_table[["id_hour", "temperature", "pv_generation_optimal"]]
     
def get_country_specific_heating_demand(folder_name: Path, perc_cooling: float, include_opt: bool = True):
    db = DB(path=folder_name / "output" / f"{folder_name.name}.sqlite")
    scenario_table = db.read_dataframe(table_name="OperationScenario")
    building_table = db.read_dataframe(table_name="OperationScenario_Component_Building")

    ref_heating, opt_heating = get_heating_demand(db)
    scenario_df = define_tech_scenario(
        prob_cooling=perc_cooling,
        df_scenarios=scenario_table,
        df_hp=building_table
    )

    # add Af:
    scen_df = pd.merge(left=scenario_df, right=building_table[["ID_Building", "Af"]], on="ID_Building")

    building_numbers = scen_df.loc[:, ["ID_Scenario", "number_of_buildings", "ID_EnergyPrice", "Af"]].copy()
    if include_opt:
        opt_stock = pd.merge(left=building_numbers, right=opt_heating, on="ID_Scenario") 
        opt_stock["Q_RoomHeating_opt_kWh/m2"] = opt_stock["Q_RoomHeating"] / opt_stock["Af"] / 1_000
        opt_stock.drop(columns="Q_RoomHeating", inplace=True)
        stock = pd.merge(left=opt_stock, right=ref_heating, on="ID_Scenario")
        stock["Q_RoomHeating_ref_kWh/m2"] = stock["Q_RoomHeating"] / stock["Af"] / 1_000
        stock.drop(columns="Q_RoomHeating", inplace=True)

        avg_heating_ref = stock.groupby(["ID_EnergyPrice"])[["Q_RoomHeating_ref_kWh/m2", "number_of_buildings"]].apply(lambda x: (x['Q_RoomHeating_ref_kWh/m2'] * x['number_of_buildings']).sum() / x['number_of_buildings'].sum()).reset_index(name="Heating ref demand (kWh/m2)")
        avg_heating_opt = stock.groupby(["ID_EnergyPrice"])[["Q_RoomHeating_opt_kWh/m2", "number_of_buildings"]].apply(lambda x: (x['Q_RoomHeating_opt_kWh/m2'] * x['number_of_buildings']).sum() / x['number_of_buildings'].sum()).reset_index(name="Heating opt demand (kWh/m2)")

        df = pd.concat([avg_heating_ref, avg_heating_opt[["Heating opt demand (kWh/m2)"]]], axis=1)
    else:
        stock = pd.merge(left=building_numbers, right=ref_heating, on="ID_Scenario")
        stock["Q_RoomHeating_ref_kWh/m2"] = stock["Q_RoomHeating"] / stock["Af"] / 1_000
        stock.drop(columns="Q_RoomHeating", inplace=True)
        df = stock.groupby(["ID_EnergyPrice"])[["Q_RoomHeating_ref_kWh/m2", "number_of_buildings"]].apply(lambda x: (x['Q_RoomHeating_ref_kWh/m2'] * x['number_of_buildings']).sum() / x['number_of_buildings'].sum()).reset_index(name="Heating ref demand (kWh/m2)")

    return df

def get_country_load_profiles_for_PV_buidlings(folder_name: Path, perc_cooling: float):
    db = DB(path=folder_name / "output" / f"{folder_name.name}.sqlite")
    scenario_table = db.read_dataframe(table_name="OperationScenario")

    # filter out only buildings with air HP and ground HP
    building_table = db.read_dataframe(table_name="OperationScenario_Component_Building")

    # return 0 for those..
    if building_table.empty:
        return pd.DataFrame.from_dict({"opt_grid_demand_stock_MW": np.zeros(8760), 
                                       "ref_grid_demand_stock_MW": np.zeros(8760),
                                       "Hour": np.arange(1, 8761)},  orient="index").T
    else:

        scenario_df = define_tech_scenario(
            prob_cooling=perc_cooling,
            df_scenarios=scenario_table,
            df_hp=building_table
        )
        # define scenario ids that have PV
        ids = [i for i in scenario_df.loc[scenario_df["ID_PV"]!=1, "ID_Scenario"]]
        opt_loads, ref_loads = load_electricity_demand_profiles(scenario_ids=ids, folder=folder_name / "output")
         
        # merge the numbers of each building scenario with the opt and ref loads:
        building_numbers = scenario_df.loc[:, ["ID_Scenario", "number_of_buildings", "ID_EnergyPrice"]].copy().set_index("ID_Scenario").sort_index()
        opt_stock = opt_loads.sort_values(by=["ID_Scenario", "Hour"]).set_index("ID_Scenario").join(building_numbers) 
        opt_stock["opt_grid_demand_stock_MW"] = opt_stock["Grid"] * opt_stock["number_of_buildings"] / 1_000_000  # MW
        opt_stock["opt_PV2Grid_MW"] = opt_stock["PV2Grid"] * opt_stock["number_of_buildings"] / 1_000_000 # MW
        opt_stock["opt_PhotovoltaicProfile_MW"] = opt_stock["PhotovoltaicProfile"] * opt_stock["number_of_buildings"] / 1_000_000 # MW
        opt_stock["opt_PV2Load_MW"] = opt_stock["PV2Load"] * opt_stock["number_of_buildings"] / 1_000_000 # MW

        ref_stock = ref_loads.sort_values(by=["ID_Scenario", "Hour"]).set_index("ID_Scenario").join(building_numbers)
        ref_stock["ref_grid_demand_stock_MW"] = ref_stock["Grid"] * ref_stock["number_of_buildings"] / 1_000_000  # MW
        ref_stock["ref_PV2Grid_MW"] = ref_stock["PV2Grid"] * ref_stock["number_of_buildings"] / 1_000_000 # MW
        ref_stock["ref_PhotovoltaicProfile_MW"] = ref_stock["PhotovoltaicProfile"] * ref_stock["number_of_buildings"] / 1_000_000 # MW
        ref_stock["ref_PV2Load_MW"] = ref_stock["PV2Load"] * ref_stock["number_of_buildings"] / 1_000_000 # MW
        
        opt_stock_grouped = opt_stock.reset_index().groupby(["ID_Scenario", "ID_EnergyPrice"])
        ref_stock_grouped = ref_stock.reset_index().groupby(["ID_Scenario", "ID_EnergyPrice"])


        self_consumption_ref = ref_stock_grouped["ref_PV2Load_MW"].sum() / ref_stock_grouped["ref_PhotovoltaicProfile_MW"].sum() 
        self_consumption_opt = opt_stock_grouped["opt_PV2Load_MW"].sum() / opt_stock_grouped["opt_PhotovoltaicProfile_MW"].sum() 
        self_consumption_ref = self_consumption_ref.reset_index().rename(columns={0: "self_consumption"})
        self_consumption_opt = self_consumption_opt.reset_index().rename(columns={0: "self_consumption"})
        self_consumption_ref["type"] = "simulation"
        self_consumption_opt["type"] = "optimization"


        country_load_df = pd.concat([self_consumption_opt, self_consumption_ref], axis=0)

        return country_load_df

def get_country_load_profiles(folder_name: Path, perc_cooling: float):
    db = DB(path=folder_name / "output" / f"{folder_name.name}.sqlite")
    scenario_table = db.read_dataframe(table_name="OperationScenario")

    # filter out only buildings with air HP and ground HP
    building_table = db.read_dataframe(table_name="OperationScenario_Component_Building")

    # return 0 for those..
    if building_table.empty:
        return pd.DataFrame.from_dict({"opt_grid_demand_stock_MW": np.zeros(8760), 
                                       "ref_grid_demand_stock_MW": np.zeros(8760),
                                       "Hour": np.arange(1, 8761)},  orient="index").T
    else:

        scenario_df = define_tech_scenario(
            prob_cooling=perc_cooling,
            df_scenarios=scenario_table,
            df_hp=building_table
        )

        opt_loads, ref_loads = load_electricity_demand_profiles(scenario_ids=list(scenario_df["ID_Scenario"]), folder=folder_name / "output")

        # merge the numbers of each building scenario with the opt and ref loads:
        building_numbers = scenario_df.loc[:, ["ID_Scenario", "number_of_buildings", "ID_EnergyPrice"]].copy().set_index("ID_Scenario").sort_index()
        opt_stock = opt_loads.sort_values(by=["ID_Scenario", "Hour"]).set_index("ID_Scenario").join(building_numbers)
        opt_stock["opt_grid_demand_stock_MW"] = opt_stock["Grid"] * opt_stock["number_of_buildings"] / 1_000_000  # MW
        opt_stock["opt_PV2Grid_MW"] = opt_stock["PV2Grid"] * opt_stock["number_of_buildings"] / 1_000_000 # MW
        opt_stock["opt_PhotovoltaicProfile_MW"] = opt_stock["PhotovoltaicProfile"] * opt_stock["number_of_buildings"] / 1_000_000 # MW
        opt_stock["opt_PV2Load_MW"] = opt_stock["PV2Load"] * opt_stock["number_of_buildings"] / 1_000_000 # MW


        ref_stock = ref_loads.sort_values(by=["ID_Scenario", "Hour"]).set_index("ID_Scenario").join(building_numbers)
        ref_stock["ref_grid_demand_stock_MW"] = ref_stock["Grid"] * ref_stock["number_of_buildings"] / 1_000_000  # MW
        ref_stock["ref_PV2Grid_MW"] = ref_stock["PV2Grid"] * ref_stock["number_of_buildings"] / 1_000_000 # MW
        ref_stock["ref_PhotovoltaicProfile_MW"] = ref_stock["PhotovoltaicProfile"] * ref_stock["number_of_buildings"] / 1_000_000 # MW
        ref_stock["ref_PV2Load_MW"] = ref_stock["PV2Load"] * ref_stock["number_of_buildings"] / 1_000_000 # MW
        
        country_load_df = pd.concat([opt_stock, ref_stock[["ref_grid_demand_stock_MW", "ref_PV2Grid_MW", "ref_PhotovoltaicProfile_MW", "ref_PV2Load_MW"]]], axis=1)

        return country_load_df


def get_price_profile(folder_name: Path):
    db = DB(path=folder_name / "output" / f"{folder_name.name}.sqlite")
    price_table = db.read_dataframe(table_name="OperationScenario_EnergyPrice", column_names=["electricity_1", "electricity_2", "electricity_3", "electricity_4", "electricity_5"])
    return price_table

def create_national_temperature_df():
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/")
    folder_names = [f"{country}_{year}_{PROJECT_ACRONYM}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2030, 2050]]
    dfs = []
    for folder_name in folder_names:
        if folder_name in ["CYP_2020", "MLT_2020"]:
            continue
        folder = path_2_model_results / folder_name
        country = folder.name.split("_")[0]
        year = folder.name.split("_")[1]
        df = get_country_temperature_profile(
            folder_name=folder,
        )
        df["country"] = country
        df["year"] = year
        dfs.append(df)
        
    big_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    big_df.columns = [str(col) for col in big_df.columns]
    return big_df

def create_number_of_HPs_df(percentage_cooling: float) -> pd.DataFrame:
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/")
    folder_names = [f"{country}_{year}_{PROJECT_ACRONYM}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2030, 2050]]
    dfs = []
    for folder_name in folder_names:
        if folder_name in ["CYP_2020", "MLT_2020"]:
            continue
        folder = path_2_model_results / folder_name
        country = folder.name.split("_")[0]
        year = folder.name.split("_")[1]
        df = get_total_number_of_buildings_with_HP(
            folder_name=folder,
            perc_cooling=percentage_cooling
        )
        df["country"] = country
        df["year"] = year
        dfs.append(df)
        
    big_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    big_df.columns = [str(col) for col in big_df.columns]
    big_df["ID_EnergyPrice"] = big_df["ID_EnergyPrice"].map({i: f"Price {i}" for i in range(1, len(big_df["ID_EnergyPrice"].unique())+1)})
    big_df["year"] = big_df["year"].astype(int)
    big_df.rename(columns={"number_of_buildings": "number of HPs in country"}, inplace=True)
    return big_df

def read_ref_2020_national_heat_demand(percentage_cooling: float) -> pd.DataFrame:
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/")
    
    folder_names = [f"{country}_{year}_{PROJECT_ACRONYM}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2020]]
    dfs = []
    for folder_name in folder_names:
        if folder_name in ["CYP_2020", "MLT_2020"]:
            continue
        folder = path_2_model_results / folder_name
        country = folder.name.split("_")[0]
        year = folder.name.split("_")[1]
        df = get_country_specific_heating_demand(
            folder_name=folder,
            perc_cooling=percentage_cooling,
            include_opt=False
        )
        df["country"] = country
        df["year"] = year
        dfs.append(df)

    big_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    big_df.columns = [str(col) for col in big_df.columns]
    return big_df

def create_national_heat_demand_df(percentage_cooling: float) -> pd.DataFrame:
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/")
    
    folder_names = [f"{country}_{year}_{PROJECT_ACRONYM}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2030, 2050]]
    dfs = []
    for folder_name in folder_names:
        if folder_name in ["CYP_2020", "MLT_2020"]:
            continue
        folder = path_2_model_results / folder_name
        country = folder.name.split("_")[0]
        year = folder.name.split("_")[1]
        print(f"analysing {country} {year}")

        df = get_country_specific_heating_demand(
            folder_name=folder,
            perc_cooling=percentage_cooling,
            include_opt=True
        )
        df["country"] = country
        df["year"] = year
        dfs.append(df)

    big_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    big_df.columns = [str(col) for col in big_df.columns]
    return big_df


def load_only_PV_building_profiles(percentage_cooling: float, parquet_file: Path, project_acronym):
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/")
    folder_names = [f"{country}_{year}_{project_acronym}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2030, 2050]]
    dfs = []
    for folder_name in folder_names:
        if folder_name in [f"CYP_2020_{PROJECT_ACRONYM}", f"MLT_2020_{PROJECT_ACRONYM}"]:
            continue
        folder = path_2_model_results / folder_name
        country = folder.name.split("_")[0]
        year = folder.name.split("_")[1]
        print(f"getting PV {country} {year}")

        country_loads = get_country_load_profiles_for_PV_buidlings(
            folder_name=folder,
            perc_cooling=percentage_cooling,
        )
       

        country_loads["country"] = country
        country_loads["year"] = year

        dfs.append(country_loads)
        
    big_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    big_df.columns = [str(col) for col in big_df.columns]
    big_df.to_parquet(parquet_file)

def create_national_demand_profiles_parquet(percentage_cooling: float, parquet_file: Path):
    path_2_model_results = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/")
    
    folder_names = [f"{country}_{year}_{PROJECT_ACRONYM}" for country in list(EUROPEAN_COUNTRIES.keys()) for year in [2030, 2050]]
    dfs = []
    for folder_name in folder_names:
        if folder_name in [f"CYP_2020_{PROJECT_ACRONYM}", f"MLT_2020_{PROJECT_ACRONYM}"]:
            continue
        folder = path_2_model_results / folder_name
        country = folder.name.split("_")[0]
        year = folder.name.split("_")[1]
        print(f"analysing {country} {year}")

        country_loads = get_country_load_profiles(
            folder_name=folder,
            perc_cooling=percentage_cooling,
        )
        # add the country loads from different building types together:
        demand_MW = country_loads.groupby(["ID_EnergyPrice", "Hour"]).sum()[["ref_grid_demand_stock_MW", 
                                                                             "opt_grid_demand_stock_MW",  
                                                                             "opt_PV2Grid_MW", 
                                                                             "ref_PV2Grid_MW", 
                                                                             "opt_PhotovoltaicProfile_MW", 
                                                                             "ref_PhotovoltaicProfile_MW", 
                                                                             "opt_PV2Load_MW",
                                                                             "ref_PV2Load_MW",
                                                                             ]].reset_index()

        price = get_price_profile(folder).rename(columns={"electricity_1": 1, "electricity_2": 2}).melt(var_name="ID_EnergyPrice", value_name="price (cent/kWh)") 
        price["price (cent/kWh)"] = price["price (cent/kWh)"] * 1000

        df = pd.concat([demand_MW, price.drop(columns="ID_EnergyPrice")], axis=1)
        df["country"] = country
        df["year"] = year

        dfs.append(df)
        
    big_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    big_df.columns = [str(col) for col in big_df.columns]
    big_df.to_parquet(parquet_file)



def main(percentage_cooling: float,):
    print(f"creating parquet file for {percentage_cooling} cooling percentage")
    parquet_file = Path(__file__).parent / f"EU27_loads_{PROJECT_ACRONYM}_cooling-{percentage_cooling}.parquet.gzip"

    create_national_demand_profiles_parquet(percentage_cooling,  parquet_file)




PROJECT_ACRONYM = "grid_fees"
if __name__ == "__main__":
    main(
        percentage_cooling=0.1,
    )

    # 10 cent/kWh levelized cost for offshore wind, onshore and PV are below that [https://www.ise.fraunhofer.de/en/publications/studies/cost-of-electricity.html]
    # 10 cent/kWh = 0.001 cent/Wh , I use this as my thresholds price 
