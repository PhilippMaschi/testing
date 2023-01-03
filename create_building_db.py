# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:07:33 2022

@author: mascherbauer
"""
from pathlib import Path
from functools import wraps
import time
import pandas as pd
import numpy as np
import h5py
import shutil
import re
import multiprocessing


def performance_counter(func):
    @wraps(func)
    def wrapper(*args):
        t_start = time.perf_counter()
        result = func(*args)
        t_end = time.perf_counter()
        exe_time = round(t_end - t_start, 3)
        print(f"Timer: {func.__name__} - {exe_time}s.")
        return result

    return wrapper


def create_dict_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir()


def create_paths(paths: dict, year: int, country: str) -> (Path, Path, Path):
    sub_folder = Path(f"_scen_{country.lower()}{paths['SUB_SCENARIO']}") / "ADD_RESULTS"
    # building classes:
    path_to_building_class = paths["SOURCE_PATH"] / paths[
        "INVERT_SCNEARIO"] / country / sub_folder / f"001_Building_Classes_{year}.csv"
    # building segment:
    building_segment_path = paths["SOURCE_PATH"] / country / sub_folder / f"001_Building_segment_SH_{year}.zip"
    # data for 5R1C model:
    path_to_dynamic_data = paths["SOURCE_PATH"] / paths[
        "INVERT_SCNEARIO"] / country / sub_folder / "Dynamic_Calc_Input_Data" / f"001__dynamic_calc_data_bc_{year}.csv"
    return path_to_building_class, building_segment_path, path_to_dynamic_data


def filter_dataframes(df_building_class: pd.DataFrame, df_building_segment: pd.DataFrame,
                      df_dynamic_data: pd.DataFrame):
    # filter building class columns:
    building_classes = df_building_class.loc[:, [
                                                    "index",
                                                    "name",
                                                    "construction_period_start",
                                                    "construction_period_end"]
                       ]

    # filter dynamic data:
    dynamic_data = df_dynamic_data.loc[:, [
                                              "bc_index",
                                              "building_categories_index",
                                              "Af",
                                              "Hop",
                                              "Htr_w",
                                              "Hve",
                                              "CM_factor",
                                              "Am_factor",
                                              "average_effective_area_wind_west_east_red_cool",
                                              "average_effective_area_wind_south_red_cool",
                                              "average_effective_area_wind_north_red_cool",
                                              "spec_int_gains_cool_watt",
                                              "hwb_norm"]
                   ]

    # filter building segment columns:
    building_segment = df_building_segment.loc[:, [
                                                      "index",
                                                      "name",
                                                      "building_classes_index",
                                                      "number_of_buildings",
                                                      "energy_carrier_region_index",
                                                      "heat_supply_system_index"]
                       ]
    return building_classes, building_segment, dynamic_data


def fetch_data(paths: dict, year: int, country: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    path_to_building_class, building_segment_path, path_to_dynamic_data = create_paths(paths=paths,
                                                                                       year=year,
                                                                                       country=country)
    # read dataframes:
    building_classes_orig = pd.read_csv(path_to_building_class)
    dynamic_data_orig = pd.read_csv(path_to_dynamic_data)
    building_segment_orig = pd.read_csv(building_segment_path.open(f"001_Building_segment_SH_{year}.csv"))
    building_classes, building_segment, dynamic_data = filter_dataframes(df_building_class=building_classes_orig,
                                                                         df_building_segment=building_segment_orig,
                                                                         df_dynamic_data=dynamic_data_orig)
    return building_classes, building_segment, dynamic_data


def to_series(col):
    if isinstance(col, (pd.DataFrame, pd.Series)):
        return col.squeeze()
    elif isinstance(col, (list, np.ndarray)):
        return pd.Series(col)
    else:
        return col

def remove_multi_index(df: pd.DataFrame) -> pd.DataFrame:
    if type(df.columns) == pd.core.indexes.multi.MultiIndex:
        df.columns = df.columns.map(''.join)  # remove the MultiIndex (which has only one layer)
    return df

def calc_mean(data: dict) -> float:
    # Multiply each key with its corresponding value and add them
    sum_products = sum(key * value for key, value in data.items())
    # Calculate the sum of the values
    sum_values = sum(value for value in data.values())
    # Return the mean value
    return sum_products / sum_values


def fix_dfs(bc_df: pd.DataFrame, bssh_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # remove the multiindex:
    bssh_df.columns = bssh_df.columns.map(''.join)
    bc_df.columns = bc_df.columns.map(''.join)
    # change columns to series:
    bssh = bssh_df.apply(to_series)
    bc = bc_df.apply(to_series)

    # filter out residential buildings:
    # filter out all buildings that are not SFH or MFH
    mask = bc["building_categories_index"].isin([1, 2, 5, 6])  # 1,2 SFH and 5, 6 are MFH
    bc_residential = bc.loc[mask.squeeze(), :]

    # only keep residential buildings in bssh frame:
    bc_indizes = list(bc_residential.loc[:, "index"])
    bssh_residential = bssh.loc[bssh["building_classes_index"].isin(bc_indizes)].reset_index(
        drop=True)  # 1,2 SFH and 5, 6 are MFH
    return bc_residential, bssh_residential


def get_number_of_heat_pumps(bc_df: pd.DataFrame, bssh_df: pd.DataFrame) -> pd.DataFrame:
    # filter out only buildings with heat pumps:
    bssh_heat_pumps = bssh_df.loc[bssh_df['heat_supply_system_index'].isin([42, 43])].reset_index(drop=True)

    # Group the rows of bssh_df by building_classes_index
    grouped = bssh_heat_pumps.groupby("building_classes_index")

    for index, group in grouped:
        # add number of heat pumps to bc df:
        total_number_of_air_hp = group.loc[group.loc[:, "heat_supply_system_index"] == 42, "number_of_buildings"].sum()
        total_number_of_ground_hp = group.loc[group.loc[:, "heat_supply_system_index"] == 43, "number_of_buildings"].sum()
        bc_df.loc[bc_df.loc[:, "index"] == index, "number_of_buildings_with_HP_air"] = total_number_of_air_hp
        bc_df.loc[bc_df.loc[:, "index"] == index, "number_of_buildings_with_HP_ground"] = total_number_of_ground_hp

        # add supply temperature to bc df:
        # check if there are multiple supply temperatures:
        supply_temperatures = list(group.loc[:, "supply_temperature"].unique())
        if len(supply_temperatures) > 1:
            # group by supply temperature
            supply_temperature_group = group.groupby("supply_temperature")
            nums = {}
            for temp in supply_temperatures:
                number_buildings_sup_temp = supply_temperature_group.get_group(temp)["number_of_buildings"].sum()
                nums[temp] = number_buildings_sup_temp
            # calculate the mean:
            mean_sup_temp = calc_mean(nums)
        else:
            mean_sup_temp = supply_temperatures[0]

        # add the supply temperature to the BC dataframe:
        bc_df.loc[bc_df.loc[:, "index"] == index, "supply_temperature"] = mean_sup_temp

    # turn nan into zeros in the residential_df (those buildings don't have hps therefore they were not counted:
    bc_df = bc_df.fillna(0)

    return bc_df


def remove_buildings_without_heatpumps(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["number_of_buildings_with_HP_air"] != 0) | (df["number_of_buildings_with_HP_ground"] != 0)]
    return df

def find_hdf5_file(directory: Path):
    for file in directory.rglob('*.hdf5'):
        return file
    return None


def find_distribution_csv_file(directory: Path):
    for file in directory.rglob('*.csv'):
        if "distribution_sh" in file.name:
            return file
    return None

def copy_file_to_disc(source: Path, destination: Path):
    """Copy a file from src to dst, creating any missing directories in the destination path."""
    # Check if the destination file already exists
    if destination.exists():
        print(f"{destination} already exists. Skipping copy operation.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination)


def hdf5_to_pandas(hdf5_file: Path, group_name, columns) -> pd.DataFrame:
    with h5py.File(hdf5_file, 'r') as file:
        # Get the table from the group
        dataset = file[group_name]
        df = pd.DataFrame(index=range(len(dataset)), columns=[list(columns.keys())])
        for name in columns.keys():
            df[name] = dataset[name]

    return df


def read_hdf5(country: str, output_path: Path, years: list,
              building_class_columns: dict,
              building_segment_columns: dict,
              heating_system_columns: dict,
              energy_carrier_index: dict):
        heating_key = {
        1: "no heating",
        2: "no heating",
        3: "district heating",
        4: "district heating",
        5: "district heating",
        6: "district heating",
        7: "district heating",
        8: "district heating",
        9: "oil",
        10: "oil",
        11: "oil",
        12: "oil",
        13: "oil",
        14: "oil",
        15: "oil",
        16: "oil",
        17: "oil",
        18: "coal",
        19: "coal",
        20: "coal",
        21: "gas",
        22: "gas",
        23: "gas",
        24: "gas",
        25: "gas",
        26: "gas",
        27: "gas",
        28: "gas",
        29: "wood",
        30: "wood",
        31: "wood",
        32: "wood",
        33: "wood",
        34: "wood",
        35: "wood",
        36: "wood",
        37: "electricity",
        38: "electricity",
        39: "electricity",
        40: "split system",
        41: "split system",
        42: "heat pump air",
        43: "heat pump ground",
        44: "electricity"
    }

        # create country specific path
        main_directory = Path(__file__).parent / "building_data" / country
        hdf5_f = find_hdf5_file(main_directory)

        for year in years:
            BC = hdf5_to_pandas(hdf5_f, f"BC_{year}", building_class_columns)
            BSSH = hdf5_to_pandas(hdf5_f, f"BSSH_{year}", building_segment_columns)
            heating_system = hdf5_to_pandas(hdf5_f, "HeatingSystems", heating_system_columns)
            energy_carrier = hdf5_to_pandas(hdf5_f, "EnergyCarrierDefinition_2020", energy_carrier_index)
            dist_df = pd.read_csv(main_directory / "distribution_sh.csv", sep=",", encoding="latin1")
            distribution_sh = dist_df.iloc[3:, :].dropna(axis=1).rename(columns={"csvid": "distribution_sh_index"}).reset_index(drop=True)
            # change the index to integer otherwise "map" does not work
            distribution_sh["distribution_sh_index"] = distribution_sh["distribution_sh_index"].astype(int)

            # map the supply temperature and the heating system type to the BSSH df:
            BSSH.loc[:, "supply_temperature"] = BSSH["distribution_sh_index"].squeeze().map(
                distribution_sh.set_index("distribution_sh_index")["supply_temperature"])
            # BSSH_2020.loc[:, "heating_distribution_system"] = BSSH_2020["distribution_sh_index"].squeeze().map(distribution_sh.set_index("distribution_sh_index")["name"])
            # uncomment the next line to include the return temperature
            # BSSH_2020.loc[:, "return_temperature"] = BSSH_2020["distribution_sh_index"].squeeze().map(distribution_sh.set_index("distribution_sh_index")["return_temperature"])

            # add the heating system type to the BSSH table
            # BSSH_2020.loc[:, "energy_carrier_name"] = BSSH_2020["energy_carrier"].squeeze().map(heating_key)


            # fix the dfs and reduce their size by only including residential buildings:
            bc_residential, bssh_residential = fix_dfs(BC, BSSH)

            # add the number of heat pumps by merging the BSSH to the BC df
            final_df = get_number_of_heat_pumps(bc_df=bc_residential, bssh_df=bssh_residential)

            # remove buildings without any heat pumps
            df = final_df[(final_df["number_of_buildings_with_HP_air"] != 0) |
                          (final_df["number_of_buildings_with_HP_ground"] != 0)].reset_index(drop=True)

            df.to_parquet(output_path / country / f'{year}.parquet.gzip', compression='gzip', index=False)

            print(f"added {year} data to {country}.")


def copy_hdf5_files(out_path: Path, countries: list):
    for country in countries:
        source_directory = paths["SOURCE_PATH"] / paths[
            "INVERT_SCENARIO"] / country / f"_scen_{country.lower()}{paths['SUB_SCENARIO']}"
        source_path = find_hdf5_file(source_directory)
        destination_path = out_path / country / source_path.name
        copy_file_to_disc(source_path, destination_path)
    print("all hdf5 files are copied to local disk")


def copy_distribution_csvs(out_path: Path, countries: list):
    invert_input_folder = "input_2022_newbuild_newlifetime"
    for country in countries:
        source_directory = paths["SOURCE_PATH"] / invert_input_folder / country
        source_path = find_distribution_csv_file(source_directory)
        destination_path = out_path / country / "distribution_sh.csv"
        copy_file_to_disc(source_path, destination_path)
        print(f"copied {country} distribution_sh.csv to local disk")
    print("all csv files are copied to local disk")


def delete_hdf5(folder=r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\building_data"):
    """Iterate through the given folder and delete all HDF5 files, prompting the user for confirmation at the beginning.

    Parameters
    ----------
    folder : str
        The path to the folder to be searched.
    """
    # Prompt the user for confirmation
    response = input("Do you want to delete all HDF5 files in the given folder? [Y/n] ")
    if response.lower() != 'y':
        print("No files were deleted.")
        return

    # Iterate through the files in the folder
    for file in Path(folder).glob('*.h5'):
        # Delete the file
        file.unlink()
        print(f"{file} deleted.")


@performance_counter
def main(years: list, out_path: Path,
         building_class_columns,
         building_segment_columns,
         heating_system_columns,
         energy_carrier_index):
    heating_key = {
        1: "no heating",
        2: "no heating",
        3: "district heating",
        4: "district heating",
        5: "district heating",
        6: "district heating",
        7: "district heating",
        8: "district heating",
        9: "oil",
        10: "oil",
        11: "oil",
        12: "oil",
        13: "oil",
        14: "oil",
        15: "oil",
        16: "oil",
        17: "oil",
        18: "coal",
        19: "coal",
        20: "coal",
        21: "gas",
        22: "gas",
        23: "gas",
        24: "gas",
        25: "gas",
        26: "gas",
        27: "gas",
        28: "gas",
        29: "wood",
        30: "wood",
        31: "wood",
        32: "wood",
        33: "wood",
        34: "wood",
        35: "wood",
        36: "wood",
        37: "electricity",
        38: "electricity",
        39: "electricity",
        40: "split system",
        41: "split system",
        42: "heat pump air",
        43: "heat pump ground",
        44: "electricity"
    }

    country_list = ['AUT',
                    'BEL',
                    'BGR',
                    'HRV',
                    'CYP',
                    'CZE',
                    'DNK',
                    'EST',
                    'FIN',
                    'FRA',
                    'DEU',
                    'GRC',
                    'HUN',
                    'IRL',
                    'ITA',
                    'LVA',
                    'LTU',
                    'LUX',
                    'MLT',
                    'NLD',
                    'POL',
                    'PRT',
                    'ROU',
                    'SVK',
                    'SVN',
                    'ESP',
                    'SWE']
    # copy the hdf files
    # copy_hdf5_files(out_path=out_path, countries=country_list)

    # copy distribution csv files:
    # copy_distribution_csvs(out_path=out_path, countries=country_list)

    # use multiprocessing:
    arglist = [(country, out_path, years,
                building_class_columns,
                building_segment_columns,
                heating_system_columns,
                energy_carrier_index) for country in country_list]
    with multiprocessing.Pool(6) as pool:
        pool.starmap(read_hdf5, arglist)



if __name__ == "__main__":
    paths = {"SOURCE_PATH": Path(r"W:\projects3\2021_RES_HC_Pathways\invert"),
             "INVERT_SCENARIO": "output_new_trends_2022_12_20_2050",
             "SUB_SCENARIO": "_res_hc_pw_alternative_1_ab"}  # always has _sce_country before
    output_folder = Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\building_data")

    BUILDING_CLASS_COLUMNS = {
        "index": int,
        "name": str,
        "construction_period_start": int,
        "construction_period_end": int,
        "building_categories_index": int,
        "number_of_dwellings_per_building": "float32",
        "number_of_persons_per_dwelling": "float32",
        "length_of_building": "float32",
        "width_of_building": "float32",
        "number_of_floors": "float32",
        "room_height": "float32",
        "percentage_of_building_surface_attached_length": "float32",
        "percentage_of_building_surface_attached_width": "float32",
        "share_of_window_area_on_gross_surface_area": "float32",
        "share_of_windows_oriented_to_south": "float32",
        "share_of_windows_oriented_to_north": "float32",
        "grossfloor_area": "float32",
        "heated_area": "float32",
        "areafloor": "float32",
        "areawindows": "float32",
        "area_suitable_solar": "float32",
        "grossvolume": "float32",
        "heatedvolume": "float32",
        "heated_norm_volume": "float32",
        "hwb": "float32",
        "hwb_norm": "float32",
        "u_value_ceiling": "float32",
        "u_value_exterior_walls": "float32",
        "u_value_windows1": "float32",
        "u_value_windows2": "float32",
        "u_value_roof": "float32",
        "u_value_floor": "float32",
        "seam_loss_windows": "float32",
        "trans_loss_walls": "float32",
        "trans_loss_ceil": "float32",
        "trans_loss_wind": "float32",
        "trans_loss_floor": "float32",
        "trans_loss_therm_bridge": "float32",
        "trans_loss_ventilation": "float32",
        "total_heat_losses": "float32",
        "average_effective_area_wind_west_east_red_cool": "float32",
        "average_effective_area_wind_south_red_cool": "float32",
        "average_effective_area_wind_north_red_cool": "float32",
        "spec_int_gains_cool_watt": "float32",
        "attached_surface_area": "float32"
    }

    BUILDING_SEGMENT_COLUMNS = {
        "index": int,
        "name": str,
        "building_classes_index": int,
        "number_of_buildings": "float32",
        "heat_supply_system_index": int,
        "installation_year_system_start": "float32",
        "installation_year_system_end": "float32",
        "distribution_sh_index": int,
        "distribution_dhw_index": int,
        "pv_system_index": int,
        "energy_carrier": int,
        "annual_energy_costs_hs": "float32",
        "total_annual_cost_hs": "float32",
        "annual_energy_costs_dhw": "float32",
        "total_annual_cost_dhw": "float32",
        "hs_efficiency": "float32",
        "dhw_efficiency": "float32",
        "size_pv_system": "float32"
    }

    HEATING_SYSTEM_COLUMNS = {
        "index": int,
        "name": str,
        "energy_carrier_main_index": int,

        "build_central_system": int,

    }

    ENERGY_CARRIER_INDEX = {
        "index": int,
        "name": str
    }
    create_dict_if_not_exists(output_folder)

    years = [2020, 2030, 2050]
    main(years, output_folder,
         BUILDING_CLASS_COLUMNS,
         BUILDING_SEGMENT_COLUMNS,
         HEATING_SYSTEM_COLUMNS,
         ENERGY_CARRIER_INDEX)
