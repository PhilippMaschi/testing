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
    "attached_surface_area": "float32",
    "effective_indoor_temp_jan": "float32",
    "uedh_sh_effective": "float32",
    "ued_dhw": "float32",
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
    # "annual_energy_costs_hs": "float32",
    # "total_annual_cost_hs": "float32",
    # "annual_energy_costs_dhw": "float32",
    # "total_annual_cost_dhw": "float32",
    "hs_efficiency": "float32",
    "dhw_efficiency": "float32",
    "size_pv_system": "float32",
    "fed_ambient_sh_per_bssh": "float32",
    "fed_ambient_dhw_per_bssh": "float32",
}

HEATING_SYSTEM_INDEX = {
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

# SELECTED_HEATING_SYSTEMS = {"electricity": [37, 38, 39, 44],
#                             "heat pump air": [42],
#                             "heat pump ground": [43]}

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

SFH_MFH = {
    1: "SFH",
    2: "SFH",
    5: "MFH",
    6: "MFH"
}


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


def get_dynamic_calc_data(path: Path, year: int, country: str) -> pd.DataFrame:
    path_to_dynamic_data = path / f"dynamic_calc_{year}.npz"
    npz = np.load(path_to_dynamic_data)
    column_names = npz["arr_1"]
    data = npz["arr_0"]
    df = pd.DataFrame(data=data)
    df.columns = column_names.squeeze()
    return df


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
    bc["building_categories_index"] = bc["building_categories_index"].astype(int)
    mask = bc["building_categories_index"].isin(SFH_MFH.keys())  # 1,2 SFH and 5, 6 are MFH
    bc_residential = bc.loc[mask.squeeze(), :].reset_index(drop=True)

    # only keep residential buildings in bssh frame:
    bc_indizes = list(bc_residential.loc[:, "index"])  # 1,2 SFH and 5, 6 are MFH
    bssh_residential = bssh.loc[bssh["building_classes_index"].isin(bc_indizes)].reset_index(drop=True)
    return bc_residential, bssh_residential


def get_number_energy_carriers(group: pd.DataFrame) -> dict:
    numbers = group.groupby("energy_carrier_name")["number_of_buildings"].sum()
    return numbers.to_dict()


def calculate_mean_supply_temperature(grouped_df: pd.DataFrame, helper_name: str = None) -> float:
    # add supply temperature to bc df:
    # check if there are multiple supply temperatures:
    supply_temperatures = list(grouped_df.loc[:, "supply_temperature"].unique())
    if len(supply_temperatures) > 1:
        # group by supply temperature
        supply_temperature_group = grouped_df.groupby("supply_temperature")
        nums = {}
        for temp in supply_temperatures:
            if helper_name == "get_number_of_buildings":
                number_buildings_sup_temp = supply_temperature_group.get_group(temp)["number_of_buildings"].sum()
            else:
                number_buildings_sup_temp = supply_temperature_group.get_group(temp)[
                                                "number_buildings_heat_pump_ground"].sum() + \
                                            supply_temperature_group.get_group(temp)[
                                                "number_buildings_heat_pump_air"].sum()

            nums[temp] = number_buildings_sup_temp
        # calculate the mean:
        mean_sup_temp = calc_mean(nums)
    else:
        mean_sup_temp = supply_temperatures[0]

    return mean_sup_temp


def get_number_of_buildings(bc_df: pd.DataFrame, bssh_df: pd.DataFrame) -> pd.DataFrame:
    # Group the rows of bssh_df by building_classes_index
    grouped = bssh_df.groupby("building_classes_index")

    for index, group in grouped:
        # add total number of buildings:
        total_number_buildings = group["number_of_buildings"].sum()
        bc_df.loc[bc_df.loc[:, "index"] == index, "number_of_buildings"] = total_number_buildings
        # add the number of buildings with other energy carriers
        numbers = get_number_energy_carriers(group=group)
        for carrier, number in numbers.items():
            bc_df.loc[bc_df.loc[:, "index"] == index, f"number_buildings_{carrier.replace(' ', '_')}"] = number

    # the supply temperature is only calculated/valid for the buildings with heat pumps:
    bssh_heat_pumps = bssh_df.loc[bssh_df['heat_supply_system_index'].isin([42, 43])].reset_index(drop=True)
    heat_pumps_group = bssh_heat_pumps.groupby("building_classes_index")  # for supply temperatures
    for index, group in heat_pumps_group:
        mean_sup_temp = calculate_mean_supply_temperature(group, "get_number_of_buildings")
        # add the supply temperature to the BC dataframe:
        bc_df.loc[bc_df.loc[:, "index"] == index, "supply_temperature"] = mean_sup_temp

        # add PV size to df:
        pv_sizes = list(group.loc[:, "size_pv_system"].unique())
        if len(pv_sizes) > 1:
            # group by pv size
            pv_size_group = group.groupby("size_pv_system")
            for size in pv_sizes:
                number_buildings_pv_size = pv_size_group.get_group(size)["number_of_buildings"].sum()
                # add the pv size to the BC dataframe:
                rounded_size = round(size)
                bc_df.loc[bc_df.loc[:, "index"] == index, f"PV_number_of_{rounded_size}_m2"] = number_buildings_pv_size

        else:
            # add the pv size to the BC dataframe:
            rounded_size = round(pv_sizes[0])
            bc_df.loc[bc_df.loc[:, "index"] == index, f"PV_number_of_{rounded_size}_m2"] = group[
                "number_of_buildings"].sum()

    # turn nan into zeros in the residential_df (those buildings don't have hps therefore they were not counted:
    bc_df = bc_df.fillna(0)

    return bc_df


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
    print(f"copied {source.name} to local disk as {destination}")


def hdf5_to_pandas(hdf5_file: Path, group_name: str, columns: dict = None) -> pd.DataFrame:
    """
    :param hdf5_file: path to the hdf5 file which needs to be opened

    :param group_name: is either BC_{year} or BSSH_{year} where year can be 2020 or any other year that has been
        modelled with Invert. Eg: BC_2030 or BSSH_2050. Other options are:
            "HeatingSystems", "EnergyCarrierDefinition_{year}".
            For more options ask Andi.

    :param columns: dictionary with columns that you want to extract from this particular dataframe. The keys should
    contain the name of the column and the values represnt the datatype of the columns. For example:
        BSSH_colums = {
        "index": int,
        "name": str,
        "building_classes_index": int,
        "number_of_buildings": "float32",
        }
    If no columns dict is provided, all the columns are imported.

    :return: pd.DataFrame containing the hdf5 data
    """
    with h5py.File(hdf5_file, 'r') as file:
        # Get the table from the group
        dataset = file[group_name]

        if columns is None:
            columns = dataset.dtype.fields

        data_array = dataset[:]
        df = pd.DataFrame(data_array, columns=columns)

    return df


def merge_dynamic_df_to_df(dynamic_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    indizes_list = list(final_df.loc[:, "index"])
    mask = dynamic_df["bc_index"].isin(indizes_list)  # 1,2 SFH and 5, 6 are MFH
    dynamic_residential = dynamic_df.loc[mask.squeeze(), :].reset_index(drop=True)
    transfer_columns = ["Af", "Hop", "Htr_w", "Hve", "CM_factor", "Am_factor"]
    final_df.loc[:, transfer_columns] = dynamic_residential.loc[:, transfer_columns]
    return final_df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    columns_2_drop = ["length_of_building",
                      "width_of_building",
                      "percentage_of_building_surface_attached_length",
                      "percentage_of_building_surface_attached_width",
                      "share_of_window_area_on_gross_surface_area",
                      "share_of_windows_oriented_to_south",
                      "share_of_windows_oriented_to_north",
                      "grossvolume",
                      "heatedvolume",
                      "heated_norm_volume",
                      "u_value_ceiling",
                      "u_value_exterior_walls",
                      "u_value_windows1",
                      "u_value_windows2",
                      "u_value_roof",
                      "u_value_floor",
                      "seam_loss_windows",
                      "trans_loss_walls",
                      "trans_loss_ceil",
                      "trans_loss_wind",
                      "trans_loss_floor",
                      "trans_loss_therm_bridge",
                      "trans_loss_ventilation",
                      "total_heat_losses",
                      "attached_surface_area"
                      ]
    df_final = df.drop(columns=columns_2_drop)
    # re-arrange the columns
    columns = df_final.columns.to_list()
    cols = columns[:10] + columns[-6:] + columns[10:-6]
    df_final = df_final[cols]
    # df for croatia in 2020 does not have ground hps.. check if ground HPs are in table:
    if "number_buildings_heat_pump_ground" not in df_final.columns:
        df_final["number_buildings_heat_pump_ground"] = 0
    return df_final.fillna(0)  # in case there are any na values


def merge_similar_buildings(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()

    # columns where numbers are summed up (PV and number of buildings)
    adding_name = [name for name in df.columns if "number" in name]
    # columns to merge: [2:] so index and name
    merging_names = [name for name in df.columns if "PV" and "number" not in name][2:] + adding_name[:3]
    # except number of persons and number of dwellings ([3:]) left out
    adding_name = adding_name[3:]

    # round the hwb column to total numbers (int) so they can be merged
    new_df.loc[:, "hbw_int"] = [int(value) for value in new_df.loc[:, "hwb"]]
    # group them by building category and contruction period:
    groups = new_df.groupby(["construction_period_start", "building_categories_index", "hbw_int"])

    for index, group in groups:
        # take the mean of all other values (most of them are the same anyways
        new_row = pd.DataFrame(columns=new_df.columns, index=[0])

        new_row.loc[0, merging_names] = group.loc[:, merging_names].mean()
        new_row.loc[0, adding_name] = group.loc[:, adding_name].sum()
        new_row.loc[0, "name"] = group.loc[:, "name"].iloc[0][0:5]  # new name is first 5 letters of first name
        new_row.loc[0, "index"] = group.loc[:, "index"].iloc[0]  # new index is the first index of merged rows
        new_row.loc[0, "supply_temperature"] = calculate_mean_supply_temperature(group)

        # drop the rows that are merged
        new_df = new_df.drop(index=group.index, axis=0)
        # add the merged row
        new_df = pd.concat([new_df, new_row])

    return new_df.reset_index(drop=True)


def fix_number_of_persons(df: pd.DataFrame) -> pd.DataFrame:
    # divide the floor area Af by 42.5
    df.loc[:, "number_of_persons_per_dwelling"] = df.loc[:, "Af"] / 42.5
    df.loc[:, "number_of_persons_per_dwelling"] = df.loc[:, "number_of_persons_per_dwelling"].apply(np.round)
    return df


def read_hdf5(country: str, output_path: Path, years: list, path_dict: dict, hdf5_was_copied: bool = False):
    print(f"reading hdf5 for {country}")
    # create country specific path
    main_directory = output_path / country
    if hdf5_was_copied:
        hdf5_f = find_hdf5_file(main_directory)
    else:
        source_directory = path_dict["SOURCE_PATH"] / path_dict[
            "INVERT_SCENARIO"] / country / f"_scen_{country.lower()}_{path_dict['SUB_SCENARIO']}"
        assert source_directory.exists(), "directory path provided does not exist"
        hdf5_f = find_hdf5_file(source_directory)
    for year in years:
        # check if files already exist:
        file = output_path / country / f'{year}.parquet.gzip'
        if file.exists():
            print(f"{country} {year} parquet already exists - skipping")
        else:
            BC = hdf5_to_pandas(hdf5_f, f"BC_{year}", BUILDING_CLASS_COLUMNS)
            BSSH = hdf5_to_pandas(hdf5_f, f"BSSH_{year}", BUILDING_SEGMENT_COLUMNS)
            # only keep residential buildings:
            bc_residential, bssh_residential = fix_dfs(BC, BSSH)

            # heating_system = hdf5_to_pandas(hdf5_f, "HeatingSystems", HEATING_SYSTEM_COLUMNS)
            # energy_carrier = hdf5_to_pandas(hdf5_f, "EnergyCarrierDefinition_2020", ENERGY_CARRIER_INDEX)
            dist_df = pd.read_csv(main_directory / "distribution_sh.csv", sep=",", encoding="latin1")
            distribution_sh = dist_df.iloc[3:, :].dropna(axis=1).rename(
                columns={"csvid": "distribution_sh_index"}).reset_index(drop=True)
            # change the index to integer otherwise "map" does not work
            distribution_sh["distribution_sh_index"] = distribution_sh["distribution_sh_index"].astype(int)

            # map the supply temperature and the heating system type to the BSSH df:
            bssh_residential.loc[:, "supply_temperature"] = bssh_residential["distribution_sh_index"].map(
                distribution_sh.set_index("distribution_sh_index")["supply_temperature"])
            # bssh_residential.loc[:, "heating_distribution_system"] = bssh_residential["distribution_sh_index"].map(distribution_sh.set_index("distribution_sh_index")["name"])
            # uncomment the next line to include the return temperature
            # bssh_residential.loc[:, "return_temperature"] = bssh_residential["distribution_sh_index"].squeeze().map(distribution_sh.set_index("distribution_sh_index")["return_temperature"])

            # add the heating system type to the bssh_residential table
            bssh_residential.loc[:, "energy_carrier_name"] = bssh_residential["heat_supply_system_index"].map(
                HEATING_SYSTEM_INDEX)

            # add the number of buildings by merging the bssh_residential to the BC df
            merged_df = get_number_of_buildings(bc_df=bc_residential, bssh_df=bssh_residential)

            # remove buildings without any heat pumps
            # df = merged_df[(merged_df["number_of_buildings_with_HP_air"] != 0) |
            #               (merged_df["number_of_buildings_with_HP_ground"] != 0)].reset_index(drop=True)

            # add the dynamic calc data to the df:
            dynamic_df = get_dynamic_calc_data(path=main_directory, year=year, country=country)
            # merge the dynamic df to the merged df:
            merged_df_2 = merge_dynamic_df_to_df(dynamic_df=dynamic_df, final_df=merged_df)

            # clean out columns that are not needed:
            cleaned_df = clean_df(merged_df_2)

            # reduce the size of buildings by merging very similar buildings:
            # reduced_df = merge_similar_buildings(cleaned_df).drop(columns=["index"]).reset_index()

            # fix the number of persons because they are wrong in Invert:
            final_df = fix_number_of_persons(cleaned_df)
            # change the type of "supply temperature" so parquet doesnt make trouble
            final_df["supply_temperature"] = final_df["supply_temperature"].astype(float)
            # rename the effective indoor set temp column
            final_df = final_df.rename(columns={"effective_indoor_temp_jan": "indoor_set_temperature"})
            # fix the name column:
            final_df['name'] = final_df['name'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            final_df.to_parquet(output_path / f'INVERT_{country}_{year}.parquet.gzip', compression='gzip', index=False)
            print_summary_data_from_invert(final_df, year, country)
            print_percentages_of_heating_systems(final_df, year, country)
            print(f"added {year} data to {country}.")


def print_summary_data_from_invert(df: pd.DataFrame, year, country):
    print(f"Printing Invert data on buildings from {country} in {year} \n")
    df.loc[:, "type"] = df["building_categories_index"].map(SFH_MFH)
    type_groups = df.groupby("type")
    for building_type, group in type_groups:
        print(
            f"{building_type}: \n "
            f"UED for DHW = "
            f"{round((group['ued_dhw'] * group['number_of_buildings']).sum() / 1_000 / 1_000)} MWh \n"  # MWh
            f"UED for space heating = "
            f"{round((group['uedh_sh_effective'] * group['number_of_buildings']).sum() / 1_000 / 1_000)} MWh \n"  # MWh

        )

def print_percentages_of_heating_systems(df: pd.DataFrame, year, country):
    print(f"\n Percentages of heating systems for {country} {year}: \n")
    total_nr_buildings = df.loc[:, "number_of_buildings"].sum()
    conventional_percentage = (df.loc[:, "number_buildings_coal"].sum() + df.loc[:, "number_buildings_gas"].sum() + df.loc[:, "number_buildings_oil"].sum() + df.loc[:, "number_buildings_district_heating"].sum() + df.loc[:, "number_buildings_wood"].sum()) / total_nr_buildings * 100
    hp_air_percentage = df.loc[:, "number_buildings_heat_pump_air"].sum() / total_nr_buildings * 100
    hp_ground_percentage = df.loc[:, "number_buildings_heat_pump_ground"].sum() / total_nr_buildings * 100
    split_percentage = df.loc[:, "number_buildings_split_system"].sum() / total_nr_buildings * 100
    direct_elec_percentage = df.loc[:, "number_buildings_electricity"].sum() / total_nr_buildings * 100
    print(f"conventional: {round(conventional_percentage)} % \n \
          Air heat pump: {round(hp_air_percentage)} % \n \
          Ground heat pump: {round(hp_ground_percentage)} % \n \
          Split system: {round(split_percentage)} % \n \
          Direct electric: {round(direct_elec_percentage)} % \n ")


def copy_hdf5_files(path_dict: dict, out_path: Path, countries: list):
    for country in countries:
        source_directory = path_dict["SOURCE_PATH"] / path_dict[
            "INVERT_SCENARIO"] / country / f"_scen_{country.lower()}_{path_dict['SUB_SCENARIO']}"
        assert source_directory.exists(), "directory path provided does not exist"
        source_path = find_hdf5_file(source_directory)
        destination_path = out_path / country / source_path.name
        copy_file_to_disc(source_path, destination_path)
    print("all hdf5 files are copied to local disk")


def copy_dynamic_calc_data(path_dict: dict, out_path: Path, countries: list, years: list):
    for country in countries:
        for year in years:
            source_file = path_dict["SOURCE_PATH"] / path_dict[
                "INVERT_SCENARIO"] / country / f"_scen_{country.lower()}_{path_dict['SUB_SCENARIO']}" / \
                          r"ADD_RESULTS/Dynamic_Calc_Input_Data" / f"001__dynamic_calc_data_bc_{year}.npz"
            destination_path = out_path / country / f"dynamic_calc_{year}.npz"
            copy_file_to_disc(source_file, destination_path)
    print("all dynamic npz files are copied to local disk")


def copy_distribution_csvs(path_dict: dict, out_path: Path, countries: list):
    invert_input_folder = path_dict["INVERT_INPUT"]
    for country in countries:
        source_directory = invert_input_folder / country
        source_path = find_distribution_csv_file(source_directory)
        destination_path = out_path / country / "distribution_sh.csv"
        copy_file_to_disc(source_path, destination_path)
    print("all csv files are copied to local disk")


def delete_hdf5_files(folder: Path):
    # Prompt the user for confirmation
    response = input(f"Do you want to delete all HDF5  files in the given folder? [Y/n] \n {folder}")
    if response.lower() != 'y':
        print("No files were deleted.")
        return

    # Iterate through the files in the folder
    for file_ending in ['*.hdf5']:
        for file in Path(folder).rglob(file_ending):
            # Delete the file
            file.unlink()
            print(f"{file} deleted.")


def clean_up(folder: Path):
    """Iterate through the given folder and delete all HDF5 files, prompting the user for confirmation at the beginning.

    Parameters
    ----------
    folder : str
        The path to the folder to be searched.
    """
    # Prompt the user for confirmation
    response = input(f"Do you want to delete all HDF5, npz, csv files in the given folder? [Y/n] \n {folder}")
    if response.lower() != 'y':
        print("No files were deleted.")
        return

    # Iterate through the files in the folder
    for file_ending in ['*.hdf5', '*.csv', '*.npz']:
        for file in Path(folder).rglob(file_ending):
            # Delete the file
            file.unlink()
            print(f"{file} deleted.")


@performance_counter
def main(paths: dict, years: list, out_path: Path):
    country_list = [
        # 'AUT',
        # 'BEL',
        # 'BGR',
        # 'HRV',
        # 'CYP',
        # 'CZE',
        # 'DNK',
        # 'EST',
        # 'FIN',
        # 'FRA',
        # 'DEU',
        # 'GRC',
        # 'HUN',
        # 'IRL',
        # 'ITA',
        # 'LVA',
        # 'LTU',
        # 'LUX',
        # 'MLT',
        # 'NLD',
        # 'POL',
        # 'PRT',
        # 'ROU',
        # 'SVK',
        # 'SVN',
        # 'ESP',
        # 'SWE'
    ]
    # copy the hdf files, NOT NEEDED AS THE READ HDF5 USES hdf5 file from orig repository
    # copy_hdf5_files(path_dict=paths, out_path=out_path, countries=country_list)

    # copy distribution csv files:
    copy_distribution_csvs(path_dict=paths, out_path=out_path, countries=country_list)

    # copy dynamic calc data
    copy_dynamic_calc_data(path_dict=paths, out_path=out_path, countries=country_list, years=years)

    # use multiprocessing:
    hdf5_was_copied = False
    arglist = [(country, out_path, years, paths, hdf5_was_copied) for country in country_list]
    read_hdf5("AUT", out_path, years, paths, hdf5_was_copied)  # for debugging
    # cores = int(multiprocessing.cpu_count() / 2)
    # with multiprocessing.Pool(cores) as pool:
    #     pool.starmap(read_hdf5, arglist)
    print("create building files: Done")


if __name__ == "__main__":
    # user inputs:
    project_path = Path(
        r"E:/projects3/2021_ECEMF/invert/output")  # NewTrends: Path(r"E:\projects3\2022_NewTrends\invert\output")     ECEMF: Path(r"E:/projects3/2021_ECEMF/invert/output")
    invert_scenario = r"output_ecemf_invert_eelab_secondround_231130_am_pm"  # NewTrends: r"output_230825_hdf5_for_Philipp" ECEMF: r"output_ecemf_invert_eelab_secondround_231130_am_pm"  
    sub_scenario = "eff_high_elec_ab"   # NewTrends: eff_high_elec_ab, eff_moderate_const_costpot_ab        ECEMF: eff_high_elec_ab, eff_moderate_elec_ab
    # invert input data ("input") folder
    invert_input_path = Path(
        # r"E:\projects3\2022_NewTrends\invert\input\input_invert_renovcosts_new_trend"      # NewTrends
        r"E:\projects3\2021_ECEMF\invert\input\input_ecemf_invert_eelab_secondround_231115"  # ECEMF
    )
    # define path where data should be saved
    output_folder = Path(
        r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\building_data\Georgina_paper_newTrends"
    )
    years = [2020, 2030, 2040, 2050]

    paths = {"SOURCE_PATH": project_path,
             "INVERT_SCENARIO": invert_scenario,
             "SUB_SCENARIO": sub_scenario,
             "INVERT_INPUT": invert_input_path}

    create_dict_if_not_exists(output_folder)
    main(paths, years, output_folder)

    # maybe delete the hdf5 files to save disc space after its done
    # delete_hdf5_files(folder=output_folder)
    # delete all files from output folder
    # clean_up(folder=output_folder)
