# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:07:33 2022

@author: mascherbauer
"""
from pathlib import Path
import pandas as pd
import zipfile
import numpy as np
import h5py
import cchardet


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


def create_country_specific_path(paths: dict, country: str):
    return paths["SOURCE_PATH"] / paths["INVERT_SCENARIO"] / country / f"_scen_{country.lower()}{paths['SUB_SCENARIO']}"


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


def get_number_of_heat_pumps(residential_df: pd.DataFrame, segment_df: pd.DataFrame) -> pd.DataFrame:
    for index, row in residential_df.iterrows():
        number_of_buildings = segment_df.loc[
            segment_df.loc[:, "building_classes_index"] == row["index"]]
        total_number_of_buildings = number_of_buildings.loc[:, "number_of_buildings"].sum()

        residential_df.loc[index, "number_of_buildings"] = total_number_of_buildings
        # TODO add all heating systems
        # filter out number of heat pumps
        # Air:
        number_of_air_hp = segment_df.loc[
            (segment_df.loc[:, "building_classes_index"] == row["index"]) &
            (segment_df.loc[:, "heat_supply_system_index"] == 42)]
        total_number_of_air_hp = number_of_air_hp.loc[:, "number_of_buildings"].sum()
        residential_df.loc[index, "number_of_buildings_with_HP_air"] = total_number_of_air_hp

        # Ground:
        number_of_ground_hp = segment_df.loc[
            (segment_df.loc[:, "building_classes_index"] == row["index"]) &
            (segment_df.loc[:, "heat_supply_system_index"] == 43)]
        total_number_of_ground_hp = number_of_ground_hp.loc[:, "number_of_buildings"].sum()
        residential_df.loc[index, "number_of_buildings_with_HP_ground"] = total_number_of_ground_hp

    return residential_df


def find_hdf5_file(directory: Path):
    for file in directory.rglob('*.hdf5'):
        return file
    return None


def read_hdf5(paths: dict, country: str):
    # find list with hdf5 files
    print("Find hdf5 file.")
    main_directory = create_country_specific_path(paths=paths, country=country)
    filename_hdf5 = find_hdf5_file(main_directory)

    with h5py.File(filename_hdf5, 'r') as hdf5_f:
        #show the strucuter of the hdf file:
        print(hdf5_f.keys())
        # get data elements stored in hdf5 container
        item_names_hdf5 = hdf5_f.items()
        # Extract simulation periods stored in data container
        year_list_hdf5 = []
        for curr_item_name in item_names_hdf5:
            if curr_item_name[0][:3] == "BC_":
                year_list_hdf5.append(int(curr_item_name[0][-4:]))

        print(f"Years stored in data container: {year_list_hdf5}")


        def hdf5_to_structured_array(hdf5_file, group_name, columns) -> pd.DataFrame:
            # Get the table from the group
            table = hdf5_file[group_name]

            # Get the names of the columns (headers)
            headers = list(table.dtype.names)

            # Select the desired columns and data types
            dtype = []
            for name in headers:
                    if name in columns:
                        dtype.append((name, table.dtype[name]))

            # Read the data for the selected columns
            data = np.array(table, dtype=dtype)

            new_np_array = np.zeros(shape=(1, len(np.array(str(data[0]).split(',')))))
            # Split the data in each cell into multiple cells using the ";" delimiter
            for i in range(data.shape[0]):
                a = np.array(str(data[i]).split(','))
                for j, value in enumerate(a):
                    a[j] = value.replace("(", "").replace(")", "").replace("b", "").replace("'", "")
                new_np_array = np.append(new_np_array, np.expand_dims(a, axis=0), axis=0)

            # drop the first column as its only zeros:
            new_array = new_np_array[1:]

            pandas_frame = pd.DataFrame(new_array).rename(columns={i: key for i, key in enumerate(building_classes_columns.keys())})

            return pandas_frame

        building_classes_columns = {
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
        array_np = hdf5_to_structured_array(hdf5_f, "BC_2020", building_classes_columns)
        # fetch the building classes
        BC_years = [f"BC_{year}" for year in [2020, 2030, 2050]]
        bc_header = hdf5_f["BC_2020"].dtype.names
        bc = hdf5_f["BC_2020"][()].tostring().decode("utf-16", errors="ignore")
        data = np.genfromtxt(bc, delimiter=',', dtype='float64')



        # Specify the column names and data types
        dtype = [(name, 'float32') if name!="name" else (name, "str") for name in bc_header]

        # Transfer the data to a NumPy array
        array = np.array(data_list, dtype=dtype)

        RESULTS = {}


def main(paths: dict, year: int):
    hdf_filename = "001_buildings.hdf5"
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

    # Define a dict comprehension to generate the heating_key dictionary
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

    df_all_countries = pd.DataFrame(columns=[
        "index",
        "name",
        "construction_period_start",
        "construction_period_end",
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
        "hwb_norm",
        "number_of_buildings",
        "number_of_buildings_with_HP_ground",
        "number_of_buildings_with_HP_air",
        "country"])

    df_mfh_all_countries = df_all_countries.copy()

    for country in country_list:
        # load the hdf5 file:
        country = "CYP"
        read_hdf5(paths, country)
        building_classes, building_segment, dynamic_data = fetch_data(paths, year, country)
        building_df_merged = pd.concat([building_classes, dynamic_data], axis=1)

        # filter out all buildings that are not SFH or MFH
        residential_building_df = building_df_merged.loc[
                                  (building_df_merged.loc[:, "building_categories_index"] == 1) |  # SFH
                                  (building_df_merged.loc[:, "building_categories_index"] == 2) |  # SFH
                                  (building_df_merged.loc[:, "building_categories_index"] == 5) |  # MFH
                                  (building_df_merged.loc[:, "building_categories_index"] == 6), :]  # MFH

        # get the total number of buildings for each bc index and the total number
        # of buildings using a heat pump
        residential_building_df["number_of_buildings"] = 0
        residential_building_df["number_of_buildings_with_HP_ground"] = 0
        residential_building_df["number_of_buildings_with_HP_air"] = 0

        # add country to building_df:
        residential_building_df["country"] = country
        # add the number of heat pumps
        residential_building_df_enhanced = get_number_of_heat_pumps(residential_building_df, building_segment)

        # append the df to the big df
        df_all_countries = df_all_countries.append(residential_building_df_enhanced)


        print(f"added {country}")
    df_all_countries = df_all_countries.reset_index(drop=True).rename(columns={"index": "invert_index"})

    # df_all_countries.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\data\SFH_building_data.json",
    #                          orient="table")

    df_mfh_all_countries = df_mfh_all_countries.reset_index(drop=True).rename(columns={"index": "invert_index"})

    # df_mfh_all_countries.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\data\MFH_building_data.json",
    #                          orient="table")


if __name__ == "__main__":
    paths = {"SOURCE_PATH": Path(r"W:\projects3\2021_RES_HC_Pathways\invert"),
             "INVERT_SCENARIO": "output_new_trends_2022_12_20_2050",
             "SUB_SCENARIO": "_res_hc_pw_alternative_1_ab"}  # always has _sce_country before
    years = [2020, 2030, 2050]
    main(paths, years)
