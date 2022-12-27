# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:07:33 2022

@author: mascherbauer
"""
from pathlib import Path

# import chardet
import pandas as pd
import zipfile
import numpy as np
import h5py
import shutil
# import cchardet
import re
import multiprocessing




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

def copy_hdf5_to_disc(source: Path, destination: Path):
    """Copy an HDF5 file from src to dst, creating any missing directories in the destination path."""
    # Check if the destination file already exists
    if destination.exists():
        print(f"{destination} already exists. Skipping copy operation.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination)



def read_hdf5(paths: dict, country: str, output_path: Path, years: list,
              building_class_columns: dict,
              building_segment_columns: dict,
              heating_system_columns: dict,
              energy_carrier_index: dict):

        def hdf5_to_pandas(hdf5_file: Path, group_name, columns) -> pd.DataFrame:
            with h5py.File(hdf5_file, 'r') as file:
                # Get the table from the group
                dataset = file[group_name]
                df = pd.DataFrame(index=range(len(dataset)), columns=[list(columns.keys())])
                for name in columns.keys():
                    df[name] = dataset[name]

            return df


        # create country specific path
        main_directory = Path(__file__).parent / "building_data" / country
        hdf5_f = find_hdf5_file(main_directory)
        BC_2020 = hdf5_to_pandas(hdf5_f, "BC_2020", building_class_columns)
        BSSH_2020 = hdf5_to_pandas(hdf5_f, "BSSH_2020", building_segment_columns)
        heating_system = hdf5_to_pandas(hdf5_f, "HeatingSystems", heating_system_columns)
        energy_carrier = hdf5_to_pandas(hdf5_f, "EnergyCarrierDefinition_2020", energy_carrier_index)



        # BC_years = [f"BC_{year}" for year in years]
        # # create country dir if it doesnt exist:
        # create_dict_if_not_exists(output_path / country)
        # for name in BC_years:
        #     df = hdf5_to_pandas(hdf5_f, name, building_class_columns)
        #     # df.to_csv(output_path / country / f"{name}.csv", sep=";", index=False)
        #     df.to_parquet(output_path / country / f'{name}.parquet.gzip', compression='gzip')
        # print(f"BC for {country} downloaded.")
        #
        # BSSH_years = [f"BSSH_{year}" for year in years]
        # for name in BSSH_years:
        #     df = hdf5_to_pandas(hdf5_f, name, building_segment_columns)
        #     # df.to_csv(output_path / country / f"{name}.csv", sep=";", index=False)
        #     df.to_parquet(output_path / country / f'{name}.parquet.gzip', compression='gzip')
        # print(f"BSSH for {country} downloaded.")
        #
        # energy_carrier_years = [f"EnergyCarrierDefinition_{year}" for year in years]
        # for name in energy_carrier_years:
        #     df = hdf5_to_pandas(hdf5_f, name, energy_carrier_index)
        #     # df.to_csv(output_path / country / f"{name}.csv", sep=";", index=False)
        #     df.to_parquet(output_path / country / f'{name}.parquet.gzip', compression='gzip')
        # print(f"Energy carriers for {country} downloaded.")
        #
        # # get Heating Systems
        # df = hdf5_to_pandas(hdf5_f, "HeatingSystems", heating_system_columns)
        # # df.to_csv(output_path / country / f"HeatingSystems.csv", sep=";", index=False)
        # df.to_parquet(output_path / country / f'{name}.parquet.gzip', compression='gzip')

        print(f"Finished retrieving data from {country}.")


def copy_hdf5_files(out_path: Path, countries: list):
    for country in countries:
        source_directory = paths["SOURCE_PATH"] / paths[
            "INVERT_SCENARIO"] / country / f"_scen_{country.lower()}{paths['SUB_SCENARIO']}"
        source_path = find_hdf5_file(source_directory)
        destination_path = out_path / country / source_path.name
        # create_dict_if_not_exists(destination_path)
        copy_hdf5_to_disc(source_path, destination_path)
    print("all hdf5 files are copied to local disk")


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


def main(paths: dict, years: list, out_path: Path, building_class_columns, building_segment_columns, heating_system_columns, energy_carrier_index):
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
    # copy the hdf files
    # copy_hdf5_files(out_path=out_path, countries=country_list)

    for country in country_list:
        read_hdf5(paths, country, out_path, years, building_class_columns, building_segment_columns, heating_system_columns, energy_carrier_index)





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



    arglist = [(paths, country, out_path, years, building_class_columns, building_segment_columns, heating_system_columns, energy_carrier_index) for country in country_list]
    # with multiprocessing.Pool(4) as pool:
        # pool.starmap(read_hdf5, arglist)


    # for country in country_list:
        #load the hdf5 file:
    # country = "CYP"
    # read_hdf5(paths, country, out_path, years, building_class_columns, building_segment_columns, heating_system_columns, energy_carrier_index)


        # building_classes, building_segment, dynamic_data = fetch_data(paths, year, country)
        # building_df_merged = pd.concat([building_classes, dynamic_data], axis=1)
        #
        # # filter out all buildings that are not SFH or MFH
        # residential_building_df = building_df_merged.loc[
        #                           (building_df_merged.loc[:, "building_categories_index"] == 1) |  # SFH
        #                           (building_df_merged.loc[:, "building_categories_index"] == 2) |  # SFH
        #                           (building_df_merged.loc[:, "building_categories_index"] == 5) |  # MFH
        #                           (building_df_merged.loc[:, "building_categories_index"] == 6), :]  # MFH
        #
        # # get the total number of buildings for each bc index and the total number
        # # of buildings using a heat pump
        # residential_building_df["number_of_buildings"] = 0
        # residential_building_df["number_of_buildings_with_HP_ground"] = 0
        # residential_building_df["number_of_buildings_with_HP_air"] = 0
        #
        # # add country to building_df:
        # residential_building_df["country"] = country
        # # add the number of heat pumps
        # residential_building_df_enhanced = get_number_of_heat_pumps(residential_building_df, building_segment)
        #
        # # append the df to the big df
        # df_all_countries = df_all_countries.append(residential_building_df_enhanced)
        #
        #
        # print(f"added {country}")

    # df_all_countries = df_all_countries.reset_index(drop=True).rename(columns={"index": "invert_index"})

    # df_all_countries.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\data\SFH_building_data.json",
    #                          orient="table")

    # df_mfh_all_countries = df_mfh_all_countries.reset_index(drop=True).rename(columns={"index": "invert_index"})

    # df_mfh_all_countries.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\data\MFH_building_data.json",
    #                          orient="table")


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
    main(paths, years, output_folder, BUILDING_CLASS_COLUMNS, BUILDING_SEGMENT_COLUMNS, HEATING_SYSTEM_COLUMNS, ENERGY_CARRIER_INDEX)
