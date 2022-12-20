# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:07:33 2022

@author: mascherbauer
"""
from pathlib import Path
import pandas as pd
import zipfile
import numpy as np


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


def main(paths: dict, year: int):
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
             "INVERT_SCENARIO": "output_2022_newbuild_reference",
             "SUB_SCENARIO": "_electrification_ref_final_ab"}  # always has _sce_country before
    years = [2020, 2030, 2050]
    main(paths)
