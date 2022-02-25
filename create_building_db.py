# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:07:33 2022

@author: mascherbauer
"""
from pathlib import Path
import pandas as pd
import zipfile
import numpy as np

source_path = "E:/projects/2021_RES_HC_Pathways/invert/output_2022_newbuild_newlifetime_reference_renshare"

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
                'SWE',
                'GBR']

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

for country in country_list:

    sub_folder_name = "_scen_"+ country.lower() + "_electrification_ref_final_ab/ADD_RESULTS/"
    # building classes:  
    building_classes_file_name = "001_Building_Classes_2019.csv"
    
    # data for 5R1C model:
    dynamic_calc_folder = "Dynamic_Calc_Input_Data"
    dynamic_calc_data = "001__dynamic_calc_data_bc_2019.csv"
    dynamic_calc_data_npz = "001__dynamic_calc_data_bc_2019.npz"
    
    # building segment data for number of heat pumps:
    building_segment_folder = "001_Building_segment_SH_2019.zip"
    building_segment_folder_zip = zipfile.ZipFile(Path(source_path) / Path(country) / Path(sub_folder_name) / Path(building_segment_folder))
    
    path_to_building_class = Path(source_path) / Path(country) / Path(sub_folder_name) / Path(building_classes_file_name)
    
    path_to_dynamic_data = Path(source_path) / Path(country) / Path(sub_folder_name) / Path(dynamic_calc_folder) / Path(dynamic_calc_data)
    path_to_dynamic_data_npz = Path(source_path) / Path(country) / Path(sub_folder_name) / Path(dynamic_calc_folder) / Path(dynamic_calc_data_npz)
    
    # read dataframes:
    building_classes_orig = pd.read_csv(path_to_building_class)
    dynamic_data_orig = pd.read_csv(path_to_dynamic_data)
    building_segment_orig = pd.read_csv(building_segment_folder_zip.open("001_Building_segment_SH_2019.csv"))
    
    # filter building class columns:
    building_classes = building_classes_orig.loc[:, [
        "index",
        "name",
        "construction_period_start",
        "construction_period_end"]
        ]
    
    # filter dynamic data:
    dynamic_data = dynamic_data_orig.loc[:, [
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
    building_segment = building_segment_orig.loc[:, [
        "index", 
         "name", 
         "building_classes_index", 
         "number_of_buildings", 
         "energy_carrier_region_index", 
         "heat_supply_system_index"]
        ]
    
    
    building_df_merged = pd.concat([building_classes, dynamic_data], axis=1)
    
    # filter out all buildings that are not SFH
    building_df = building_df_merged.loc[building_df_merged["name"].str.contains("sfh", case=False)]

    # get the total number of buildings for each bc index and the total number 
    # of buildings using a heat pump
    building_df.loc[:, "number_of_buildings"] = 0
    building_df.loc[:,"number_of_buildings_with_HP_ground"] = 0
    building_df.loc[:,"number_of_buildings_with_HP_air"] = 0
    
    # add country to building_df:
    building_df.loc[:,"country"] = country
    
    for index, row in building_df.iterrows():
        number_of_buildings = building_segment.loc[
            building_segment.loc[:, "building_classes_index"] == row["index"]]
        total_number_of_buildings = number_of_buildings.loc[:, "number_of_buildings"].sum()
        
        building_df.loc[index, "number_of_buildings"] = total_number_of_buildings
        
        # filter out number of heat pumps 
        # Air:
        number_of_air_hp = building_segment.loc[
            (building_segment.loc[:, "building_classes_index"] == row["index"]) &
            (building_segment.loc[:, "heat_supply_system_index"] == 42)]
        total_number_of_air_hp = number_of_air_hp.loc[:, "number_of_buildings"].sum()
        building_df.loc[index, "number_of_buildings_with_HP_air"] = total_number_of_air_hp
        
        # Ground:
        number_of_ground_hp = building_segment.loc[
            (building_segment.loc[:, "building_classes_index"] == row["index"]) & 
            (building_segment.loc[:, "heat_supply_system_index"] == 43)]
        total_number_of_ground_hp = number_of_ground_hp.loc[:, "number_of_buildings"].sum() 
        building_df.loc[index, "number_of_buildings_with_HP_ground"] = total_number_of_ground_hp
        
    # append the df to the big df
    df_all_countries = df_all_countries.append(building_df)
    
           
    print(f"added {country}")
df_all_countries = df_all_countries.reset_index(drop=True).rename(columns={"index": "invert_index"})
 
df_all_countries.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\SFH_building_data.json", orient="table")

   


