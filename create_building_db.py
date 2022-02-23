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
    building_df["number_of_buildings"] = 0
    building_df["number_of_buildings_with_HP_ground"] = 0
    building_df["number_of_buildings_with_HP_air"] = 0
    bc_indices = building_df.loc[:, "index"].to_numpy()
    
    for index, row in building_df.iterrows():
        number_of_buildings = building_segment.loc[
            building_segment.loc[:, "building_classes_index"] == row["index"]]
        total_number_of_buildings = number_of_buildings.loc[:, "number_of_buildings"].sum()
        
        building_df.loc[index, "number_of_buildings"] = total_number_of_buildings
        
        # filter out number of heat pumps 
    

    
   


