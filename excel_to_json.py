# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:11:28 2022

@author: mascherbauer
"""
import pandas as pd
import numpy as np
import pyodbc
from pathlib import Path


# path_to_file = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\inputdata\NUTS2021.xlsx"
# NUTS_codes = pd.read_excel(path_to_file,
#                               sheet_name="NUTS & SR 2021",
#                               engine="openpyxl"
#                               )["Code 2021"].dropna().to_numpy()
# NUTS1 = []
# NUTS2 = []
# NUTS3 = []
# # iterate through NUTS code and create tables for NUTS1, NUTS2 and NUTS3
# for code in NUTS_codes:
#     if "Z" in code:  # drop extra nuts regions
#         continue
#     elif len(code) == 2:
#         continue
#     elif len(code) == 3:  # NUTS1
#         NUTS1.append(code)
#     elif len(code) == 4:  # NUTS2
#         NUTS2.append(code)
#     elif len(code) == 5:  # NUTS3
#         NUTS3.append(code)
#
# nuts1_country_column = [nuts[:2] for nuts in NUTS1]
# nuts2_country_column = [nuts[:2] for nuts in NUTS2]
# nuts3_country_column = [nuts[:2] for nuts in NUTS3]
#
# nuts1_frame = pd.DataFrame(np.column_stack([nuts1_country_column, NUTS1]), columns=["country", "nuts_id"])
# nuts2_frame = pd.DataFrame(np.column_stack([nuts2_country_column, NUTS2]), columns=["country", "nuts_id"])
# nuts3_frame = pd.DataFrame(np.column_stack([nuts3_country_column, NUTS3]), columns=["country", "nuts_id"])


# # frames to json:
# nuts1_frame.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\NUTS1.json", orient="table")
# nuts2_frame.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\NUTS2.json", orient="table")
# nuts3_frame.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\NUTS3.json", orient="table")



# path_to_file_heat_demand = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\inputdata\heat_demand_nuts_3.csv"

# heat_demand_df = pd.read_csv(path_to_file_heat_demand)
# heat_demand_df.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\heat_demand.json", orient="table")

# path_to_floor_area = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\inputdata\ground_floor_area_residential_nuts_3.csv"
# ground_floor_area = pd.read_csv(path_to_floor_area, sep=";")
# ground_floor_area = ground_floor_area.loc[:, ["nuts_id", "sum"]]
# ground_floor_area.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\residential_floor_area.json", orient="table")
#
#
# #%%
# path_to_synth_load = Path(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\inputdata\synthetic_profiles")
# synth_load_df = pd.DataFrame()
# for child in path_to_synth_load.iterdir():
#     for file in child.iterdir():
#         if file.suffix == ".xlsx":
#             df = pd.read_excel(file, sheet_name="Profile").set_index("Unnamed: 0")
#             household_profile = pd.to_numeric(df.loc[:, "H0"].drop("ts [UTC]"))
#             household_profile.index = pd.to_datetime(household_profile.index)
#             household_profile = household_profile.resample("1H").sum()
#             if len(household_profile) == 8785:  # leap year (drop 29th february)
#                 household_profile = household_profile[~((household_profile.index.month == 2) &
#                                                         (household_profile.index.day == 29))]
#             household_profile = household_profile.reset_index(drop=True).drop(0)  # drop first row (23:00-24:00 last year)
#             household_profile = household_profile.to_numpy() * 1_000  # from kWh in Wh
#             year = file.stem[-4:]
#             synth_load_df[year] = household_profile
#
# # save file to json
# synth_load_df.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\synthetic_load_household.json", orient="table")
#
# # Hot water demand profile
hot_water_path = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Philipp\inputdata\AUT\Hot_water_profile.xlsx"
hot_water = pd.read_excel(Path(hot_water_path), engine="openpyxl")
hot_water = hot_water["Profile"] * 1_000
# hot_water.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\hot_water_demand.json", orient="table")

path_to_prices = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Philipp\inputdata\Variable_prices\el_prices_2030_AT.xlsx"
prices = pd.read_excel(path_to_prices, sheet_name="Tabelle1")
prices = prices.loc[:, [53, 106, 211]].drop(0)
prices = prices.rename(columns={53: "53", 106: "106", 211: "211"})
# add first day as last day as one day is missing:
prices = pd.concat([prices, prices.loc[:24, :]], axis=0).reset_index(drop=True)
prices = prices.apply(pd.to_numeric)
# von €/MWh into cent/Wh
prices = prices * 100 / 1_000 / 1_000

prices.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\data\price_profiles.json", orient="table")

# path_to_synth_load = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\inputdata\synthetic_profiles\synthload2017\SynthLoad2017.mdb"
# driver = "{Microsoft Access Driver (*.mdb)}"
# PWD = ""

# con = pyodbc.connect('DRIVER={};DBQ={};PWD={}'.format(driver,path_to_synth_load,PWD))
# cur = con.cursor()

# SQL = 'SELECT * FROM all_tables;'
# rows = cur.execute(SQL).fetchall()




