# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:11:28 2022

@author: mascherbauer
"""
import pandas as pd
import numpy as np


path_to_file = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\inputdata\NUTS2021.xlsx"
NUTS_codes = pd.read_excel(path_to_file,
                              sheet_name="NUTS & SR 2021",
                              engine="openpyxl"
                              )["Code 2021"].dropna().to_numpy()
NUTS1 = []
NUTS2 = []
NUTS3 = []
# iterate through NUTS code and create tables for NUTS1, NUTS2 and NUTS3
for code in NUTS_codes:
    if "Z" in code:  # drop extra nuts regions
        continue
    elif len(code) == 2:
        continue
    elif len(code) == 3:  # NUTS1
        NUTS1.append(code)
    elif len(code) == 4:  # NUTS2
        NUTS2.append(code)
    elif len(code) == 5:  # NUTS3
        NUTS3.append(code)

nuts1_country_column = [nuts[:2] for nuts in NUTS1]
nuts2_country_column = [nuts[:2] for nuts in NUTS2]
nuts3_country_column = [nuts[:2] for nuts in NUTS3]

nuts1_frame = pd.DataFrame(np.column_stack([nuts1_country_column, NUTS1]), columns=["country", "nuts_id"])
nuts2_frame = pd.DataFrame(np.column_stack([nuts2_country_column, NUTS2]), columns=["country", "nuts_id"])
nuts3_frame = pd.DataFrame(np.column_stack([nuts3_country_column, NUTS3]), columns=["country", "nuts_id"])


# frames to json:
nuts1_frame.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\NUTS1.json", orient="table")
nuts2_frame.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\NUTS2.json", orient="table")
nuts3_frame.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\NUTS3.json", orient="table")



path_to_file_heat_demand = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\inputdata\heat_demand_nuts_3.csv"

heat_demand_df = pd.read_csv(path_to_file_heat_demand)
heat_demand_df.to_json(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\heat_demand.json", orient="table")

