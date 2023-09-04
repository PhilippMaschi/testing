import pandas as pd
import numpy as np
from pathlib import Path

country_code_correction = {
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "HRV": "HR",
    "CYP": "CY",
    "CZE": "CZ",
    "DNK": "DK",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "DEU": "DE",
    "GRC": "GR",
    "HUN": "HU",
    "IRL": "IE",
    "ITA": "IT",
    "LVA": "LV",
    "LTU": "LT",
    "LUX": "LU",
    "MLT": "MT",
    "NLD": "NL",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SVK": "SK",
    "SVN": "SI",
    "ESP": "ES",
    "SWE": "SE"
}
swapped_dict = {value: key for key, value in country_code_correction.items()}

gen_source = Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\NewTrends\AURES2 generation data export")
import_export_source = Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\NewTrends\AURES 2 import_export")

for scen in ["shiny", "leviathan"]:
    print(f"calculating {scen}")
    save_to = Path(
        r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\NewTrends\AURES2 generation data export") / f"gen_data_{scen}.csv"
    generation = pd.DataFrame()
    for year in [2030, 2040, 2050]:
        gen_data = pd.read_excel(
            gen_source / f"gen_data_{scen}_{year}.xlsx",
            engine="openpyxl",
            skiprows=3
        ).rename(columns={
            "Unnamed: 0": "year",
            "Unnamed: 1": "country",
        }).drop(columns=[
            "Unnamed: 2",
            "Unnamed: 3"
        ])

        summed_gen = gen_data.iloc[:, 2:].sum(axis=1)
        new_gen = pd.concat([gen_data[["year", "country"]], summed_gen], axis=1).rename(columns={0: "generation"})
        new_gen["country"] = new_gen["country"].str.replace("_A", "").str.replace("PO", "PL")
        generation = pd.concat([generation, new_gen], axis=0)

    new_generation = pd.DataFrame()
    for country2, country in swapped_dict.items():
        for year in [2030, 2040, 2050]:
            import_data = pd.read_excel(
                import_export_source / f"{country2}_shiny{year}.xlsx",
                engine="openpyxl",
                sheet_name="Import",
                skiprows=1
            ).drop(columns=["Year", "Region_Im", "S", "T"]).sum(axis=1).to_numpy()
            export_data = pd.read_excel(
                import_export_source / f"{country2}_shiny{year}.xlsx",
                engine="openpyxl",
                sheet_name="Export",
                skiprows=1
            ).drop(columns=["Year", "Region_Ex", "S", "T"]).sum(axis=1).to_numpy()
            print(f"{country} {year}")
            old_gen = generation.query(f"year == {year} and country == '{country2}'")
            old_gen["generation"] += import_data
            old_gen["generation"] -= export_data
            new_generation = pd.concat([new_generation, old_gen], axis=0)

    new_generation.to_csv(gen_source / f"gen_data_{scen}.csv", index=False)


entsoe_source = Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\ENTSOE Generation")
etnsoe_gen = pd.read_csv(entsoe_source / "ENTSOE_generation_MWh_2019.csv",
                         sep=";"
                         ).drop(columns=["Unnamed: 0"]).melt(var_name="country", value_name="generation")
etnsoe_gen["year"] = 2020

for scen in ["shiny", "leviathan"]:
    gen = pd.read_csv(gen_source / f"gen_data_{scen}.csv")
    final_df = pd.concat([gen, etnsoe_gen], axis=0)
    final_df.to_csv(gen_source / f"gen_data_{scen}.csv", index=False)


