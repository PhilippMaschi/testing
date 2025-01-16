import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


COLUMNS = [
    "country_name",
    "subsector",
    "AREA [m2]_BASE_YEAR",
    "fed_sh [kWh/m2]_BASE_YEAR",
    "AREA [m2]_2050",
    "fed_sh [kWh/m2]_2050",
    "cum_inv_env_exist_build [EUR/all m2]_2050",
    "cum_inv_env_exist_build [EUR/m2] where investment != 0_2050",

]



PATH_2_SUMMARY = Path(r"X:\projects3\2021_ECEMF\invert\output\output_ecemf_invert_eelab_secondround_231130_am_pm\__SUMMARY_DATA")


def load_cum_sum_file(path2file: Path):
    try:
        # Read the header separately
        with open(path2file, "r", encoding="utf-8") as f:
            header = f.readline().strip().split('"')[1:]
        
        columns = [header[0]] + header[-1].split(",")[1:]
        
        # Read the file with the specified encoding
        df = pd.read_csv(path2file, sep=",", header=0, names=columns, encoding="utf-8").iloc[0:10, ]
    except UnicodeDecodeError:
        # Retry with a different encoding
        with open(path2file, "r", encoding="latin1") as f:
            header = f.readline().strip().split('"')[1:]
        
        columns = [header[0]] + header[-1].split(",")[1:]
        df = pd.read_csv(path2file, sep=",", header=0, names=columns, encoding="latin1").iloc[0:10, ]
    
    first_column = df.columns[0]
    df.rename(columns={first_column: "name"}, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def calc_numbers(cum_df: pd.DataFrame) ->(float, float):
    sfh = 0
    mfh = 0
    for name in cum_df["name"]:
        if "sfh" in name.lower() and "_unused" not in name:
            sfh += cum_df.loc[cum_df["name"]==name, "2050"].values[0]
        elif "mfh" in name.lower() and "_unused" not in name:
            mfh += cum_df.loc[cum_df["name"]==name, "2050"].values[0]
    return sfh, mfh


def get_summary_areas_all_countries(filename: Path) -> pd.DataFrame:
    df = pd.read_csv(PATH_2_SUMMARY/filename, sep=",")[COLUMNS]
    df = df.loc[(df["subsector"].isin([11, 12])) & (df["AREA [m2]_BASE_YEAR"]>0), :]
    area = df.groupby(["country_name", "subsector"])["AREA [m2]_2050"].sum().reset_index()
    area["subsector"] = area["subsector"].map({11: "SFH", 12:"MFH"})
    return area


def get_summary_areas(filename: Path) -> pd.DataFrame:
    df = pd.read_csv(PATH_2_SUMMARY/filename, sep=",")[COLUMNS]
    df = df.loc[(df["country_name"].isin(["ESP", "NLD"])) & (df["subsector"].isin([11, 12])) & (df["AREA [m2]_BASE_YEAR"]>0), :]

    area = df.groupby(["country_name", "subsector"])["AREA [m2]_2050"].sum().reset_index()
    area["subsector"] = area["subsector"].map({11: "SFH", 12:"MFH"})
    return area

def calc_rennovation_rate(cum_df: pd.DataFrame, area_df: pd.DataFrame, country: str, scenario: str):
    sfh, mfh = calc_numbers(cum_df=cum_df)
    country_df = area_df.loc[area_df["country_name"]==country, :]

    sfh_renov_rate = sfh / (country_df.loc[country_df["subsector"]=="SFH", "AREA [m2]_2050"].values[0]/1e6) /1_000 / 31 * 100
    mfh_renov_rate = mfh / (country_df.loc[country_df["subsector"]=="MFH", "AREA [m2]_2050"].values[0]/1e6) /1_000 / 31 * 100
    print(f"The renovation rate from 2019 until 2050 for SFH in {country} for the {scenario} scenario is: {sfh_renov_rate.round(2)}%.")
    print(f"The renovation rate from 2019 until 2050 for MFH in {country} for the {scenario} scenario is: {mfh_renov_rate.round(2)}%.")
    print("")

def calc_rennovation_rate_EU(scenario: str):
    if scenario == "high":
        filename="_EURAC_format_results_Invert_CP_eff_high_elec_ab_2050.csv"
    else:
        filename="_EURAC_format_results_EURAC_CP_eff_moderate_elec_ab_2050.csv"
    area_df = get_summary_areas_all_countries(filename=filename)
    countries = area_df["country_name"].unique()
    zaehler = 0
    nenner = 0
    for country in countries:
        path2_cum_fa = Path(r"X:\projects3\2021_ECEMF\invert\output\output_ecemf_invert_eelab_secondround_231130_am_pm") / f"{country}" / f"_scen_{country.lower()}_eff_high_elec_ab" / "001_bca_fa_cum_heated_gross_floor_area_renov_build_sh.csv"
        cum_area = load_cum_sum_file(path2_cum_fa)
        sfh_renov_area, mfh_renov_area = calc_numbers(cum_df=cum_area)
        country_df = area_df.loc[area_df["country_name"]==country, :]
        sfh_total_area = country_df.loc[country_df["subsector"]=="SFH", "AREA [m2]_2050"].values[0]/1e3
        mfh_total_area = country_df.loc[country_df["subsector"]=="MFH", "AREA [m2]_2050"].values[0]/1e3

        zaehler += sfh_renov_area + mfh_renov_area
        nenner += sfh_total_area + mfh_total_area

    rennovation_rate_eu = zaehler / nenner / 31 * 100
    print(f"The average rennovation rate for residential buildings over all of Europe is {round(rennovation_rate_eu, 2)}% in the {scenario} scenario.")

def get_final_energy_demand_EU(filename: str, scenario: str):
    df = pd.read_csv(PATH_2_SUMMARY/filename, sep=",")[COLUMNS]
    df = df.loc[df["subsector"].isin([11, 12]), :].copy()

    df.loc[:, "fed_kWh_baseyear"] = df["fed_sh [kWh/m2]_BASE_YEAR"] * df["AREA [m2]_BASE_YEAR"]
    df.loc[:, "fed_kWh_2050"] = df["fed_sh [kWh/m2]_2050"] * df["AREA [m2]_2050"]
    
    new_df = df.groupby(["country_name"])[["fed_kWh_baseyear", "fed_kWh_2050", "AREA [m2]_2050", "AREA [m2]_BASE_YEAR"]].sum().reset_index()
    new_df.loc[:, "fed_kWh/m2_baseyear"] = new_df["fed_kWh_baseyear"] / new_df["AREA [m2]_BASE_YEAR"]
    new_df.loc[:, "fed_kWh/m2_2050"] = new_df["fed_kWh_2050"] / new_df["AREA [m2]_2050"]

    fed_2020 = new_df.loc[:, "fed_kWh/m2_baseyear"].mean().round(2)
    fed_2050 = new_df.loc[:, "fed_kWh/m2_2050"].mean().round(2)

    print(f"In the EU27 the average final energy demand (kWh/m2) goes from {fed_2020}kWh/m2 (2019) to {fed_2050}kWh/m2 in 2050. Thus it is reduced by {round((fed_2020-fed_2050)/fed_2020*100,2)}% in the {scenario} scenario.")
    print("")


def get_final_emergy_demand(filename: str, scenario: str, country: str):
    df = pd.read_csv(PATH_2_SUMMARY/filename, sep=",")[COLUMNS]
    df = df.loc[(df["country_name"].isin(["ESP", "NLD"])) & (df["subsector"].isin([11, 12])), :].copy()

    df.loc[:, "fed_kWh_baseyear"] = df["fed_sh [kWh/m2]_BASE_YEAR"] * df["AREA [m2]_BASE_YEAR"]
    df.loc[:, "fed_kWh_2050"] = df["fed_sh [kWh/m2]_2050"] * df["AREA [m2]_2050"]
    
    new_df = df.groupby(["country_name"])[["fed_kWh_baseyear", "fed_kWh_2050", "AREA [m2]_2050", "AREA [m2]_BASE_YEAR"]].sum().reset_index()
    new_df.loc[:, "fed_kWh/m2_baseyear"] = new_df["fed_kWh_baseyear"] / new_df["AREA [m2]_BASE_YEAR"]
    new_df.loc[:, "fed_kWh/m2_2050"] = new_df["fed_kWh_2050"] / new_df["AREA [m2]_2050"]

    fed_2020 = new_df.loc[new_df["country_name"]==country, "fed_kWh/m2_baseyear"].values[0].round(2)
    fed_2050 = new_df.loc[new_df["country_name"]==country, "fed_kWh/m2_2050"].values[0].round(2)

    print(f"In {country} the average final energy demand (kWh/m2) goes from {fed_2020}kWh/m2 (2019) to {fed_2050}kWh/m2 in 2050. Thus it is reduced by {round((fed_2020-fed_2050)/fed_2020*100,2)}% in the {scenario} scenario.")
    print("")


if __name__ == "__main__":
    path2_cum_fa_ESP_moderate = Path(r"X:\projects3\2021_ECEMF\invert\output\output_ecemf_invert_eelab_secondround_231130_am_pm\ESP\_scen_esp_eff_moderate_elec_ab") / "001_bca_fa_cum_heated_gross_floor_area_renov_build_sh.csv"
    path2_cum_fa_NLD_moderate = Path(r"X:\projects3\2021_ECEMF\invert\output\output_ecemf_invert_eelab_secondround_231130_am_pm\NLD\_scen_nld_eff_moderate_elec_ab") / "001_bca_fa_cum_heated_gross_floor_area_renov_build_sh.csv"
    path2_cum_fa_ESP_high = Path(r"X:\projects3\2021_ECEMF\invert\output\output_ecemf_invert_eelab_secondround_231130_am_pm\ESP\_scen_esp_eff_high_elec_ab") / "001_bca_fa_cum_heated_gross_floor_area_renov_build_sh.csv"
    path2_cum_fa_NLD_high = Path(r"X:\projects3\2021_ECEMF\invert\output\output_ecemf_invert_eelab_secondround_231130_am_pm\NLD\_scen_nld_eff_high_elec_ab") / "001_bca_fa_cum_heated_gross_floor_area_renov_build_sh.csv"

    cum_area_ESP_moderate = load_cum_sum_file(path2_cum_fa_ESP_moderate)
    cum_area_NLD_moderate = load_cum_sum_file(path2_cum_fa_NLD_moderate)
    cum_area_ESP_high = load_cum_sum_file(path2_cum_fa_ESP_high)
    cum_area_NLD_high = load_cum_sum_file(path2_cum_fa_NLD_high)

    get_final_energy_demand_EU(filename="_EURAC_format_results_Invert_CP_eff_high_elec_ab_2050.csv",
                               scenario="high",
                               )
    calc_rennovation_rate_EU(scenario="high")
    calc_rennovation_rate_EU(scenario="moderate")

    calc_rennovation_rate(cum_df=cum_area_ESP_moderate,
                        area_df=get_summary_areas(filename="_EURAC_format_results_EURAC_CP_eff_moderate_elec_ab_2050.csv"),
                        country="ESP",
                        scenario="moderate")

    calc_rennovation_rate(cum_df=cum_area_ESP_high,
                        area_df=get_summary_areas(filename="_EURAC_format_results_Invert_CP_eff_high_elec_ab_2050.csv"),
                        country="ESP",
                        scenario="high")

    calc_rennovation_rate(cum_df=cum_area_NLD_moderate,
                        area_df=get_summary_areas(filename="_EURAC_format_results_EURAC_CP_eff_moderate_elec_ab_2050.csv"),
                        country="NLD",
                        scenario="moderate")

    calc_rennovation_rate(cum_df=cum_area_NLD_high,
                        area_df=get_summary_areas(filename="_EURAC_format_results_Invert_CP_eff_high_elec_ab_2050.csv"),
                        country="NLD",
                        scenario="high")
    get_final_emergy_demand(filename="_EURAC_format_results_EURAC_CP_eff_moderate_elec_ab_2050.csv",
                            scenario="moderate",
                            country="ESP"
                            )
    get_final_emergy_demand(filename="_EURAC_format_results_Invert_CP_eff_high_elec_ab_2050.csv",
                            scenario="high",
                            country="ESP"
                            )

    get_final_emergy_demand(filename="_EURAC_format_results_EURAC_CP_eff_moderate_elec_ab_2050.csv",
                            scenario="moderate",
                            country="NLD"
                            )
    get_final_emergy_demand(filename="_EURAC_format_results_Invert_CP_eff_high_elec_ab_2050.csv",
                            scenario="high",
                            country="NLD"
                            )
