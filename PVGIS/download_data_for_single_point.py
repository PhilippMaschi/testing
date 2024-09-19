import pandas as pd
import numpy as np
import urllib.error
from pathlib import Path


def scalar2array(value):
    return [value for _ in range(0, 8760)]


def get_temperature(config: str) -> dict:
    temperature_dict = {}
    try:
        df = get_temperature_and_solar_radiation(config, 0)
        temperature_dict["temperature"] = pd.to_numeric(df["T2m"].reset_index(drop=True)).values
        temperature_dict["temperature_unit"] = scalar2array("C")
        return temperature_dict
    except Exception as e:
        print(f"Temperature source is not available for region {config['region_name']}.")

def get_radiation(config) -> dict:
    radiation_dict = {}
    celestial_direction_aspect = {
        "south": 0,
        "east": -90,
        "west": 90,
        "north": -180
    }
    try:
        for direction, aspect in celestial_direction_aspect.items():
            df = get_temperature_and_solar_radiation(config, aspect)
            radiation = pd.to_numeric(df["Gb(i)"]) + pd.to_numeric(df["Gd(i)"])
            radiation_dict["radiation_" + direction] = radiation.reset_index(drop=True).to_numpy()
        radiation_dict["radiation_unit"] = scalar2array("W")
        return radiation_dict
    except Exception as e:
        print(f"Radiation source is not available for region {config['region_name']}.")

def get_pv_generation(config: str) -> np.array:
    pv_generation_dict = {}
    pv_calculation = 1
    optimal_inclination = 1
    optimal_angle = 1
    lat, lon = config["lat"], config["lon"]
    req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}&" \
            f"startyear={config['start_year']}&" \
            f"endyear={config['end_year']}&" \
            f"pvcalculation={pv_calculation}&" \
            f"peakpower={config['peak_power']}&" \
            f"loss={config['pv_loss']}&" \
            f"pvtechchoice={config['pv_tech']}&" \
            f"components={1}&" \
            f"trackingtype={config['tracking_type']}&" \
            f"optimalinclination={optimal_inclination}&" \
            f"optimalangles={optimal_angle}"
    try:
        # Read the csv from api and use 20 columns to receive the source, because depending on the parameters,
        # the number of columns could vary. Empty columns are dropped afterwards:
        df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
        df = df.dropna().reset_index(drop=True)
        # set header to first row
        header = df.iloc[0]
        df = df.iloc[1:, :]
        df.columns = header
        df = df.reset_index(drop=True)
        pv_generation_dict["pv_generation"] = pd.to_numeric(df["P"]).to_numpy()  # unit: W
        pv_generation_dict["pv_generation_unit"] = scalar2array("W")
        return pv_generation_dict
    except urllib.error.HTTPError:
        print(f"pv_generation source is not available for region {config['region_name']}.")


def get_temperature_and_solar_radiation(config, aspect) -> pd.DataFrame:
    pv_calculation = 0
    optimal_inclination = 0
    optimal_angle = 0
    lat, lon = config["lat"], config["lon"]
    req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}&" \
            f"startyear={config['start_year']}&" \
            f"endyear={config['end_year']}&" \
            f"pvcalculation={pv_calculation}&" \
            f"peakpower={config['pv_peak_power']}&" \
            f"loss={config['pv_loss']}&" \
            f"pvtechchoice={config['pv_tech']}&" \
            f"components={1}&" \
            f"trackingtype={config['tracking_type']}&" \
            f"optimalinclination={optimal_inclination}&" \
            f"optimalangles={optimal_angle}&" \
            f"angle={config['angle']}&" \
            f"aspect={aspect}"

    # Read the csv from api and use 20 columns to receive the source, because depending on the parameters,
    # the number of columns could vary. Empty columns are dropped afterwards:
    try:
        df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
        df = df.dropna().reset_index(drop=True)
        # set header to first row
        header = df.iloc[0]
        df = df.iloc[1:, :]
        df.columns = header
        return df
    except urllib.error.HTTPError:
        pass


def get_pv_gis_data(config) -> pd.DataFrame:
    result_dict = {
        "region": scalar2array(config['region_name']),
        "year": config["start_year"],
        "id_hour": np.arange(1, 8761),
    }
    pv_generation_dict = get_pv_generation(config)
    temperature_dict = get_temperature(config)
    radiation_dict = get_radiation(config)
    try:
        assert pv_generation_dict["pv_generation"].sum() != 0
        assert temperature_dict["temperature"].sum() != 0
        assert radiation_dict["radiation_south"].sum() != 0
        assert radiation_dict["radiation_east"].sum() != 0
        assert radiation_dict["radiation_west"].sum() != 0
        assert radiation_dict["radiation_north"].sum() != 0
        result_dict.update(pv_generation_dict)
        result_dict.update(temperature_dict)
        result_dict.update(radiation_dict)
        result_df = pd.DataFrame.from_dict(result_dict)
        return result_df
    except Exception as e:
        print(f"At least one pv_gis source of Region {config['region_name']} includes all zeros.")


def save_pv_gis_data(config, path):
    df = get_pv_gis_data(config)
    df.to_csv(path, sep=";", index=False)



if __name__ == "__main__":
    config = {
        'region_name': "Vienna",
        "start_year": 2019,
        "end_year": 2019,
        "lat": 48.193,
        "lon": 16.378,
        "pv_peak_power": 1,
        "pv_calculation" : 1,
        "peak_power" : 1,
        "pv_loss" : 14,
        "pv_tech" : "crystSi",
        "tracking_type" : 0,
        "angle" : 90,
        "optimal_inclination" : 1,
        "optimal_angle" : 1,
    }
    save_pv_gis_data(config, Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\Sonstiges") / "pv_gis_weather_vienna.csv")



