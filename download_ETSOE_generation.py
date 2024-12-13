import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient, Area
from pathlib import Path
import multiprocessing
from typing import Union


def get_entsoe_generation_per_type(
        country_code: str,
        year: int,

):
    
    pass


def get_entsoe_load(country_code: str,
                    year: int) -> pd.DataFrame:
    print(f'Load in {country_code} {year}')
    client = EntsoePandasClient(api_key='d90ceefb-e4d3-4666-8fdf-e53bd46d996a')
    start = pd.Timestamp(f"{year}0101", tz='CET')
    end = pd.Timestamp(f"{year + 1}0101", tz='CET')
    # Get day-ahead prices from ENTSO-E Transparency
    try:
        electricity_load = client.query_load(country_code=country_code,
                                                  start=start,
                                                  end=end,
                                                  )  # 
        
        if electricity_load.index.nlevels > 1:
            df = electricity_load.loc[
                    :, ['Aggregated' in name for name in electricity_load.columns.get_level_values(1)]
                    ]
        else:
            df = electricity_load

        total_consumption = df.sum(axis=1)
        # to make sure the consumption is hourly
        total_consumption = total_consumption.resample('1h').sum()
        # check the length:
        if len(total_consumption) != 8760:
            # if its a leap year:
            if len(total_consumption) == 8784:
                # drop the day of the 29th february:
                total_consumption = total_consumption[
                    ~((total_consumption.index.month == 2) & (total_consumption.index.day == 29))
                ]
            else:
                # check if first or last hour is missing:
                first = pd.Timestamp(f'{year}-01-01')
                total_consumption.index = total_consumption.index.tz_localize(None)
                if total_consumption.index[0] != first:
                    first_value = total_consumption[0]
                    entsoe_price = pd.concat([pd.DataFrame(index=[first], data=[first_value], columns=[0]), total_consumption])
                # check if last is missing
                last = pd.Timestamp(f'{year}-12-31 23:00:00')
                if total_consumption.index[-1] != last:
                    last_value = total_consumption[-1]
                    total_consumption = pd.concat([total_consumption, pd.DataFrame(index=[last], data=[last_value], columns=[0])])
        return total_consumption
    except Exception as e:
        print(f"{country_code} {year} no entsoe-data available.")
        print(f"{e}")
        return pd.DataFrame()

def get_entsoe_generation(country_code: str,
                          year: int) -> Union[pd.Series, None]:
    # %% parameter definitions
    print(f'Generation in {country_code} {year}')
    client = EntsoePandasClient(api_key='f09ebb03-5ae0-4b72-bb02-dd63a5817921')
    start = pd.Timestamp(f"{year}0101", tz='CET')
    end = pd.Timestamp(f"{year + 1}0101", tz='CET')
    # Get day-ahead prices from ENTSO-E Transparency
    try:
        electricity_production = client.query_generation(country_code=country_code,
                                                  start=start,
                                                  end=end,
                                                  resolution="60T")  # MW
        # drop consumption columns
        # check how many layers are in the multi-index:
        if electricity_production.index.nlevels > 1:
            df = electricity_production.loc[
                 :, ['Aggregated' in name for name in electricity_production.columns.get_level_values(1)]
                 ]
        else:
            df = electricity_production
        total_consumption = df.sum(axis=1)
        # to make sure the consumption is hourly
        total_consumption = total_consumption.resample('1h').sum()
        # check the length:
        if len(total_consumption) != 8760:
            # if its a leap year:
            if len(total_consumption) == 8784:
                # drop the day of the 29th february:
                total_consumption = total_consumption[
                    ~((total_consumption.index.month == 2) & (total_consumption.index.day == 29))
                ]
            else:
                # check if first or last hour is missing:
                first = pd.Timestamp(f'{year}-01-01')
                total_consumption.index = total_consumption.index.tz_localize(None)
                if total_consumption.index[0] != first:
                    first_value = total_consumption[0]
                    entsoe_price = pd.concat([pd.DataFrame(index=[first], data=[first_value], columns=[0]), total_consumption])
                # check if last is missing
                last = pd.Timestamp(f'{year}-12-31 23:00:00')
                if total_consumption.index[-1] != last:
                    last_value = total_consumption[-1]
                    total_consumption = pd.concat([total_consumption, pd.DataFrame(index=[last], data=[last_value], columns=[0])])
        return total_consumption
    except Exception as e:
        print(f"{country_code} {year} no entsoe-data available.")
        print(f"{e}")
        return pd.DataFrame()


def dowload_generation(country: str, year: int, path: Path):
    # check if file already exists:
    file_exists = False
    for file in path.iterdir():
        if str(year) in file.name and country in file.name:
            file_exists = True
    # file exists will be true if there is a file for the country for the same year
    if not file_exists:
        intermediate_df = pd.DataFrame()
        entsoe_generation = get_entsoe_generation(
            country_code=country,
            year=year
        )
        # add country name to consumption profile for later
        if not entsoe_generation.empty:
            if not len(entsoe_generation) != 8760:
                intermediate_df.loc[:, country] = entsoe_generation.to_numpy()  # MWh
                timestamp = entsoe_generation.index

                filename = f"ENTSOE_generation_for_{country}_{year}_MWh_intermediate.csv"
                intermediate_df.index = timestamp
                # save prices to excel
                intermediate_df.to_csv(path / Path(filename), sep=";", index=False)


def download_load(country: str, year: int, path: Path):
       # check if file already exists:
    file_exists = False
    for file in path.iterdir():
        if str(year) in file.name and country in file.name:
            file_exists = True
    # file exists will be true if there is a file for the country for the same year
    if not file_exists:
        intermediate_df = pd.DataFrame()
        entsoe_load = get_entsoe_load(
            country_code=country,
            year=year
        )
        if not entsoe_load.empty:
            if not len(entsoe_load) != 8760:
                # add country name to consumption profile for later
                intermediate_df.loc[:, country] = entsoe_load.to_numpy()  # MWh
                timestamp = entsoe_load.index

                filename = f"ENTSOE_load_for_{country}_{year}_MWh_intermediate.csv"
                intermediate_df.index = timestamp
                # save prices to excel
                intermediate_df.to_csv(path / Path(filename), sep=";", index=False) 
            else:
                print(f"no entsoe data for {country} {year}") 


def compress_to_single_files(year: int, path: Path):
    """
    merge the intermediate csv files together to large files for each year containing all countries
    :return:
    """
    print(f"creating consumption file for {year}")
    df_generation = pd.DataFrame()
    df_load = pd.DataFrame()

    for file in path.iterdir():
        if str(year) in file.name and "generation" in file.name.lower():
            read = pd.read_csv(file, sep=";")
            df_generation = pd.concat([df_generation, read], axis=1)
            file.unlink()
        elif str(year) in file.name and "load" in file.name.lower():
            read = pd.read_csv(file, sep=";")
            df_load = pd.concat([df_load, read], axis=1)
            file.unlink()

    if not df_generation.empty:
        df_generation.to_csv(path / f"ENTSOE_generation_MWh_{year}.csv", sep=";", index=False)
        print(f"ENTSOE_generation_MWh_{year}.csv saved")
    if not df_load.empty:
        df_load.to_csv(path / f"ENTSOE_load_MWh_{year}.csv", sep=";", index=False)
        print(f"ENTSOE_load_MWH_{year}.csv saved")


if __name__ == "__main__":
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    country_codes = [
        "AT",
        "DE",
        "AT",
        "LU",
        "BE",
        "BG",
        "HR",
        "CY",
        "CZ",
        "DK",
        "EE",
        "FI",
        "FR",
        "DE",
        "GR",
        "HU",
        "IE",
        "IT",
        "LV",
        "LT",
        "LU",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SK",
        "SI",
        "ES",
        "SE",
    ]
    path = Path(__file__).parent / r"ENTSOE Generation"
    # for country in country_codes:
    #     for year in years:
    #         download_load(country=country, year=year, path=path)

    input_list = [(country, y, path) for country in country_codes for y in years]
    with multiprocessing.Pool(6) as pool:
        pool.starmap(download_load, input_list)
        pool.starmap(dowload_generation, input_list)


    # for year in years:
    #     compress_to_single_files(year, path)

