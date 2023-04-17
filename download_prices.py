import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient, Area
from pathlib import Path
import multiprocessing

"""
Query data from ENTSO-E transparency portal
------------------------------------------------------------------------------
DISCLAIMER: 
You may use the Code for any private or commercial purpose. However, you may not sell, 
sub-license, rent, lease, lend, assign or otherwise transfer, duplicate or otherwise 
reproduce, directly or indirectly, the Code in whole or in part. 

You acknowledge that the Code is provided “AS IS” and thesmartinsights.com expressly 
disclaims all warranties and conditions including, but not limited to, any implied 
warranties for suitability of the Code for a particular purpose, or any form of warranty 
that operation of the Code will be error-free.

You acknowledge that in no event shall thesmartinsights.com or any of its affiliates be 
liable for any damages arising out of the use of the Code or otherwise in connection with 
this agreement, including, without limitation, any direct, indirect special, incidental 
or consequential damages, whether any claim for such recovery is based on theories of 
contract, negligence, and even if thesmartinsights.com has knowledge of the possibility 
of potential loss or damage.
------------------------------------------------------------------------------- 
"""

pd.options.display.max_columns = None
__year = 2019

electricity_price_config = {
    # variable price
    "api_key": 'c06ee579-f827-486d-bc1f-8fa0d7ccd3da',
    "start": f"{__year}0101",
    "end": f"{__year + 1}0101",
    "grid_fee": 20,  # ct/kWh
}


def get_entsoe_prices(api_key: str,
                      country_code: str,
                      year: int) -> pd.Series:
    # %% parameter definitions
    print('Prices in zone ' + country_code)
    client = EntsoePandasClient(api_key=api_key)
    start = pd.Timestamp(f"{year}0101", tz='CET')
    end = pd.Timestamp(f"{year + 1}0101", tz='CET')
    try:
        # Get day-ahead prices from ENTSO-E Transparency
        DA_prices = client.query_day_ahead_prices(country_code=country_code,
                                                  start=start,
                                                  end=end,
                                                  resolution="60T")
        # check if price is hourly our subhourly:
        if len(DA_prices) > 9000:
            print(f"prices for {country_code} are given subhourly intervals.")
        # drop the last hour because its already the first hour of the new year
        entsoe_price = DA_prices[DA_prices.index.year == year]
        if len(entsoe_price) != 8760:
            # just resample..
            entsoe_price = entsoe_price.resample('1H').mean()
            #check again
        if len(entsoe_price) != 8760:
            # check if first or last hour is missing:
            first = pd.Timestamp(f'{year}-01-01')
            entsoe_price.index = entsoe_price.index.tz_localize(None)
            if entsoe_price.index[0] != first:
                price = entsoe_price[0]
                entsoe_price = pd.concat([pd.DataFrame(index=[first], data=[price], columns=[0]), entsoe_price])
            # check if last is missing
            last = pd.Timestamp(f'{year}-12-31 23:00:00')
            if entsoe_price.index[-1] != last:
                price = entsoe_price[-1]
                entsoe_price = pd.concat([entsoe_price, pd.DataFrame(index=[last], data=[price], columns=[0])])
        return entsoe_price
    except Exception as e:
        print(f"{country_code} no entsoe-data available.")
        print(f"{e}")
        return None


def create_avg_price_from_variable_price(var_price: np.array) -> np.array:
    avg = var_price.mean()
    flat_price = np.ones(shape=var_price.shape) * avg
    return flat_price


def main(path, year):
    country_codes = [
        "AT",
        "DE_AT_LU",
        "BE",
        "BG",
        "HR",
        "CY",  # no data available on ENTSO-E
        "CZ",
        "DK_1",
        "DK_2",
        "EE",
        "FI",
        "FR",
        "DE",
        "GR",
        "HU",
        "IE_SEM",
        "IT_CALA",
        "IT_CALA",
        "IT_BRNN",
        "IT_CNOR",
        "IT_CSUD",
        "IT_FOGN",
        "IT_GR",
        "IT_MALTA",
        "IT_NORD",
        "IT_NORD_AT",
        "IT_NORD_CH",
        "IT_NORD_FR",
        "IT_NORD_SI",
        "IT_PRGP",
        "IT_ROSN",   # has one missing value for 2020
        "IT_SARD",
        "IT_SICI",
        "IT_SUD",
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
        "SE_1",
        "SE_2",
        "SE_3",
        "SE_4",
    ]
    prices_df = pd.DataFrame()
    for country in country_codes:
        if country == "DE":
            country = "DE_LU"
        elif country == "LU":
            continue

        entsoe_price = get_entsoe_prices(api_key=electricity_price_config["api_key"],
                                         country_code=country,
                                         year=year)
        if type(entsoe_price) == pd.DataFrame:
            entsoe_price = entsoe_price.squeeze()
        if type(entsoe_price) == pd.Series:
            if len(entsoe_price) == 0:
                continue
            prices_df.loc[:, country] = entsoe_price.to_numpy() / 10  # €/MWh in ct/kWh
            timestamp = entsoe_price.index

    filename = f"ENTSOE_prices_for_{year}_in_ct_per_kWh.csv"
    prices_df.index = timestamp
    # save prices to excel
    prices_df.to_csv(path / Path(filename), sep=";", index=False)


if __name__ == "__main__":
    path_to_save_csv = Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\ENTSOE Prices")
    years = [2018, 2019, 2020, 2021]
    input_list = [(path_to_save_csv, y) for y in years]

    # ---------------------------
    # year = 2019
    # main(path_to_save_csv, year)
    # ---------------------------
    cores = int(multiprocessing.cpu_count() / 2)
    with multiprocessing.Pool(cores) as pool:
        pool.starmap(main, input_list)
