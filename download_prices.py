import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient, Area

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
__country = "DE"

electricity_price_config = {
    # variable price
    "api_key": 'c06ee579-f827-486d-bc1f-8fa0d7ccd3da',
    "start": f"{__year}0101",
    "end": f"{__year + 1}0101",
    "country_code": __country,
    "grid_fee": 30,  # ct/kWh
}


def get_entsoe_prices(api_key: str,
                      start_time: str,
                      end_time: str,
                      country_code: str,
                      grid_fee: float) -> np.array:
    # %% parameter definitions

    # ToDo missing for all countries
    if country_code == "DE":
        country = "DE_LU"
    else:
        country = country_code

    client = EntsoePandasClient(api_key=api_key)

    start = pd.Timestamp(electricity_price_config["start"], tz='CET')
    end = pd.Timestamp(electricity_price_config["end"], tz='CET')

    # Get day-ahead prices from ENTSO-E Transparency
    print('Prices in zone ' + country)
    DA_prices = client.query_day_ahead_prices(country, start=start, end=end)
    prices = pd.DataFrame(DA_prices).reset_index(drop=True).to_numpy() / 10 / 1_000  # €/MWh in ct/kWh & ct/kWh in ct/Wh
    # add grid fees:
    prices_total = prices + grid_fee / 1_000  # also in ct/Wh
    return prices_total


def create_avg_price_from_variable_price(var_price: np.array) -> np.array:
    avg = var_price.mean()
    flat_price = np.ones(shape=var_price.shape) * avg
    return flat_price



def main():
    entsoe_price = get_entsoe_prices(api_key=electricity_price_config["api_key"],
                                     start_time=electricity_price_config["start"],
                                     end_time=electricity_price_config["end"],
                                     country_code=electricity_price_config["country_code"],
                                     grid_fee=electricity_price_config["grid_fee"])
    # drop the last hour because its already the first hour of the new year
    entsoe_price = entsoe_price[:-1]

    flat_price = create_avg_price_from_variable_price(entsoe_price)

    # save prices to excel
    df = pd.DataFrame(columns=[f"entsoe {__year}", "flat price"])
    df.loc[:, f"entsoe {__year}"] = entsoe_price.flatten()
    df.loc[:, "flat price"] = flat_price.flatten()
    df.to_csv(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\data\input_operation\prices_D5.2.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
