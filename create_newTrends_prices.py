import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.offline



def main(years):
    long_df = pd.DataFrame()
    for year in years:
        path_to_csv = Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\ENTSOE Prices")

        path_to_file = path_to_csv / f"ENTSOE_prices_for_{year}_in_ct_per_kWh.csv"
        df = pd.read_csv(path_to_file, sep=";")

        # iterate through column names. If a column name has "_" in it, more than one price zone exists. In that case compare
        # the price zones to each other, rank them after the standard deviation and take the middle one:
        final_prices = pd.DataFrame()
        for column_name in df.columns:
            if "_" in column_name:
                country = column_name.split("_")[0]
                prices_in_country = df.filter(regex=country)

                std = prices_in_country.std()
                final_price_zone = std.sort_values().index[len(std) // 2]
                final_prices = pd.concat([final_prices, df[final_price_zone]], axis=1)
            else:
                final_prices[column_name] = df[column_name]

        # drop the identical columns
        final_df = final_prices.loc[:, ~final_prices.columns.duplicated()].copy()
        # add year to final price
        final_df["year"] = str(year)

        # prepare plot to show standard deviation:
        plot_df = final_df.melt(id_vars="year", var_name="country", value_name="electricity price")
        long_df = pd.concat([long_df, plot_df], axis=0)


    fig = px.box(
        long_df,
        x="country",
        y="electricity price",
        color="country",
        facet_row="year",
    )
    fig.update_layout(showlegend=False, title=f"electricity wholesale prices")
    fig.update_yaxes(matches=None)
    plotly.offline.plot(figure_or_data=fig, filename=r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\ENTSOE Prices\prices.html")
    fig.show()



if __name__ == "__main__":
    year = [2019, 2020, 2021, 2022]
    main(year)