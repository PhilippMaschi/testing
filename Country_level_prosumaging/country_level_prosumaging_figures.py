import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlalchemy
from typing import List
import matplotlib.gridspec as gridspec
import Country_level_prosumager as Cp



def plot_supply_and_demand_matching_over_price(price: np.array, s_d_match: pd.DataFrame, country: str, year: str):
    bin_size = 20
    df = s_d_match.copy()
    df["price"] = df["price"] * 1000  # cent/kWh

    df["price_bins"] = pd.cut(df['price'], bins=bin_size)
    df['price_bins_str'] = df['price_bins'].apply(lambda x: f"[{x.left:.1f}, {x.right:.1f}]")

    for price_bin, group in df.groupby("price_bins"):
        df.loc[group.index, "price frequency"] = len(group)
        df.loc[group.index, "cumulative energy"] = group["change in electricity demand"].sum()

    df["cumulative energy"] = df["cumulative energy"] / 1_000  # GWh
      

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], figure=fig)  # Two rows, stacked vertically with height ratios

    ax2 = fig.add_subplot(gs[0])
    sns.barplot(
        data=df,
        x="price_bins_str",
        y="price frequency",
        ax=ax2,
        color="green",
        alpha=0.5
    )
    ax2.set_ylabel("Price frequency")
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-tick labels on top plot
    ax2.set_xlabel("")

    ax = fig.add_subplot(gs[1])
    sns.lineplot(
        data=df,
        x="price_bins_str",
        y="cumulative energy",
        ax=ax,
    )
    ax.set_ylabel("cumulative shifted electricity (GWh)")
    ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color="black")
    ax.set_xlabel("Wholesale electricity price (cent/kWh)")
    plt.xticks(rotation=45)
    ax.set_xlim(left=0, right=bin_size)

    # Save the figure
    output_path = Path(__file__).parent / "single_country_plots" / f"Energy_over_price_{country}_{year}.png"
    plt.tight_layout()
    plt.savefig(output_path)

    # Close the plot to free up memory
    plt.close()


def plot_flexibility_factor(loads: pd.DataFrame, national: pd.DataFrame, scenario: str):
    plot_df = pd.DataFrame()
    i = 0
    for country in Cp.EUROPEAN_COUNTRIES.keys():
        for year in [2020, 2030, 2040, 2050]:
            if year == 2020 and (country == "CYP" or country=="MLT" or country=="NLD"):
                continue
            else:
                demand = national.loc[(national["country"]==country) & (national["year"]==year), "generation"].reset_index(drop=True)
                peak_demand_hour = demand.idxmax()
                min_demand_hour = demand.idxmin()
                ref_col = f"{country}_{year}_ref_load_MW"
                opt_col = f"{country}_{year}_opt_load_MW"
                # peak demand
                plot_df.loc[i, "country"] = country
                plot_df.loc[i, "year"] = year
                plot_df.loc[i, f"peak_demand_load_factor"] = (loads.loc[peak_demand_hour, ref_col] - loads.loc[peak_demand_hour, opt_col]) / demand[peak_demand_hour] * 100
                plot_df.loc[i, "min_demand_load_factor"] = (loads.loc[min_demand_hour, ref_col] - loads.loc[min_demand_hour, opt_col]) / demand[min_demand_hour] * 100

                # peak price
                peak_price_hour = loads[f"{country}_{year}_price"].idxmax()
                min_price_hour = loads[f"{country}_{year}_price"].idxmin()

                plot_df.loc[i, f"peak_price_load_factor"] = (loads.loc[peak_price_hour, ref_col] - loads.loc[peak_price_hour, opt_col]) / demand[peak_price_hour] * 100
                plot_df.loc[i, f"min_price_load_factor"] = (loads.loc[min_price_hour, ref_col] - loads.loc[min_price_hour, opt_col]) / demand[min_price_hour] * 100

                i += 1


    sns.barplot(
        data=plot_df,
        x="country",
        y="peak_demand_load_factor",
        hue="year"
    )
    plt.suptitle("peak demand load factor")
    plt.xticks(rotation=90)
    plt.show()

    sns.barplot(
        data=plot_df,
        x="country",
        y="min_demand_load_factor",
        hue="year"
    )
    plt.suptitle("min demand load factor")
    plt.xticks(rotation=90)
    plt.show()

    sns.barplot(
        data=plot_df,
        x="country",
        y="peak_price_load_factor",
        hue="year"
    )
    plt.suptitle("peak price load factor")
    plt.xticks(rotation=90)
    plt.show()

    sns.barplot(
        data=plot_df,
        x="country",
        y="min_price_load_factor",
        hue="year"
    )
    plt.xticks(rotation=90)
    plt.suptitle("min price load factor")
    plt.show()

def main(percentage_dhw_tanks, percentage_buffer_tanks, scenario: str):

    df = pd.read_parquet(Path(__file__).parent / f"EU27_loads_dhw-{percentage_dhw_tanks}_buffer-{percentage_buffer_tanks}.parquet.gzip")
    national_demand = Cp.get_national_demand_profiles()
    national_demand = national_demand.loc[(national_demand["scenario"]==scenario) | (national_demand["scenario"]=="baseyear"), :]
    
    plot_flexibility_factor(loads=df, national=national_demand, scenario=scenario)

    # calculate the factors:

    # flexiblity_factor_hourly[f"{country}_{year}"] = flexibility_factor_hourly(
    #     consumer_profile=consumers_MW,
    #     prosumager_profile=prosumer_MW,
    #     national_demand_profile=np.array(national_demand[COUNTRY_CODES[country]])
    # )

    # s_and_d_matching_dict[f"{country}_{year}"] = supply_and_demand_matching(
    #     price_profile=price,
    #     consumer_profile=consumers_MW,
    #     prosumager_profile=prosumer_MW
    # )

    # storage_efficiency[f"{country}_{year}"] = flexible_storage_efficiency(
    #     consumer_profile=consumers_MW,
    #     prosumager_profile=prosumer_MW
    # )

    # plot_supply_and_demand_matching_over_price(price=price,
    #                                             s_d_match=s_and_d_matching_dict[f"{country}_{year}"],
    #                                             country=country,
    #                                             year=year)


if __name__ == "__main__":

    main(
        percentage_dhw_tanks=0.1,
        percentage_buffer_tanks=0.1,
        scenario="shiny happy"
    )
