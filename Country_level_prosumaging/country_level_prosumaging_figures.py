import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlalchemy
from typing import List
import matplotlib.gridspec as gridspec
import Country_level_prosumager as Cp
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from itertools import cycle

def plot_supply_and_demand_matching_over_price(loads: pd.DataFrame):
    bin_size = 20
    df = loads.copy()

    plot_df = pd.DataFrame()

    for country in Cp.EUROPEAN_COUNTRIES.keys():
        for year in [2020, 2030, 2040, 2050]:
            if year == 2020 and (country == "CYP" or country=="MLT" or country=="NLD"):
                continue
            else:
                ref_col = f"{country}_{year}_ref_load_MW"
                opt_col = f"{country}_{year}_opt_load_MW"

                plot_df.loc[:, f"supply_demand_match"] = loads.loc[:, ref_col] - loads.loc[:, opt_col]
                plot_df.loc[:, "country"] = country
                plot_df.loc[:, "year"] = year
    

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




def plot_flexible_storage_efficiency(loads: pd.DataFrame):
    plot_df = pd.DataFrame()
    i = 0
    for country in Cp.EUROPEAN_COUNTRIES.keys():
        for year in [2020, 2030, 2040, 2050]:
            if year == 2020 and (country == "CYP" or country=="MLT" or country=="NLD"):
                continue
            else:
                
                consumer_profile = pd.concat([loads.loc[:3500, f"{country}_{year}_ref_load_MW"], loads.loc[6500:, f"{country}_{year}_ref_load_MW"]])
                prosumager_profile = pd.concat([loads.loc[:3500, f"{country}_{year}_opt_load_MW"], loads.loc[6500:, f"{country}_{year}_opt_load_MW"]])
                charging_energy = prosumager_profile - consumer_profile
                charging_energy[charging_energy < 0] = 0
                # eta from https://doi.org/10.1016/j.apenergy.2017.04.061
                eta = 1 - np.sum(prosumager_profile - consumer_profile) / np.sum(charging_energy)

                plot_df.loc[i, "country"] = country
                plot_df.loc[i, "year"] = year
                plot_df.loc[i, f"storage efficiency"] = eta

                i +=1
    
    sns.barplot(
        data=plot_df,
        x="country",
        y="storage efficiency",
        hue="year",
        palette=sns.color_palette()
    )
    plt.suptitle("Storage efficiency")
    plt.xticks(rotation=90)
    plt.show()
    plt.close()

    sp_opt = loads["DEU_2020_opt_load_MW"]
    sp_ref = loads["DEU_2020_ref_load_MW"]
    plt.plot(sp_opt, label="opt")
    plt.plot(sp_ref, label="ref", alpha=0.5)
    plt.legend()
    plt.show()
    print((sp_ref.sum()-sp_opt.sum())/sp_ref.sum())

def plot_flexibility_factor(loads: pd.DataFrame, national: pd.DataFrame, scenario: str):
    plot_df = pd.DataFrame()
    i = 0
    for country in Cp.EUROPEAN_COUNTRIES.keys():
        for year in [2020, 2030, 2040, 2050]:
            if year == 2020 and (country == "CYP" or country=="MLT" or country=="NLD"):
                continue
            else:
                if year == 2020:
                    scen = "baseyear"
                else:
                    scen = scenario
                demand = national.loc[(national["country"]==country) & (national["year"]==year) & (national["scenario"]==scen), "generation"].reset_index(drop=True)
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

def plot_national_peaks(peak_df: pd.DataFrame):
    plot_df = peak_df.reset_index()
    sns.barplot(
        data=plot_df,
        x="country",
        y="change in peak",
        hue="year",
    )
    plt.xticks(rotation=90)
    plt.ylabel("change in peak demand (MW)")
    plt.show()


    sns.barplot(
        data=plot_df,
        x="country",
        y="change in peak relative",
        hue="year",
    )
    plt.xticks(rotation=90)
    plt.ylabel("relative change in peak demand (%)")
    plt.show()


def plot_national_peak_days(day_df: pd.DataFrame):
    year_rows = {"2020": 1, "2030": 2, "2040": 3, "2050": 4}

    fig = make_subplots(
        rows=len(year_rows),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"Year {year}" for year in year_rows.keys()],
    )

    # Create the color map
    countries = day_df.columns.str.split("_").str[1].unique()
    palette = list(qualitative.Plotly)
    extended_palette = list(cycle(palette))[:len(countries)]  # Cycle through the palette if needed
    color_map = {country: color for country, color in zip(countries, extended_palette)}

    for column in day_df.columns:
        year = column.split("_")[-1]
        country = column.split("_")[1]
        typ = column.split("_")[0]
        if year == 2020 and (country == "CYP" or country=="MLT" or country=="NLD"):
            continue

        if typ == "ref":
            dash = "dot"
        else:
            dash = "solid"

        fig.add_trace(
            go.Scatter(
                x=day_df.index,
                y=day_df[column],
                mode="lines",
                name=f"{typ} {country} {year}",
                line=dict(dash=dash, color=color_map[country]),
                # legendgroup=country,  # Grouping traces by year
                xaxis=f"x{year_rows[year] + 1}",
                yaxis=f"y{year_rows[year] + 1}",

            ),
            row=year_rows[year],
            col=1,
        )

    # Add layout settings to separate rows for each year
    fig.update_layout(
        height=1200,  # Adjust height based on the number of subplots
        title="Interactive Plot by Year",
        xaxis_title="hours",
        yaxis_title="national demand on peak day (MW)",
    )
    
    # Save the plot as an HTML file
    saving_path = Path(__file__).parent / "figures"
    saving_path.mkdir(exist_ok=True, parents=True)
    fig.write_html(saving_path / f"national_peak_day_loads.html")









def show_day_with_peak_deamand(profiles: pd.DataFrame, scenario: str, national: pd.DataFrame):
    day_df = pd.DataFrame()
    peak_df = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["country", "year"]))
    for country in Cp.EUROPEAN_COUNTRIES.keys():
        for year in [2020, 2030, 2040, 2050]:
            if year == 2020 and (country == "CYP" or country=="MLT" or country=="NLD"):
                continue
            else:
                if year == 2020:
                    scen = "baseyear"
                else:
                    scen = scenario
                ref_col = f"{country}_{year}_ref_load_MW"
                opt_col = f"{country}_{year}_opt_load_MW"
                national_demand_ref = national.loc[(national["country"]==country) & (national["year"]==year) & (national["scenario"]==scen), "generation"].copy().reset_index(drop=True)
                national_demand_opt = national_demand_ref - profiles.loc[:, ref_col] + profiles.loc[:, opt_col]


                ref_minus_opt = national_demand_ref - national_demand_opt
                peak_demand_hour_opt = national_demand_opt.idxmax()
                peak_demand_hour_ref = national_demand_ref.idxmax()

                peak_opt = national_demand_opt.loc[peak_demand_hour_opt]
                peak_ref = national_demand_ref.loc[peak_demand_hour_ref]
                peak_diff = peak_opt - peak_ref

                peak_day_opt = int(peak_demand_hour_opt/24)
                peak_day_ref = int(peak_demand_hour_ref/24)
                if peak_day_opt != peak_day_ref:
                    print(f"peak demand day has been shifted in {country} {year}")

                # cut the peak day profile for the plot:
                ref_peak_day = national_demand_ref.iloc[peak_day_ref*24: peak_day_ref*24+24]
                opt_peak_day = national_demand_opt.iloc[peak_day_ref*24: peak_day_ref*24+24]

                day_df.loc[:, f"ref_{country}_{year}"] = ref_peak_day.reset_index(drop=True)
                day_df.loc[:, f"opt_{country}_{year}"] = opt_peak_day.reset_index(drop=True)

                peak_df.loc[(country, year), f"change in peak"] = peak_diff
                peak_df.loc[(country, year), f"change in peak relative"] = peak_diff / peak_ref * 100


    plot_national_peaks(peak_df=peak_df)
    plot_national_peak_days(day_df=day_df)



def main(percentage_dhw_tanks, percentage_buffer_tanks, scenario: str):

    df = pd.read_parquet(Path(__file__).parent / f"EU27_loads_dhw-{percentage_dhw_tanks}_buffer-{percentage_buffer_tanks}.parquet.gzip")
    national_demand = Cp.get_national_demand_profiles()
    national_demand = national_demand.loc[(national_demand["scenario"]==scenario) | (national_demand["scenario"]=="baseyear"), :]
    
    # plot_flexible_storage_efficiency(loads=df)
    # plot_flexibility_factor(loads=df, national=national_demand, scenario=scenario)

    # TODO show peak demand day profiles
    # TODO show total peaks for all countries and color them for seasonality somehow
    # TODO rerun with positive prices! 20 cent grid fee
    show_day_with_peak_deamand(profiles=df, scenario=scenario, national=national_demand)


if __name__ == "__main__":

    main(
        percentage_dhw_tanks=0.1,
        percentage_buffer_tanks=0.1,
        scenario="shiny happy"
    )
