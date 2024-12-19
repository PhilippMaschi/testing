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
import logging



    # Ensure the log directory exists
log_file_path = Path(__file__).parent / "logfile.log"
log_level = logging.INFO
# Configure the logger
logging.basicConfig(
    level=log_level,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Write logs to the file
        logging.StreamHandler()             # Optionally output logs to the console
    ]
)
LOGGER = logging.getLogger(__name__)


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


def get_season(day_of_year):
    """
    Determine the season based on the day of the year.
    
    Args:
        day_of_year (int): The day of the year (1-365 or 1-366 for leap years).
    
    Returns:
        str: The season ("Winter", "Spring", "Summer", "Autumn").
    """
    if day_of_year < 0 or day_of_year > 365:
        raise ValueError("day_of_year must be between 1 and 366")
    
    if 0 <= day_of_year <= 78 or day_of_year >= 356:  # Approx. Dec 21 - Mar 20
        return "Winter"
    elif 81 <= day_of_year <= 172:  # Approx. Mar 21 - Jun 20
        return "Spring"
    elif 173 <= day_of_year <= 265:  # Approx. Jun 21 - Sep 22
        return "Summer"
    elif 266 <= day_of_year <= 355:  # Approx. Sep 23 - Dec 20
        return "Autumn"


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
    LOGGER.info("saved plotly figure national_peak_day_loads.html")


def create_sankey_diagram(df):
    # sankey diagramm um zu sehen wie sich die Peaks ändern:
    flows = df.groupby(["all HEMS", "no HEMS"]).size().reset_index(name="count")
    source_column = "no HEMS"
    target_column = "all HEMS"
    value_column = "count"
    labels = list(pd.concat([flows[source_column], flows[target_column]]).unique())
    
    season_colors = {
        "Spring": "rgba(102, 194, 165, 0.8)",  # light green
        "Summer": "rgba(252, 141, 98, 0.8)",   # orange
        "Autumn": "rgba(141, 160, 203, 0.8)",  # light blue
        "Winter": "rgba(231, 138, 195, 0.8)"   # pink
    }
    node_colors = (
        [season_colors["Spring"]]  +
        [season_colors["Summer"]]  +
        [season_colors["Autumn"]]  +
        [season_colors["Winter"]]  +
        [season_colors["Spring"]]  +
        [season_colors["Summer"]]  +
        [season_colors["Autumn"]]  +
        [season_colors["Winter"]]   
    )
    link_colors = flows[source_column].map(lambda x: season_colors[x]).tolist()

    # Map source and target to indices
    source_labels = ["Spring (no HEMS)", "Summer (no HEMS)", "Autumn (no HEMS)", "Winter (no HEMS)"]
    target_labels = ["Spring (all HEMS)", "Summer (all HEMS)", "Autumn (all HEMS)", "Winter (all HEMS)"]

    # Combine source and target labels into a single list
    labels = source_labels + target_labels
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Map source and target to their respective indices
    flows = df.groupby(["all HEMS", "no HEMS"]).size().reset_index(name="count")
    source_column = "no HEMS"
    target_column = "all HEMS"
    value_column = "count"

    # Map "all HEMS" (source) to source_labels indices
    sources = flows[source_column].map(lambda x: label_to_index[f"{x} (no HEMS)"])
    # Map "no HEMS" (target) to target_labels indices
    targets = flows[target_column].map(lambda x: label_to_index[f"{x} (all HEMS)"])
    values = flows[value_column]

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])

    # Save the Sankey diagram
    saving_path = Path(__file__).parent / "figures"
    saving_path.mkdir(exist_ok=True, parents=True)
    fig.write_html(saving_path / "sankey_seasonal_peaks.html")
    fig.write_image(saving_path / "sankey_seasonal_peaks.svg")

def plot_frequency_of_peaks_in_seasons(peak_df: pd.DataFrame):
    plot_df = peak_df.drop(columns=["change in peak", "change in peak relative"]).reset_index().melt(id_vars=["country", "year"], value_name="season")
    count_df = plot_df.groupby(["season", "variable", "year"]).size().reset_index(name="count").copy()

    g = sns.FacetGrid(count_df, col="year", col_wrap=3, height=4, sharey=True)
    g.map_dataframe(
        sns.barplot,
        x="season",
        y="count",
        hue="variable",
        order=["Winter", "Spring", "Summer", "Autumn"],  # Ensure consistent order if needed
    )
    g.add_legend()
    g.set_axis_labels("Season", "number of peaks occuring in each season")
    g.tight_layout()
    plt.show()




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
                peak_day_opt_season = get_season(peak_day_opt)
                peak_day_ref_season = get_season(peak_day_ref)
                if peak_day_opt != peak_day_ref:
                    LOGGER.info(f"peak demand day has been shifted in {country} {year}")
                if peak_day_opt_season != peak_day_ref_season:
                    LOGGER.warning(f"peak demand day has been shifted to another season {country} {year}")


                # cut the peak day profile for the plot:
                ref_peak_day = national_demand_ref.iloc[peak_day_ref*24: peak_day_ref*24+24]
                opt_peak_day = national_demand_opt.iloc[peak_day_ref*24: peak_day_ref*24+24]

                day_df.loc[:, f"ref_{country}_{year}"] = ref_peak_day.reset_index(drop=True)
                day_df.loc[:, f"opt_{country}_{year}"] = opt_peak_day.reset_index(drop=True)

                peak_df.loc[(country, year), f"change in peak"] = peak_diff
                peak_df.loc[(country, year), f"change in peak relative"] = peak_diff / peak_ref * 100
                peak_df.loc[(country, year), f"all HEMS"] = peak_day_opt_season
                peak_df.loc[(country, year), f"no HEMS"] = peak_day_ref_season


    plot_national_peaks(peak_df=peak_df)
    create_sankey_diagram(df=peak_df)
    plot_frequency_of_peaks_in_seasons(peak_df=peak_df)
    plot_national_peak_days(day_df=day_df)


def main(percentage_dhw_tanks, percentage_buffer_tanks, scenario: str):

    df = pd.read_parquet(Path(__file__).parent / f"EU27_loads_dhw-{percentage_dhw_tanks}_buffer-{percentage_buffer_tanks}.parquet.gzip")
    national_demand = Cp.get_national_demand_profiles()
    national_demand = national_demand.loc[(national_demand["scenario"]==scenario) | (national_demand["scenario"]=="baseyear"), :]
    
    plot_flexible_storage_efficiency(loads=df)
    plot_flexibility_factor(loads=df, national=national_demand, scenario=scenario)

    # TODO rerun with positive prices! 20 cent grid fee
    show_day_with_peak_deamand(profiles=df, scenario=scenario, national=national_demand)


if __name__ == "__main__":

    main(
        percentage_dhw_tanks=0.1,
        percentage_buffer_tanks=0.1,
        scenario="shiny happy"
    )
