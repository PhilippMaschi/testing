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


def plot_PV_self_consumption(loads: pd.DataFrame):

    grouped = loads.groupby(["country", "year", "ID_EnergyPrice"])
    self_consumption_ref = grouped["ref_PV2Load_MW"].sum() / grouped["ref_PhotovoltaicProfile_MW"].sum() 
    self_consumption_opt = grouped["opt_PV2Load_MW"].sum() / grouped["opt_PhotovoltaicProfile_MW"].sum() 

    self_consumption_ref = self_consumption_ref.reset_index()
    self_consumption_opt = self_consumption_opt.reset_index()
    self_consumption_ref["type"] = "reference"
    self_consumption_opt["type"] = "HEMS"

    self_consumption_ref.rename(columns={0: "PV self consumption"}, inplace=True)
    # drop the ID_EnergyPrice = 2 of ref because its the same

    self_consumption_opt.rename(columns={0: "PV self consumption"}, inplace=True)
    # merge the type and energy price columns:
    
    plot_df = pd.concat([self_consumption_ref, self_consumption_opt] , axis=0).reset_index(drop=True)
    x_order = plot_df.groupby("country")["PV self consumption"].mean().sort_values().index

    g = sns.FacetGrid(plot_df, col="year", col_wrap=2, height=5, aspect=1.5)

    # Map the barplot to each facet
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="PV self consumption",
        hue="type",
        dodge=True,  # Ensures bars are grouped within x-axis categories
        hue_order=None, 
        order=x_order
    )

    # Adjust plot aesthetics
    g.set_axis_labels("Country", "PV self consumption")
    g.add_legend()
    g.set_titles("Year {col_name}")

    # Show the plot
    plt.tight_layout()
    plt.show()

    # average PV self consumption over Europe:
    eu_df = plot_df.groupby(["year", "type"])["PV self consumption"].mean().reset_index()
    sns.barplot(
        data=eu_df,
        x="PV self consumption",
        y="year",
        hue="type"
    )
    plt.suptitle("average PV self consumption over all countries")
    plt.tight_layout()
    plt.show()



def plot_flexible_storage_efficiency(loads: pd.DataFrame):

    loads["charging_energy"] = loads["opt_grid_demand_stock_MW"] - loads["ref_grid_demand_stock_MW"]
    loads.loc[loads["charging_energy"] < 0, "charging_energy"] = 0

    grouped = loads.groupby(["country", "year", "ID_EnergyPrice"])
    storage_efficiency = 1 - (grouped["opt_grid_demand_stock_MW"].sum() - grouped["ref_grid_demand_stock_MW"].sum()) / grouped["charging_energy"].sum()
    storage_efficiency = storage_efficiency.reset_index()
    storage_efficiency.rename(columns={0: "storage efficiency"}, inplace=True)

    sns.barplot(
        data=storage_efficiency,
        x="country",
        y="storage efficiency",
        hue="year",
        palette=sns.color_palette(),
        

    )
    plt.suptitle("Storage efficiency")
    plt.xticks(rotation=90)
    plt.show()
    plt.close()

    # calculate the storage efficiency without PV:
    loads["ref_Load_MW"] = loads["ref_grid_demand_stock_MW"] + loads["ref_PV2Load_MW"]
    loads["opt_Load_MW"] = loads["opt_grid_demand_stock_MW"] + loads["opt_PV2Load_MW"]
    loads["charging_energy_pv_cleaned"] = loads["opt_Load_MW"] - loads["ref_Load_MW"]
    loads.loc[loads["charging_energy_pv_cleaned"] < 0, "charging_energy_pv_cleaned"] = 0

    grouped = loads.groupby(["country", "year", "ID_EnergyPrice"])
    storage_efficiency = 1 - (grouped["opt_Load_MW"].sum() - grouped["ref_Load_MW"].sum()) / grouped["charging_energy_pv_cleaned"].sum()
    storage_efficiency = storage_efficiency.reset_index()
    storage_efficiency.rename(columns={0: "storage efficiency"}, inplace=True)

    sns.barplot(
        data=storage_efficiency,
        x="country",
        y="storage efficiency",
        hue="year",
        palette=sns.color_palette(),
        

    )
    plt.suptitle("Storage efficiency including PV")
    plt.xticks(rotation=90)
    plt.show()
    plt.close()

def show_average_day_profile(loads: pd.DataFrame):
    # plot once the load and once the grid

    loads["ref_Load_MW"] = loads["ref_grid_demand_stock_MW"] + loads["ref_PV2Load_MW"]
    loads["opt_Load_MW"] = loads["opt_grid_demand_stock_MW"] + loads["opt_PV2Load_MW"]
    loads["day_hour"] = (loads["Hour"]-1) % 24

    df = loads.groupby(["year", "ID_EnergyPrice", "day_hour"])[["ref_grid_demand_stock_MW", "opt_grid_demand_stock_MW", "ref_Load_MW", "opt_Load_MW"]].mean().reset_index()
    plot_df = df.rename(columns={
        "ref_grid_demand_stock_MW": "grid demand reference",
        "opt_grid_demand_stock_MW": "grid demand HEMS",
        "ref_Load_MW": "Load reference",
        "opt_Load_MW": "Load HEMS"
    }).melt(id_vars=["year", "ID_EnergyPrice", "day_hour"], value_name="electricity demand")

    # normalize the data:
    plot_df["electricity demand"] = plot_df.groupby("year")["electricity demand"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    g = sns.FacetGrid(plot_df.loc[plot_df["variable"].isin(["grid demand reference", "grid demand HEMS"])], col="year", col_wrap=2, height=5, aspect=1.5, sharey=False)

    # Map the barplot to each facet
    g.map_dataframe(
        sns.lineplot,
        x="day_hour",
        y="electricity demand",
        hue="variable",
        style="ID_EnergyPrice"

    )

    # Adjust plot aesthetics
    g.set_axis_labels("hour", "average grid demand normalized")
    g.add_legend(title="")
    g.set_titles("Year {col_name}")

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_load_factor(loads: pd.DataFrame, national: pd.DataFrame, scenario: str):
    plot_df = pd.DataFrame()
    # national.drop(columns=["scenario"], inplace=True)
    loads["year"] = loads["year"].astype(int)
    loads["country"] = loads["country"].astype(str)
    loads["ID_EnergyPrice"] = loads["ID_EnergyPrice"].astype(int)
    # df = pd.merge(left=loads, right=national, on=["Hour", "country", "year"]).drop_duplicates().reset_index(drop=True)

    
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
                for price in [1, 2]:
                    df = loads.loc[(loads["year"]==year) & (loads["country"]==country) & (loads["ID_EnergyPrice"]==price), :].copy().reset_index(drop=True)

                    # peak demand
                    plot_df.loc[i, "country"] = country
                    plot_df.loc[i, "year"] = year
                    plot_df.loc[i, "ID_EnergyPrice"] = price
                    plot_df.loc[i, f"peak_demand_load_factor"] = (df.loc[peak_demand_hour, "opt_grid_demand_stock_MW"]- df.loc[peak_demand_hour, "ref_grid_demand_stock_MW"]) / demand[peak_demand_hour] * 100
                    plot_df.loc[i, "min_demand_load_factor"] = (df.loc[min_demand_hour, "opt_grid_demand_stock_MW"] - df.loc[min_demand_hour, "ref_grid_demand_stock_MW"]) / demand[min_demand_hour] * 100

                    # peak price
                    peak_price_hour = df[f"price (cent/kWh)"].idxmax()
                    min_price_hour = df[f"price (cent/kWh)"].idxmin()

                    plot_df.loc[i, f"peak_price_load_factor"] = (df.loc[peak_price_hour, "opt_grid_demand_stock_MW"] - df.loc[peak_price_hour, "ref_grid_demand_stock_MW"]) / demand[peak_price_hour] * 100
                    plot_df.loc[i, f"min_price_load_factor"] = (df.loc[min_price_hour, "opt_grid_demand_stock_MW"] - df.loc[min_price_hour, "ref_grid_demand_stock_MW"]) / demand[min_price_hour] * 100

                    i += 1


    sns.barplot(
        data=plot_df,
        x="country",
        y="peak_demand_load_factor",
        hue="year"
    )
    plt.suptitle("load factor in peak hour")
    plt.xticks(rotation=90)
    plt.show()

    sns.barplot(
        data=plot_df,
        x="country",
        y="min_demand_load_factor",
        hue="year"
    )
    plt.suptitle("load factor in minimum demand hour")
    plt.xticks(rotation=90)
    plt.show()

    sns.barplot(
        data=plot_df,
        x="country",
        y="peak_price_load_factor",
        hue="year"
    )
    plt.suptitle("load factor at peak price hour")
    plt.xticks(rotation=90)
    plt.show()

    sns.barplot(
        data=plot_df,
        x="country",
        y="min_price_load_factor",
        hue="year"
    )
    plt.xticks(rotation=90)
    plt.suptitle("load factor at min price hour")
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

    print("the demand data from AURESII is smaller than the electricity demand of the buildings in the same scenarios, therefore the load factor is so insanely high.")


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


def main(percentage_cooling: float):
    path_2_demand_file = Path(__file__).parent / f"EU27_loads_cooling-{percentage_cooling}.parquet.gzip"
    if not path_2_demand_file.exists():
        Cp.main(percentage_cooling)
    df = pd.read_parquet(Path(__file__).parent / f"EU27_loads_cooling-{percentage_cooling}.parquet.gzip")
    national_demand = Cp.get_national_demand_profiles()
    national_demand = national_demand.loc[(national_demand["scenario"]=="shiny happy") | (national_demand["scenario"]=="baseyear"), :]
    
    # plot_PV_self_consumption(loads=df)
    # plot_flexible_storage_efficiency(loads=df)
    # show_average_day_profile(loads=df)
    plot_load_factor(loads=df, national=national_demand, scenario="shiny happy")

    # TODO rerun with positive prices! 20 cent grid fee
    # show_day_with_peak_deamand(profiles=df, scenario=scenario, national=national_demand)


if __name__ == "__main__":

    main(
        percentage_cooling=0.1,
    )
