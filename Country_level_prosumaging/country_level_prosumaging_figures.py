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
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.signal import find_peaks
from matplotlib.patches import Patch, ConnectionPatch
import matplotlib.patches as mpatches


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
    output_path = Path(__file__).parent / "single_country_plots" / f"Energy_over_price_{country}_{year}_cooling{COOLING_PERCENTAGE}.png"
    plt.tight_layout()
    plt.savefig(output_path)

    # Close the plot to free up memory
    plt.close()


def plot_PV_self_consumption(loads: pd.DataFrame, percentage_cooling, project_acronym):
    # need to load only the buildings that have PV installed:
    parquet_file = Path(__file__).parent / f"PV_loads_{project_acronym}_cooling-{percentage_cooling}.parquet.gzip"
    if not parquet_file.exists():
        Cp.load_only_PV_building_profiles(percentage_cooling, parquet_file, project_acronym)
    df = pd.read_parquet(parquet_file)

    plot_df=df.copy()
    plot_df["ID_EnergyPrice"]=plot_df["ID_EnergyPrice"].map({i+1:val for i,val in enumerate(add_price_information().values())})
    x_order = plot_df.groupby(["country"])["self_consumption"].mean().sort_values().index

    fontsize = 20
    g = sns.FacetGrid(plot_df, col="type", row="year", height=8, aspect=1, sharey=True)
    g.map_dataframe(
        sns.boxplot,
        x="country",
        y="self_consumption",
        hue="ID_EnergyPrice",
        dodge=True,  # Ensures bars are grouped within x-axis categories
        hue_order=None, 
        order=x_order,
        palette=sns.color_palette()
    )

    # Adjust plot aesthetics
    g.set_axis_labels("Country", "PV self consumption")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False,
        prop={"size": fontsize}
    )
    g.set_titles(row_template='{row_name}', col_template='{col_name}', size=fontsize)
    g.set_axis_labels("Country", "PV self consumption (%)", fontsize=fontsize)
    for ax in g.axes.flat:
        ax.tick_params(labelbottom=True, labelsize=fontsize)  # Force show x labels
    g.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"PV_self_consumption_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

    # average PV self consumption over Europe:
    eu_df = df.copy()
    eu_df = eu_df.groupby(["year", "type", "ID_EnergyPrice"])["self_consumption"].mean().reset_index()
    eu_df = eu_df[~((eu_df["type"] == "simulation") & (eu_df["ID_EnergyPrice"] != 1))]
    eu_df["ID_EnergyPrice"]=eu_df["ID_EnergyPrice"].map({i+1:val for i,val in enumerate(add_price_information().values())})
    eu_df.loc[eu_df["type"] == "simulation","ID_EnergyPrice"] = "simulation"

    plt.rcParams.update({
        'axes.labelsize': 10,    # x and y axis labels
        'axes.titlesize': 12,    # subplot titles
        'xtick.labelsize': 8,   # x-axis tick labels
        'ytick.labelsize': 8,   # y-axis tick labels
        'legend.fontsize': 8,   # legend text size
    })
    palette = {"simulation": sns.color_palette()[5]} | {list(add_price_information().values())[i]:sns.color_palette()[i] for i in range(5)}
    fig, ax = plt.subplots(figsize=(3,5))
    sns.barplot(
        data=eu_df,
        x="year",
        y="self_consumption",
        hue="ID_EnergyPrice",
        orient="x",
        dodge=True,  # Ensures bars are grouped within x-axis categories
        hue_order=["simulation"]+list(add_price_information().values()), 
        palette=palette,
        linewidth=0.5
    )
    ax.set_xlabel("year")
    ax.set_ylabel("average PV self consumption over all countries (%)")
    ax.legend(
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=2,
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"PV_self_consumption_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()


def plot_flexible_storage_efficiency(loads: pd.DataFrame):

    loads["charging_energy"] = loads["opt_grid_demand_stock_MW"] - loads["ref_grid_demand_stock_MW"]
    loads.loc[loads["charging_energy"] < 0, "charging_energy"] = 0

    grouped = loads.groupby(["country", "year", "ID_EnergyPrice"])
    storage_efficiency = 1 - (grouped["opt_grid_demand_stock_MW"].sum() - grouped["ref_grid_demand_stock_MW"].sum()) / grouped["charging_energy"].sum()
    storage_efficiency = storage_efficiency.reset_index()
    storage_efficiency.rename(columns={0: "storage efficiency"}, inplace=True)

    g = sns.FacetGrid(storage_efficiency, col="ID_EnergyPrice", col_wrap=1, height=5, aspect=1.5, sharex=True, sharey=False)
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="storage efficiency",
        hue="year",
        palette=sns.color_palette(),
    )

    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  
    g.set_axis_labels("Country", "storage efficiency ignoring PV")

    plt.savefig(SAVING_PATH / f"Storage_efficiency_cooling{COOLING_PERCENTAGE}.svg")
    # plt.show()
    plt.close()

    # calculate the storage efficiency without PV:
    loads["ref_Load_MW"] = loads["ref_grid_demand_stock_MW"] + loads["ref_PV2Load_MW"]
    loads["opt_Load_MW"] = loads["opt_grid_demand_stock_MW"] + loads["opt_PV2Load_MW"]
    loads["charging_energy_pv_cleaned"] = loads["opt_Load_MW"] - loads["ref_Load_MW"]
    loads.loc[loads["charging_energy_pv_cleaned"] < 0, "charging_energy_pv_cleaned"] = 0

    grouped = loads.groupby(["country", "year", "ID_EnergyPrice"])
    storage_efficiency = 1 - (grouped["opt_Load_MW"].sum() - grouped["ref_Load_MW"].sum()) / grouped["charging_energy_pv_cleaned"].sum()
    storage_efficiency = storage_efficiency.reset_index(name="storage efficiency")
    order = storage_efficiency.groupby("country")["storage efficiency"].mean().sort_values().index

    g = sns.FacetGrid(storage_efficiency, col="ID_EnergyPrice", col_wrap=1, height=6, aspect=1.5, sharex=True, sharey=False)
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="storage efficiency",
        hue="year",
        palette=sns.color_palette(),
        order=order,
    )

    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  
    # g.fig.subplots_adjust(bottom=0.2)
    g.set_axis_labels("", "storage efficiency including PV")
    g.add_legend()
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Storage_efficiency_with_PV_cooling{COOLING_PERCENTAGE}.svg")
    # plt.show()
    plt.close()

def show_average_day_profile(loads: pd.DataFrame):

    # loads["ref_Load_MW"] = loads["ref_grid_demand_stock_MW"] + loads["ref_PV2Load_MW"]
    # loads["opt_Load_MW"] = loads["opt_grid_demand_stock_MW"] + loads["opt_PV2Load_MW"]
    loads["day_hour"] = (loads["Hour"]-1) % 24 + 1

    df = loads.groupby(["year", "ID_EnergyPrice", "day_hour", "country"])[["ref_grid_demand_stock_MW", "opt_grid_demand_stock_MW",]].mean().reset_index()
    plot_df = df.rename(columns={
        "ref_grid_demand_stock_MW": "grid demand reference",
        "opt_grid_demand_stock_MW": "grid demand prosumager",
        # "ref_Load_MW": "Load reference",
        # "opt_Load_MW": "Load prosumager"
    }).melt(id_vars=["year", "ID_EnergyPrice", "day_hour", "country"], value_name="electricity demand")

    # normalize the data:
    plot_df["electricity demand"] = plot_df.groupby(["year", "ID_EnergyPrice", "country", "variable"])["electricity demand"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    plot_df_mean = plot_df.groupby(["year", "ID_EnergyPrice", "day_hour", "variable"])["electricity demand"].mean().reset_index()
    df_ = plot_df_mean.loc[(plot_df_mean["variable"]=="grid demand reference") & (plot_df_mean["ID_EnergyPrice"]=="Price 2"), :]
    df_["ID_EnergyPrice"] = "reference"
    plot_df_merged = pd.concat([df_, plot_df_mean.loc[plot_df_mean["variable"]=="grid demand prosumager", :]], axis=0)

    pallette = [sns.color_palette()[i-1] if name != "reference" else "black" for i, name in enumerate(plot_df_merged["ID_EnergyPrice"].unique()) ]

    g = sns.FacetGrid(plot_df_merged, 
                      col="year", col_wrap=2, height=5, aspect=1.5, sharey=True)
    g.map_dataframe(
        sns.lineplot,
        x="day_hour",
        y="electricity demand",
        hue="ID_EnergyPrice",
        palette=pallette,

    )
    # Adjust plot aesthetics
    g.set_axis_labels("hour", "average grid demand normalized")
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0, 1.15),
        title="",
        frameon=False
    )
    g.set_titles("{col_name}")

    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Average_day_grid_demand_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

def plot_load_factor(loads: pd.DataFrame, national: pd.DataFrame, scenario: str):
    plot_df = pd.DataFrame()
    # national.drop(columns=["scenario"], inplace=True)
    loads["year"] = loads["year"].astype(int)
    loads["country"] = loads["country"].astype(str)
    # df = pd.merge(left=loads, right=national, on=["Hour", "country", "year"]).drop_duplicates().reset_index(drop=True)

    
    i = 0
    for country in Cp.EUROPEAN_COUNTRIES.keys():
        for year in [2030, 2050]:
            if country == "CYP" or country=="MLT":
                continue
            else:
                if year == 2020:
                    scen = "baseyear"
                else:
                    scen = scenario
                demand = national.loc[(national["country"]==country) & (national["year"].astype(int)==year) & (national["scenario"]==scen), "demand MW"].reset_index(drop=True)
                peak_demand_hour = demand.idxmax()
                min_demand_hour = demand.idxmin()
                for price in ["Price 1", "Price 2"]:
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

    plot_df["year"] = plot_df["year"].astype(int)

    def plot_load_factor(column_name: str):    
        g = sns.FacetGrid(plot_df, col="ID_EnergyPrice", col_wrap=3, height=5, aspect=1.5, sharey=True, sharex=True)
        g.map_dataframe(
            sns.barplot,
                x="country",
                y=column_name,
                hue="year",
                palette=sns.color_palette()
        )

        # Adjust plot aesthetics
        g.set_axis_labels("country", "load factor in peak hour (%)")
        g.add_legend(
            title="",
            loc='lower center',  # centered horizontally
            bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
            ncol=5,
            frameon=False
        )
        col_title = "{col_name}"
        g.set_titles(f"load factor in {column_name.split('_')[0]} {column_name.split('_')[1]} hour with price {col_title}")
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.tight_layout()
        plt.savefig(SAVING_PATH / f"{column_name}_cooling{COOLING_PERCENTAGE}.svg")
        # plt.show()
        plt.close()

    plot_load_factor(column_name="peak_demand_load_factor")
    plot_load_factor(column_name="min_demand_load_factor")
    plot_load_factor(column_name="peak_price_load_factor")
    plot_load_factor(column_name="min_price_load_factor")


    def plot_load_factor_EU(column_name: str):
        plt.figure()
        sns.boxplot(
            data=plot_df,
            y="year",
            x=column_name,
            hue="ID_EnergyPrice",
            orient="y",
            palette=sns.color_palette()

        )
        plt.legend(title="Price Scenario")
        plt.xlabel(f"load factors of all EU27 in {column_name.split('_')[0]} {column_name.split('_')[1]} hour (%)")
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        plt.savefig(SAVING_PATH / f"{column_name}_EU_cooling{COOLING_PERCENTAGE}.svg")
        plt.close()

    plot_load_factor_EU("peak_demand_load_factor")
    plot_load_factor_EU(column_name="min_demand_load_factor")
    plot_load_factor_EU(column_name="peak_price_load_factor")
    plot_load_factor_EU(column_name="min_price_load_factor")

def plot_national_peaks(peak_df: pd.DataFrame):
    plot_df = peak_df.reset_index()
    # order = plot_df.groupby("country")["price"].mean().sort_values().index
    g = sns.FacetGrid(plot_df, col="year", col_wrap=2, height=5, aspect=1.5, sharey=True)

    g.map_dataframe(
        sns.barplot,
        x="country",
        y="change in peak relative",
        hue="price",
        # order=order,
        palette=sns.color_palette()
    )

    # Adjust plot aesthetics
    g.set_axis_labels("country", "relative change in peak demand (%)")
    g.add_legend()
    g.set_titles("Electricity price {col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Relative_Change_in_Peak_demand_cooling{COOLING_PERCENTAGE}.svg")
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
        year = column.split("_")[1]
        country = column.split("_")[2]
        typ = column.split("_")[0]
        price = column.split("_")[-1]

        if year == 2020 and (country == "CYP" or country=="MLT" or country=="NLD"):
            continue

        if typ == "ref":
            dash = "dot"
        else:
            if price == 1:
                dash = "solid"
            else:
                dash = "dash"

        fig.add_trace(
            go.Scatter(
                x=day_df.index,
                y=day_df[column],
                mode="lines",
                name=f"{typ} {country} {year} {price}",
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
    source_column = "no prosumager"
    target_column = "all prosumager"
    value_column = "count"
    for price in [1, 2]:
        plot_df = df.reset_index()
        plot_df = plot_df.loc[plot_df["price"]==price, :]
        flows = plot_df.groupby(["all prosumager", "no prosumager"]).size().reset_index(name="count")
        
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
        source_labels = ["Spring (no prosumager)", "Summer (no prosumager)", "Autumn (no prosumager)", "Winter (no prosumager)"]
        target_labels = ["Spring (all prosumager)", "Summer (all prosumager)", "Autumn (all prosumager)", "Winter (all prosumager)"]

        # Combine source and target labels into a single list
        labels = source_labels + target_labels
        label_to_index = {label: i for i, label in enumerate(labels)}


        # Map "all prosumager" (source) to source_labels indices
        sources = flows[source_column].map(lambda x: label_to_index[f"{x} (no prosumager)"])
        # Map "no prosumager" (target) to target_labels indices
        targets = flows[target_column].map(lambda x: label_to_index[f"{x} (all prosumager)"])
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
        fig.write_html(saving_path / f"sankey_seasonal_peaks_price{price}.html")
        fig.write_image(saving_path / f"sankey_seasonal_peaks_price{price}.svg")

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
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_axis_labels("Season", "number of peaks occuring in each season")
    g.tight_layout()
    plt.show()

def show_national_demand_increase_in_high_and_low_price_quantile(loads: pd.DataFrame, national: pd.DataFrame):
    merged = pd.merge(left=loads, right=national[["country", "demand MW", "year", "Hour"]], on=["year", "Hour", "country"])
    merged["demand_opt"] = merged["demand MW"] - merged["ref_grid_demand_stock_MW"] + merged["opt_grid_demand_stock_MW"]
    groups = merged.groupby(["year", "country", "ID_EnergyPrice"])

    # increase in demand at low prices:
    low_demand_ref = groups[["demand MW", "price (cent/kWh)"]].apply(
        lambda g: g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "demand MW"].sum()
    ).reset_index(name="reference low price demand")
    low_demand_opt = groups[["demand_opt", "price (cent/kWh)"]].apply(
        lambda g: g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "demand_opt"].sum()
    ).reset_index(name="prosumager low price demand")
    plot_df = pd.merge(right=low_demand_ref, left=low_demand_opt, on=["year", "country", "ID_EnergyPrice"])
    plot_df["1st quartile demand increase (%)"] = (plot_df["prosumager low price demand"] - plot_df["reference low price demand"]) / plot_df["reference low price demand"] * 100
    
    
    eu_groups = plot_df.groupby(["year", "ID_EnergyPrice"])[["prosumager low price demand", "reference low price demand"]].sum().reset_index()
    eu_groups["1st quartile demand increase (%)"] = (eu_groups["prosumager low price demand"] - eu_groups["reference low price demand"]) / eu_groups["reference low price demand"] * 100
    
    
    high_demand_ref = groups[["demand MW", "price (cent/kWh)"]].apply(
        lambda g: g.loc[g["price (cent/kWh)"] >= g["price (cent/kWh)"].quantile(0.75), "demand MW"].sum()
    ).reset_index(name="reference high price demand")
    high_demand_opt = groups[["demand_opt", "price (cent/kWh)"]].apply(
        lambda g: g.loc[g["price (cent/kWh)"] >= g["price (cent/kWh)"].quantile(0.75), "demand_opt"].sum()
    ).reset_index(name="prosumager high price demand")
    plot_df2 = pd.merge(right=high_demand_ref, left=high_demand_opt, on=["year", "country", "ID_EnergyPrice"])
    plot_df2["4th quartile demand increase (%)"] = (plot_df2["prosumager high price demand"] - plot_df2["reference high price demand"]) / plot_df2["reference high price demand"] * 100
    
    eu_groups2 = plot_df2.groupby(["year", "ID_EnergyPrice"])[["prosumager high price demand", "reference high price demand"]].sum().reset_index()
    eu_groups2["4th quartile demand increase (%)"] = (eu_groups2["prosumager high price demand"] - eu_groups2["reference high price demand"]) / eu_groups2["reference high price demand"] * 100
    eu_df = pd.merge(left=eu_groups, right=eu_groups2[["year", "ID_EnergyPrice", "4th quartile demand increase (%)"]], on=["year", "ID_EnergyPrice"])

    plot_df_large = pd.merge(left=plot_df, right=plot_df2[["year", "ID_EnergyPrice", "4th quartile demand increase (%)", "country"]], on=["year", "ID_EnergyPrice", "country"])
    
    plt.figure(figsize=(8,6))
    ax2 = sns.barplot(
        data=eu_df,
        x="4th quartile demand increase (%)",
        y="year",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        orient="y",
    )
    for i, p in enumerate(ax2.patches):
        p.set_hatch('//') 
    ax1 = sns.barplot(
        data=eu_df,
        x="1st quartile demand increase (%)",
        y="year",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        orient="y",
    )
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel("change in total electricity grid demand in 1st and 4th price quantile on EU level (%)")
    plt.ylabel("year")
    legend_handles = [
        mpatches.Patch(color=sns.color_palette()[0], label="0 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[1], label="5 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[2], label="10 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[3], label="20 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[4], label="40 cent/kWh grid fees"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="1st price quartile"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="4th price quartile"),
    ]
    plt.legend(handles=legend_handles, title="", loc="lower right")
    plt.xlim(
        min(eu_df["4th quartile demand increase (%)"]),
        max(eu_df["1st quartile demand increase (%)"])
    )
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Change_in_total_demand_in_price_quantiles_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

    order = plot_df_large.groupby("country")["4th quartile demand increase (%)"].mean().sort_values().index
    g = sns.FacetGrid(plot_df_large, col="year",col_wrap=2, height=5, aspect=1.5, sharey=True, sharex=True)

    # Map the barplot to each facet
    ax1 = g.map_dataframe(
        sns.barplot,
        x="country",
        y="4th quartile demand increase (%)",
        hue="ID_EnergyPrice",
        order=order,
        palette=sns.color_palette()
    )
    for i, ax in enumerate(ax1.axes):
        for p in ax.patches:
            p.set_hatch('///')
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="1st quartile demand increase (%)",
        hue="ID_EnergyPrice",
        order=order,
        palette=sns.color_palette()
    )
    # Adjust plot aesthetics
    g.set_axis_labels("country", "increase in total electricity grid demand in 1st and 4th price quantiles")
    legend_handles = [
        mpatches.Patch(color=sns.color_palette()[0], label="0 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[1], label="5 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[2], label="10 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[3], label="20 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[4], label="40 cent/kWh grid fees"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="1st price quartile"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="4th price quartile"),
    ]
    plt.legend(handles=legend_handles, title="", loc="lower right",)
    g.set_titles("Price {col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Increase_in_total_demand_in_1st_and_4th_price_quantile_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()


def show_residential_demand_increase_in_high_and_low_price_quantile(loads: pd.DataFrame):
    groups = loads.groupby(["year", "country", "ID_EnergyPrice"])

    # increase in demand at low prices:
    low_demand_ref = groups[["ref_grid_demand_stock_MW", "price (cent/kWh)"]].apply(
        lambda g: g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "ref_grid_demand_stock_MW"].sum()
    ).reset_index(name="reference low price demand")
    low_demand_opt = groups[["opt_grid_demand_stock_MW", "price (cent/kWh)"]].apply(
        lambda g: g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "opt_grid_demand_stock_MW"].sum()
    ).reset_index(name="prosumager low price demand")
    plot_df = pd.merge(right=low_demand_ref, left=low_demand_opt, on=["year", "country", "ID_EnergyPrice"])
    plot_df["1st quartile demand increase (%)"] = (plot_df["prosumager low price demand"] - plot_df["reference low price demand"]) / plot_df["reference low price demand"] * 100
    x_order = plot_df.groupby("country")["1st quartile demand increase (%)"].mean().sort_values().index
    
    g = sns.FacetGrid(plot_df, col="year",col_wrap=2, height=5, aspect=1.5, sharey=True, sharex=True)

    # Map the barplot to each facet
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="1st quartile demand increase (%)",
        hue="ID_EnergyPrice",
        order=x_order,
        palette=sns.color_palette()
    )

    # Adjust plot aesthetics
    g.set_axis_labels("country", "increase in residential electricity grid demand in 1st price quantile")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_titles("Price {col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Increase_in_residential_demand_in_1st_price_quantile_cooling{COOLING_PERCENTAGE}.svg")
    # plt.show()
    plt.close()

    # Increase of demand in 1st price quantile on EU level:
    eu_groups = plot_df.groupby(["year", "ID_EnergyPrice"])[["prosumager low price demand", "reference low price demand"]].sum().reset_index()
    eu_groups["1st quartile demand increase (%)"] = (eu_groups["prosumager low price demand"] - eu_groups["reference low price demand"]) / eu_groups["reference low price demand"] * 100

    # increase in demand at high prices:
    high_demand_ref = groups[["ref_grid_demand_stock_MW", "price (cent/kWh)"]].apply(
        lambda g: g.loc[g["price (cent/kWh)"] >= g["price (cent/kWh)"].quantile(0.75), "ref_grid_demand_stock_MW"].sum()
    ).reset_index(name="reference high price demand")
    high_demand_opt = groups[["opt_grid_demand_stock_MW", "price (cent/kWh)"]].apply(
        lambda g: g.loc[g["price (cent/kWh)"] >= g["price (cent/kWh)"].quantile(0.75), "opt_grid_demand_stock_MW"].sum()
    ).reset_index(name="prosumager high price demand")
    plot_df = pd.merge(right=high_demand_ref, left=high_demand_opt, on=["year", "country", "ID_EnergyPrice"])
    plot_df["4th quartile demand increase (%)"] = (plot_df["prosumager high price demand"] - plot_df["reference high price demand"]) / plot_df["reference high price demand"] * 100
    
    g = sns.FacetGrid(plot_df, col="ID_EnergyPrice",col_wrap=2, height=5, aspect=1.5, sharey=True, sharex=True)

    # Map the barplot to each facet
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="4th quartile demand increase (%)",
        hue="year",
        order=x_order,
        palette=sns.color_palette()
    )

    # Adjust plot aesthetics
    g.set_axis_labels("country", "increase in residential electricity grid demand in 4th price quantile")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_titles("Price {col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Increase_in_residential_demand_in_4th_price_quantile_cooling{COOLING_PERCENTAGE}.svg")
    # plt.show()
    plt.close()

    # Increase of demand in 1st price quantile on EU level:
    eu_groups2 = plot_df.groupby(["year", "ID_EnergyPrice"])[["prosumager high price demand", "reference high price demand"]].sum().reset_index()
    eu_groups2["4th quartile demand increase (%)"] = (eu_groups2["prosumager high price demand"] - eu_groups2["reference high price demand"]) / eu_groups2["reference high price demand"] * 100
    eu_df = pd.merge(left=eu_groups, right=eu_groups2[["year", "ID_EnergyPrice", "4th quartile demand increase (%)"]], on=["year", "ID_EnergyPrice"])
    

    ax2 = sns.barplot(
        data=eu_df,
        x="4th quartile demand increase (%)",
        y="year",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        orient="y",
    )
    for p in ax2.patches:
        p.set_hatch("//")

    ax1 = sns.barplot(
        data=eu_df,
        x="1st quartile demand increase (%)",
        y="year",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        orient="y",
    )
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel("change in residential electricity grid demand in 1st and 4th price quartile on EU level (%)")
    plt.ylabel("year")
    legend_handles = [
        mpatches.Patch(color=sns.color_palette()[0], label="Price 1: 0 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[1], label="Price 2: 5 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[2], label="Price 3: 10 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[3], label="Price 4: 20 cent/kWh grid fees"),
        mpatches.Patch(color=sns.color_palette()[4], label="Price 5: 40 cent/kWh grid fees"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="1st price quartile"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="4th price quartile"),
    ]
    plt.legend(handles=legend_handles, title="Electricity price scenario",)
    plt.xlim(
        np.floor(min(eu_df["4th quartile demand increase (%)"])/10)*10,
        np.ceil(max(eu_df["1st quartile demand increase (%)"])/10)*10
    )
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Change_in_residential_demand_in_price_quantiles_EU_cooling{COOLING_PERCENTAGE}.svg")
    # plt.show()
    plt.close()


def show_flexibility_factor(loads: pd.DataFrame):
    # from https://www.sciencedirect.com/science/article/pii/S0360544216306934?via%3Dihub#bib57
    groups = loads.groupby(["year", "country", "ID_EnergyPrice"])
    factor_ref = groups[["ref_grid_demand_stock_MW", "price (cent/kWh)"]].apply(
        lambda g: (g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "ref_grid_demand_stock_MW"].sum() - g.loc[g["price (cent/kWh)"] >=g["price (cent/kWh)"].quantile(0.75), "ref_grid_demand_stock_MW"].sum()) / (g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "ref_grid_demand_stock_MW"].sum() + g.loc[g["price (cent/kWh)"] >=g["price (cent/kWh)"].quantile(0.75), "ref_grid_demand_stock_MW"].sum())
    ).reset_index(name="flexibility factor reference")

    factor_opt = groups[["opt_grid_demand_stock_MW", "price (cent/kWh)"]].apply(
        lambda g: (g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "opt_grid_demand_stock_MW"].sum() - g.loc[g["price (cent/kWh)"] >=g["price (cent/kWh)"].quantile(0.75), "opt_grid_demand_stock_MW"].sum()) / (g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "opt_grid_demand_stock_MW"].sum() + g.loc[g["price (cent/kWh)"] >=g["price (cent/kWh)"].quantile(0.75), "opt_grid_demand_stock_MW"].sum())
    ).reset_index(name="flexibility factor prosumager")
    
    plot_df = pd.merge(right=factor_ref, left=factor_opt, on=["year", "country", "ID_EnergyPrice"])
    plot_df["flexibility factor change"] = plot_df["flexibility factor prosumager"] - plot_df["flexibility factor reference"]
    x_order = plot_df.groupby("country")["flexibility factor change"].mean().sort_values().index
    x_order_ref = plot_df.groupby("country")["flexibility factor reference"].mean().sort_values().index


    sns.barplot(
        data=plot_df,
        x="country",
        y="flexibility factor reference",
        hue="year",
        order=x_order_ref,
        palette=sns.color_palette()

    )
    # Show the plot
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Flexibility_factor_reference_cooling{COOLING_PERCENTAGE}.svg")
    # plt.show()
    plt.close()


    g = sns.FacetGrid(plot_df, col="year", col_wrap=2, height=5, aspect=0.5, sharey=True)
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="flexibility factor change",
        hue="ID_EnergyPrice",
        order=x_order,
        palette=sns.color_palette()

    )
    g.set_axis_labels("country", "change in flexibility factor")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_titles("Price {col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Flexibility_factor_change_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    print("A higher flexibility factor means that more energy is consumed at low prices. if the factor = 1 it means that no electricity is consumed from the grid at high prices")
    plt.close()


    # flex factor on EU level:
    copy_df = plot_df.loc[plot_df["ID_EnergyPrice"]=="Price 1", :].copy().rename(columns={"flexibility factor reference": "flexibility factor"})
    copy_df["ID_EnergyPrice"] = "reference"
    plot_df["ID_EnergyPrice"] = plot_df["ID_EnergyPrice"].map(add_price_information())
    new_df = pd.concat([plot_df.rename(columns={"flexibility factor prosumager": "flexibility factor"}), copy_df], axis=0)

    sns.boxplot(
        data=new_df,
        x="flexibility factor",
        y="year",
        hue="ID_EnergyPrice",
        orient="y",
        palette=sns.color_palette()
    )
    plt.xlabel("Flexibility factor across the EU27")
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Flexibility_factor_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()


def calculate_GSC(loads: pd.DataFrame)-> pd.DataFrame:
# from https://www.sciencedirect.com/science/article/pii/S0306261915013434

    # GSC absolute is the sum of consumption times price in every hour divided by the sum of consumption times thes average price
    loads["demand_price_ref"] = loads["ref_grid_demand_stock_MW"] * loads["price (cent/kWh)"]
    loads["demand_price_opt"] = loads["opt_grid_demand_stock_MW"] * loads["price (cent/kWh)"]
    loads["day"] = (loads["Hour"]-1) // 24 +1
    # we calculate the GSC relative also on daily basis and take the mean over the year
    day_groups = loads.groupby(["day", "ID_EnergyPrice", "country", "year"])
    P_max = loads["opt_grid_demand_stock_MW"].max()
    dfs= []
    for (day, price_id, country, year), group in day_groups:
        demand_ref_sum = group["ref_grid_demand_stock_MW"].sum()
        demand_opt_sum = group["opt_grid_demand_stock_MW"].sum()

        full_load_h_ref = demand_ref_sum / P_max
        last_hour_weights_ref = full_load_h_ref - int(full_load_h_ref)
        full_load_h_opt = demand_opt_sum / P_max
        last_hour_weights_opt = full_load_h_opt - int(full_load_h_opt)

        # full_load_h = int(min([full_load_h_ref, full_load_h_opt]))
        demand_in_full_load_h_ref = demand_ref_sum / full_load_h_ref
        demand_in_full_load_h_opt= demand_ref_sum / full_load_h_opt

        best_hours_indices_ref = group["price (cent/kWh)"].sort_values().iloc[:int(np.ceil(full_load_h_ref))].index
        best_hours_indices_opt = group["price (cent/kWh)"].sort_values().iloc[:int(np.ceil(full_load_h_opt))].index

        worst_hours_indices_ref = group["price (cent/kWh)"].sort_values().iloc[-int(np.ceil(full_load_h_ref)):].index
        worst_hours_indices_opt = group["price (cent/kWh)"].sort_values().iloc[-int(np.ceil(full_load_h_opt)):].index

        mean_price =  group["price (cent/kWh)"].mean()
        ref_hour_weights = np.array([1 if i!=len(best_hours_indices_ref)-1 else last_hour_weights_ref for i in range(len(best_hours_indices_ref))])
        opt_hour_weights = np.array([1 if i!=len(best_hours_indices_opt)-1 else last_hour_weights_opt for i in range(len(best_hours_indices_opt))])

        # calculate GSC abs for worst and best case:
        GSC_ref_worst = (demand_in_full_load_h_ref * group["price (cent/kWh)"].loc[worst_hours_indices_ref].values * ref_hour_weights).sum() / (mean_price * demand_ref_sum)
        GSC_ref_best = (demand_in_full_load_h_ref * group["price (cent/kWh)"].loc[best_hours_indices_ref].values * ref_hour_weights).sum() / (mean_price * demand_ref_sum)
        GSC_ref_achieved = (group["ref_grid_demand_stock_MW"] * group["price (cent/kWh)"]).sum() / (mean_price * demand_ref_sum)

        GSC_opt_worst = (demand_in_full_load_h_opt * group["price (cent/kWh)"].loc[worst_hours_indices_opt].values * opt_hour_weights).sum() / (mean_price * demand_opt_sum)
        GSC_opt_best = (demand_in_full_load_h_opt * group["price (cent/kWh)"].loc[best_hours_indices_opt].values * opt_hour_weights).sum() / (mean_price * demand_opt_sum)
        GSC_opt_achieved = (group["opt_grid_demand_stock_MW"] * group["price (cent/kWh)"]).sum() / (mean_price * demand_opt_sum)

        if GSC_ref_worst - GSC_ref_best == 0:
            GSC_rel_ref = 0
            GSC_rel_opt = 0
        else:
            GSC_rel_ref = 200 * (GSC_ref_worst - GSC_ref_achieved) / (GSC_ref_worst - GSC_ref_best) - 100
            GSC_rel_opt = 200 * (GSC_opt_worst - GSC_opt_achieved) / (GSC_opt_worst - GSC_opt_best) - 100
        
        GSC_rel_ref = max(min(100, GSC_rel_ref), -100)
        GSC_rel_opt = max(min(100, GSC_rel_opt), -100)

        GSC_rel_ref_weighted = GSC_rel_ref * (mean_price * demand_ref_sum)
        GSC_rel_opt_weighted = GSC_rel_opt * (mean_price * demand_opt_sum)


        dfs.append(pd.DataFrame(data=[[day, country, year, price_id, GSC_ref_achieved, GSC_opt_achieved, GSC_rel_ref, GSC_rel_opt, GSC_rel_ref_weighted, GSC_rel_opt_weighted, (mean_price * demand_opt_sum)]], 
                     columns=["day", "country", "year", "ID_EnergyPrice", "GSC_abs_ref", "GSC_abs_opt", "GSC_rel_ref", "GSC_rel_opt", "GSC_rel_ref_weighted", "GSC_rel_opt_weighted", "mean_price*demand"]))
    
    GSC_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    GSC_weighted = GSC_df.groupby(["ID_EnergyPrice", "country", "year"])[["GSC_rel_ref_weighted", "GSC_rel_opt_weighted", "mean_price*demand"]].sum().reset_index()
    GSC_weighted["GSC_rel_ref_weighted"] = GSC_weighted["GSC_rel_ref_weighted"] / GSC_weighted["mean_price*demand"]
    GSC_weighted["GSC_rel_opt_weighted"] = GSC_weighted["GSC_rel_opt_weighted"] / GSC_weighted["mean_price*demand"]

    GSC_mean = GSC_df.groupby(["ID_EnergyPrice", "country", "year"])[["GSC_abs_ref", "GSC_abs_opt", "GSC_rel_ref", "GSC_rel_opt"]].mean().reset_index()
    GSC = pd.merge(left=GSC_mean, right=GSC_weighted[["ID_EnergyPrice", "country", "year", "GSC_rel_ref_weighted", "GSC_rel_opt_weighted"]], on=["ID_EnergyPrice", "country", "year"])

    copy = GSC.loc[GSC["ID_EnergyPrice"]=="Price 1", :].copy()
    copy["ID_EnergyPrice"] = "reference"
    eu_df = pd.concat([GSC[["ID_EnergyPrice", "country", "year", "GSC_rel_opt", "GSC_rel_opt_weighted"]].rename(columns={"GSC_rel_opt": "GSC_rel", "GSC_rel_opt_weighted": "GSC_rel_weighted"}), 
                       copy[["ID_EnergyPrice", "country", "year", "GSC_rel_ref", "GSC_rel_ref_weighted"]].rename(columns={"GSC_rel_ref": "GSC_rel", "GSC_rel_ref_weighted": "GSC_rel_weighted"})], axis=0)
    return eu_df


def show_GSCrel_and_GSC_abs(loads: pd.DataFrame):
    eu_df = calculate_GSC(loads=loads)
    eu_df["ID_EnergyPrice"] = eu_df["ID_EnergyPrice"].map(add_price_information())
    eu_df["ID_EnergyPrice"] = eu_df["ID_EnergyPrice"].fillna("reference")
    sns.boxplot(
        data=eu_df,
        x="GSC_rel",
        y="year",
        hue="ID_EnergyPrice",
        orient="y",
        palette=sns.color_palette()
    )
    plt.xlabel("GSC relative")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"GSC_relative_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()


    x_order = eu_df.groupby("country")["GSC_rel"].max().sort_values().index
    g = sns.FacetGrid(eu_df, col="year", col_wrap=2, height=6, aspect=1, sharey=True, sharex=True)
    # Map the barplot to each facet
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="GSC_rel",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        order=x_order,
    )

    g.set_axis_labels("country", "relative GSC prosumager")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=6,
        frameon=False
    )
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"GSC_relative_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

    # gewichteter GSC
    sns.boxplot(
        data=eu_df,
        x="GSC_rel_weighted",
        y="year",
        hue="ID_EnergyPrice",
        orient="y",
        palette=sns.color_palette()
    )
    plt.xlabel("GSC relative weighted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"GSC_relative_weighted_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()


def show_day_with_peak_deamand(loads: pd.DataFrame, national: pd.DataFrame):
    day_df = pd.DataFrame()
    peak_df = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["country", "year", "price"]))
    for country in Cp.EUROPEAN_COUNTRIES.keys():
        for year in [2030, 2050]:
            if country == "CYP" or country=="MLT":
                continue
            else:
                ref_col = f"ref_grid_demand_stock_MW"
                opt_col = f"opt_grid_demand_stock_MW"
                national_demand_ref = national.loc[(national["country"]==country) & (national["year"]==year), "demand MW"].copy().reset_index(drop=True)
                peak_demand_hour_ref = national_demand_ref.idxmax()
                peak_day_ref = int(peak_demand_hour_ref/24)
                peak_day_ref_season = get_season(peak_day_ref)

                ref_peak_day = national_demand_ref.iloc[peak_day_ref*24: peak_day_ref*24+24]
                peak_ref = national_demand_ref.loc[peak_demand_hour_ref]
                day_df.loc[:, f"ref_{country}_{year}_price1"] = ref_peak_day.reset_index(drop=True)
            
                for price in loads["ID_EnergyPrice"].unique():
                    price_load = loads.loc[(loads["year"]==year) & (loads["country"]==country) & (loads["ID_EnergyPrice"]==price), :].copy().reset_index(drop=True)
                    national_demand_opt = national_demand_ref - price_load.loc[:, ref_col] + price_load.loc[:, opt_col]


                    ref_minus_opt = national_demand_ref - national_demand_opt
                    peak_demand_hour_opt = national_demand_opt.idxmax()
                    peak_opt = national_demand_opt.loc[peak_demand_hour_opt]
                    peak_diff = (peak_opt - peak_ref)

                    peak_day_opt = int(peak_demand_hour_opt/24)
                    peak_day_opt_season = get_season(peak_day_opt)
                    if peak_day_opt != peak_day_ref:
                        LOGGER.info(f"peak demand day has been shifted in {country} {year}")
                    if peak_day_opt_season != peak_day_ref_season:
                        LOGGER.warning(f"peak demand day has been shifted to another season {country} {year}")


                    # cut the peak day profile for the plot:
                    opt_peak_day = national_demand_opt.iloc[peak_day_ref*24: peak_day_ref*24+24]

                    day_df.loc[:, f"opt_{country}_{year}_{price}"] = opt_peak_day.reset_index(drop=True)

                    peak_df.loc[(country, year, price), f"change in peak"] = peak_diff
                    peak_df.loc[(country, year, price), f"change in peak relative"] = peak_diff / peak_ref * 100
                    peak_df.loc[(country, year, price), f"all prosumager"] = peak_day_opt_season
                    peak_df.loc[(country, year, price), f"no prosumager"] = peak_day_ref_season


    plot_national_peaks(peak_df=peak_df)
    create_sankey_diagram(df=peak_df)
    # plot_frequency_of_peaks_in_seasons(peak_df=peak_df)
    # plot_national_peak_days(day_df=day_df)

def plot_grid_demand_increase(loads: pd.DataFrame, national: pd.DataFrame):
    demand = loads.groupby(["country", "year", "ID_EnergyPrice"])[["opt_grid_demand_stock_MW", "ref_grid_demand_stock_MW"]].sum().reset_index()
    demand["change (%)"] = (demand["opt_grid_demand_stock_MW"] - demand["ref_grid_demand_stock_MW"]) / demand["ref_grid_demand_stock_MW"] * 100  #%

    x_order = demand.groupby("country")["change (%)"].mean().sort_values().index
    plot_df = demand.copy()
    plot_df["ID_EnergyPrice"] = plot_df["ID_EnergyPrice"].map(add_price_information())

    plt.rcParams.update({
        'axes.labelsize': 16,    # x and y axis labels
        'axes.titlesize': 18,    # subplot titles
        'xtick.labelsize': 14,   # x-axis tick labels
        'ytick.labelsize': 14,   # y-axis tick labels
        'legend.fontsize': 14,   # legend text size
    })
    g = sns.FacetGrid(plot_df, col="year", col_wrap=2, height=6, aspect=0.5)
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="change (%)",
        hue="ID_EnergyPrice",
        dodge=True,  # Ensures bars are grouped within x-axis categories
        hue_order=None, 
        order=x_order,
        palette=sns.color_palette()
    )

    # Adjust plot aesthetics
    g.set_axis_labels("Country", "Change in HP heated residential electricity grid demand (%)")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.02),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_titles("Year {col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Change_in_residential_grid_demand_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()

    # average increae on EU level:
    demand["change_MW"] = demand["opt_grid_demand_stock_MW"] - demand["ref_grid_demand_stock_MW"]
    eu_demand = demand.groupby(["year", "ID_EnergyPrice"])[["change_MW", "ref_grid_demand_stock_MW"]].sum().reset_index()
    eu_demand["change (%)"] = eu_demand["change_MW"] / eu_demand["ref_grid_demand_stock_MW"] * 100
    eu_demand["ID_EnergyPrice"] = eu_demand["ID_EnergyPrice"].map(add_price_information())


    plt.rcParams.update({
        'axes.labelsize': 14,    # x and y axis labels
        'axes.titlesize': 16,    # subplot titles
        'xtick.labelsize': 12,   # x-axis tick labels
        'ytick.labelsize': 12,   # y-axis tick labels
        'legend.fontsize': 12,   # legend text size
    })
    sns.barplot(
        data=eu_demand,
        x="change (%)",
        y="year",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        orient="y"
    )
    plt.legend(loc="lower right")
    plt.xlabel("change in residential electricity grid demand (%)")
    plt.xticks(rotation=0)
    # plt.xlim(-0.2, round(max(eu_demand["change (%)"])))
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Change_in_residential_grid_demand_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()
    


    merged = pd.merge(left=loads, right=national[["country", "demand MW", "year", "Hour"]], on=["year", "Hour", "country"])
    merged["demand_opt"] = merged["demand MW"] - merged["ref_grid_demand_stock_MW"] + merged["opt_grid_demand_stock_MW"]
    demand_grid = merged.groupby(["country", "year", "ID_EnergyPrice"])[["demand_opt", "demand MW"]].sum().reset_index()
    demand_grid["change_MW"] = demand_grid["demand_opt"] - demand_grid["demand MW"]
    eu_demand_grid = demand_grid.groupby(["year", "ID_EnergyPrice"])[["change_MW", "demand MW"]].sum().reset_index()
    eu_demand_grid["change (%)"] = eu_demand_grid["change_MW"] / eu_demand_grid["demand MW"] * 100
    eu_demand_grid["ID_EnergyPrice"] = eu_demand_grid["ID_EnergyPrice"].map(add_price_information())
    sns.barplot(
        data=eu_demand_grid,
        x="change (%)",
        y="year",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        orient="y"
    )
    plt.legend(loc="lower right")
    plt.xlabel("change in total electricity grid demand (%)")
    plt.xticks(rotation=0)
    plt.xlim(-0.4, 0.4)#(max(eu_demand_grid["change (%)"])))
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Change_in_total_grid_demand_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()
    

def plot_shifted_electricity(loads: pd.DataFrame):
    # calculate shifted electricity which is the difference between (ref-opt) if ref > opt and sum it up
    loads["shifted MW"] = loads["ref_grid_demand_stock_MW"] - loads["opt_grid_demand_stock_MW"]
    loads.loc[loads["shifted MW"] < 0, "shifted MW"] = 0

    shifted_df = loads.groupby(["year", "country", "ID_EnergyPrice"])[["shifted MW", "ref_grid_demand_stock_MW"]].sum().reset_index()
    shifted_df["shifted (%)"] = shifted_df["shifted MW"] / shifted_df["ref_grid_demand_stock_MW"] * 100
    
    x_order = shifted_df.groupby("country")["shifted (%)"].max().sort_values().index
    plot_df = shifted_df.copy()
    plot_df["ID_EnergyPrice"] = plot_df["ID_EnergyPrice"].map(add_price_information())

    plt.rcParams.update({
        'axes.labelsize': 16,    # x and y axis labels
        'axes.titlesize': 18,    # subplot titles
        'xtick.labelsize': 16,   # x-axis tick labels
        'ytick.labelsize': 16,   # y-axis tick labels
        'legend.fontsize': 16,   # legend text size
    })

    g = sns.FacetGrid(plot_df, col="year", col_wrap=2, height=5, aspect=0.5, sharex=True, sharey=True)
    # Map the barplot to each facet
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="shifted (%)",
        hue="ID_EnergyPrice",
        dodge=True,  # Ensures bars are grouped within x-axis categories
        hue_order=None, 
        order=x_order,
        palette=sns.color_palette()
    )

    g.set_axis_labels("Country", "Relative shifted electricity grid demand (%)")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Shifted_grid_demand_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

    plot_df = shifted_df.copy()
    plot_df["ID_EnergyPrice"] = plot_df["ID_EnergyPrice"].map(add_price_information())
    plot_df["shifted TWh"] = plot_df["shifted MW"] / 1_000 / 1_000
    order = plot_df.groupby("country")["shifted TWh"].max().sort_values().index
    g = sns.FacetGrid(plot_df, col="year", col_wrap=2, height=5, aspect=0.5, sharex=True, sharey=True)
    # Map the barplot to each facet
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="shifted TWh",
        hue="ID_EnergyPrice",
        dodge=True,  # Ensures bars are grouped within x-axis categories
        hue_order=None, 
        order=order,
        palette=sns.color_palette()
    )

    g.set_axis_labels("Country", "Absolute amount of electricity grid demand \n shifted within a year (TWh)")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Shifted_absolute_grid_demand_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

    plot_df = shifted_df.groupby(["year", "ID_EnergyPrice"])["shifted MW"].sum().reset_index().copy()
    plot_df["ID_EnergyPrice"] = plot_df["ID_EnergyPrice"].map(add_price_information())
    plot_df["shifted TWh"] = plot_df["shifted MW"] / 1_000 / 1_000
    sns.barplot(
        data=plot_df,
        x="year",
        y="shifted TWh",
        hue="ID_EnergyPrice",
        palette=sns.color_palette()
    )
    plt.legend(title="", loc="upper left")
    plt.xlabel("year")
    plt.ylabel("shifted electricity in the EU in TWh")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Shifted_EU_absolute_grid_demand_coling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

def plot_values_on_EU_map(plot_df: pd.DataFrame, columns2plot: str, y_label: str) -> list:
    """
    This function takes a df that has to have a column which represents the countries (3 digits), 
    and the year. 

    plot_df: pd.DataFrame
    columns2plot: str name of the column that should be plotted as countries
    y_label: str name that should be displayed next to the colorbar describing the data displayed
    returns a list of the figures created for each year in case they should be put on a single plot later
    """
    world = gpd.read_file(SAVING_PATH.parent / "gpd_data" / "110m_cultural" / "ne_110m_admin_0_countries.shp")
    c_list = [key for key, value in Cp.EUROPEAN_COUNTRIES.items()]
    europe = world[world["ADM0_A3"].isin(c_list)].rename(columns={"ADM0_A3": "country"})
    merged = gpd.GeoDataFrame(pd.merge(left=plot_df, right=europe[["country", "geometry"]], on="country"))
    
    years = plot_df["year"].unique()
    vmin, vmax = round(merged[columns2plot].min()), round(merged[columns2plot].max())
    plt.rcParams.update({
        'axes.labelsize': 20,    # x and y axis labels
        'axes.titlesize': 20,    # subplot titles
        'xtick.labelsize': 18,   # x-axis tick labels
        'ytick.labelsize': 18,   # y-axis tick labels
        'legend.fontsize': 18,   # legend text size
    })
    fig, axes = plt.subplots(nrows=1, ncols=len(years), figsize=(12, 6))
    for ax, year in zip(axes, years):
        df = merged.loc[merged["year"]==year, :].copy()
        df.boundary.plot(ax=ax, linewidth=1, edgecolor="black")  # Country borders
        df.plot(
            column=columns2plot, 
            cmap="coolwarm", 
            edgecolor="black", 
            linewidth=0.5,
            legend=True, 
            ax=ax, 
            missing_kwds={"color": "white", "label": "No Data"},
            legend_kwds={"label": y_label, "orientation": "vertical"},
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(-12, 32)
        ax.set_ylim(32, 72)
        ax.set_xticks([],)
        ax.set_yticks([],)
        ax.set_title(f"{year}")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"EU_map_{y_label.replace(' ', '_').replace('cent/kWh','').replace('(HP/GWh)','').replace('(kWh/m2)','').replace('(kWh/m$^2$)','').replace('(kWh/m^2)','').replace('(kWh/(m$^2$*year))','')}.svg")
    # plt.show()
    plt.close()

    import matplotlib.colors as colors
    cmap = plt.get_cmap("coolwarm")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    color_codes = {}

    for year in years:
        # Filter the DataFrame for the given year
        df_year = merged.loc[merged["year"] == year, ["country", columns2plot]].dropna()
        df_year_sorted = df_year.sort_values(by=columns2plot)
        # Map values to normalized range [0,1]
        normalized_values = norm(df_year_sorted[columns2plot].values)

        # Convert normalized values to RGBA
        rgba_colors = cmap(normalized_values)

        # Convert RGBA to HEX
        hex_colors = [colors.rgb2hex(rgba) for rgba in rgba_colors]

        # Save in dictionary
        color_codes[year] = dict(zip(df_year_sorted["country"], hex_colors))
    return color_codes
        


def create_demand_peaks_for_different_prosumager_shares(df: pd.DataFrame):
    shares = np.arange(0, 1.1, 0.1)
    for s in shares:
        df[f"demand_{round(s,1)}_share_prosumagers"] = df["demand MW"] - df["ref_grid_demand_stock_MW"] * s + df["opt_grid_demand_stock_MW"] * s
    demand_columns = [f"demand_{round(s,1)}_share_prosumagers" for s in shares]
    demand_peaks = df.groupby(["year", "country", "ID_EnergyPrice"])[["demand MW"] + demand_columns].max().reset_index()
    peak_columns = [round(s,1) for s in shares]
    demand_peaks[peak_columns] = demand_peaks[demand_columns].sub(demand_peaks["demand MW"], axis=0).div(demand_peaks["demand MW"], axis=0) * 100
    demand_peaks.drop(columns=["demand MW"], inplace=True)
    demand_peaks = demand_peaks.loc[demand_peaks["year"]!=2020]

    unique_countries = demand_peaks['country'].unique()
    df_long = demand_peaks.melt(
        id_vars=['year', 'country', 'ID_EnergyPrice'],
        value_vars=peak_columns,
        var_name='prosumager share (%)',
        value_name='peak increase (%)'
    )

    # highlight_countries = df_long.loc[(df_long["peak increase (%)"]<0) & (df_long["year"]!=2020)]["country"].unique()
    palette = {country: "darkgrey" for country in unique_countries}
    plt.rcParams.update({'font.size': 16})
    g = sns.FacetGrid(df_long, 
                      row="year", 
                      height=4, 
                      sharey=True, 
                      col="ID_EnergyPrice")
    g.map_dataframe(sns.lineplot, x="prosumager share (%)", y="peak increase (%)", hue="country",
                    estimator=None, units="ID_EnergyPrice", alpha=0.9,  legend=False, palette=palette)

    # Add axis labels and titles
    g.set_axis_labels("Prosumager Share (%)", "Peak Increase (%)")
    
    g.set_titles("{col_name}")
    i = 0
    for ax in g.axes.flat:
        title = ax.get_title()
        parts = title.split("|")
        if i == 0:
            calculated_value=0
            col_part = parts[-1].strip()
            i = 1
        else:
            col_part = parts[-1].strip()
            last_digit = float(str(col_part)[-1])
            calculated_value = 2**(last_digit-2) * 5 
        new_col_title = f"{round(calculated_value)} cent/kWh"
        ax.set_title(f"{new_col_title}")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))

    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Peak_increase_price_over_prosumager_share.svg")
    plt.show()
    plt.close()

def add_price_information()->dict:
    """returns a dict to map to the ID_Energyprice"""
    return {
        "Price 1": "0 cent/kWh grid fees",
        "Price 2": "5 cent/kWh grid fees",
        "Price 3": "10 cent/kWh grid fees",
        "Price 4": "20 cent/kWh grid fees",
        "Price 5": "40 cent/kWh grid fees",

    }

def analyse_peak_demand(loads: pd.DataFrame, national: pd.DataFrame):
    merged = pd.merge(left=loads, right=national[["country", "demand MW", "year", "Hour"]], on=["year", "Hour", "country"])
    merged["demand_opt"] = merged["demand MW"] - merged["ref_grid_demand_stock_MW"] + merged["opt_grid_demand_stock_MW"]
    demand_peaks = merged.groupby(["year", "country", "ID_EnergyPrice"])[["demand MW", "demand_opt"]].max().reset_index()
    demand_peaks["peak increase"] = (demand_peaks["demand_opt"] - demand_peaks["demand MW"]) / demand_peaks["demand MW"] * 100  # %

    # the std in electricity price and number of heat pumps per country showed to have the highest correlytion with the raise in peak demand
    price_std = loads.groupby(["ID_EnergyPrice", "year", "country"])["price (cent/kWh)"].std().reset_index()

    hp_numbers = Cp.create_number_of_HPs_df(COOLING_PERCENTAGE)
    national_demand = national.groupby(["country", "year"])["demand MW"].sum().reset_index()
    national_demand["year"] = national_demand["year"].astype(int)
    hp = pd.merge(left=hp_numbers, right=national_demand, on=["country", "year"])
    hp["number HPs per national demand"] = hp["number of HPs in country"] / hp["demand MW"] * 1_000

    create_demand_peaks_for_different_prosumager_shares(df=merged)
    
    price_figs = plot_values_on_EU_map(price_std.loc[price_std["ID_EnergyPrice"]=="Price 1", :], columns2plot="price (cent/kWh)", y_label="standard deviation of the price profiles (cent/kWh)")
    hp_figs = plot_values_on_EU_map(hp.loc[hp["ID_EnergyPrice"]=="Price 1", :], columns2plot="number HPs per national demand", y_label="# HPs per national demand (HP/GWh)")
    # peak_figs = plot_values_on_EU_map(demand_peaks.loc[demand_peaks["ID_EnergyPrice"]=="Price 2", :], columns2plot="peak increase", y_label="change in national peak demand (%)")

    plot_df = demand_peaks.copy()
    plot_df["ID_EnergyPrice"] = plot_df["ID_EnergyPrice"].map(add_price_information())
    plt.figure(figsize=(10, 8))
    sns.boxplot(
        data=plot_df,
        x="peak increase",
        y="year",
        hue="ID_EnergyPrice",
        orient="y",
        palette=sns.color_palette()
    )
    plt.legend()
    plt.xlabel("national peak demand increase through prosumager in EU (%)")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Peak_demand_increase_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()

    x_order = demand_peaks.groupby("country")["peak increase"].mean().sort_values().index
    plt.figure(figsize=(15,8))
    g = sns.FacetGrid(plot_df, col="year", col_wrap=2, height=8, aspect=1, sharex=True, sharey=True)
    # Map the barplot to each facet
    g.map_dataframe(
        sns.barplot,
        x="country",
        y="peak increase",
        hue="ID_EnergyPrice",
        dodge=True,  # Ensures bars are grouped within x-axis categories
        hue_order=None, 
        order=x_order,
        palette=sns.color_palette(),
    )

    g.set_axis_labels("Country", "Peak demand increase through prosumager (%)")
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Show the plot
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Peak_demand_increase_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()


    merged["day"] = (merged["Hour"]-1) // 24 + 1
    peak_to_peak_ref = merged.groupby(["year", "country", "ID_EnergyPrice", "day"])["demand MW"].agg(lambda x: x.max() - x.min()).reset_index().rename(columns={"demand MW": "peak to peak ref"})
    peak_to_peak_opt = merged.groupby(["year", "country", "ID_EnergyPrice", "day"])["demand_opt"].agg(lambda x: x.max() - x.min()).reset_index().rename(columns={"demand_opt": "peak to peak opt"})
    peak_merge = pd.merge(left=peak_to_peak_ref, right=peak_to_peak_opt, on=["year", "country", "day", "ID_EnergyPrice"])
    peak_merge["peak to peak change"] = peak_merge["peak to peak opt"] - peak_merge["peak to peak ref"]
    peak_merge["peak to peak change (%)"] = peak_merge["peak to peak change"] / peak_merge["peak to peak ref"] * 100
    
    peak_merge["ID_EnergyPrice"] = peak_merge["ID_EnergyPrice"].map(add_price_information())
    g = sns.FacetGrid(peak_merge, col="year", col_wrap=2, height=8, aspect=1, sharex=True, sharey=False)
    order = peak_merge.groupby("country")["peak to peak change (%)"].mean().sort_values().index
    g.map_dataframe(
        sns.boxplot,
        x="country",
        y="peak to peak change (%)",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        order=order,
        showfliers=False
    )
    for ax in g.axes.flat:
        ax.set_xticklabels(order, rotation=90)
    g.add_legend(
        title="",
        loc='lower center',  # centered horizontally
        bbox_to_anchor=(0.5, 1.05),  # positioned above the plot
        ncol=5,
        frameon=False
    )
    g.set_axis_labels("Country", "Daily peak to peak demand change (%)")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Peak_to_peak_demand_change_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()

    plt.figure(figsize=(10,8))
    sns.boxplot(
        data=peak_merge,
        x="peak to peak change (%)",
        y="year",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        orient="y",
        showfliers=False
    )
    plt.legend()
    plt.xlabel("daily peak to peak demand change (%)")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Peak_to_peak_demand_change_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()

    
def mean_absolute_successive_difference(time_series):
    return  np.mean(np.abs(np.diff(time_series)))   
    
def analyse_prices(loads: pd.DataFrame, national: pd.DataFrame):
    def frequency_of_changes(group):
        diff = group.diff().dropna()
        signs = np.sign(diff)
        sign_changes = signs.diff().fillna(0) != 0
        return np.sum(sign_changes)
    def local_peaks(group_df, prominence=0):
        peaks, properties = find_peaks(group_df, prominence=prominence)
        n_peaks = len(peaks)
        avg_prom = properties["prominences"].mean() if n_peaks > 0 else 0
        return pd.Series({
            'peaks': n_peaks,
            'prominence': avg_prom
        })
     
    price_props = loads.groupby(["ID_EnergyPrice", "year", "country"])["price (cent/kWh)"].agg(
        ['mean', 'std', 'max', 'min', frequency_of_changes, mean_absolute_successive_difference]
        ).reset_index()
    
    local_peaks = loads.groupby(["ID_EnergyPrice", "year", "country"])["price (cent/kWh)"].apply(local_peaks).reset_index().rename(columns={"level_3": "type", "price (cent/kWh)": "values"})
    merged_price_props = pd.merge(left=local_peaks, right=price_props.loc[:, ["ID_EnergyPrice", "year", "country", "frequency_of_changes", "mean"]], on=["ID_EnergyPrice", "year", "country"])
    merged_price_props["prominence_per_mean"] = merged_price_props["values"] / merged_price_props["mean"]
    
    plt.rcParams.update({
        'axes.labelsize': 18,    # x and y axis labels
        'axes.titlesize': 20,    # subplot titles
        'xtick.labelsize': 18,   # x-axis tick labels
        'ytick.labelsize': 18,   # y-axis tick labels
        'legend.fontsize': 18,   # legend text size
    })
    fig = plt.figure(figsize=(8,6))
    sns.barplot(
        price_props.loc[(price_props["ID_EnergyPrice"]=="Price 4"), :],
        x="country",
        y="frequency_of_changes",
        hue="year",
        palette=sns.color_palette(),
        order=price_props.loc[(price_props["ID_EnergyPrice"]=="Price 4"), :].groupby("country")["frequency_of_changes"].mean().sort_values().index,
    )
    plt.ylabel("frequency of price changes per year (-)")
    plt.legend(title="")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Price_frequency_of_changes_cooling{COOLING_PERCENTAGE}.svg")

    order=local_peaks.loc[(local_peaks["type"]=="prominence") & (local_peaks["ID_EnergyPrice"]=="Price 4"), :].groupby("country")["values"].mean().sort_values().index
    fig = plt.figure(figsize=(8,6))
    sns.barplot(
        local_peaks.loc[(local_peaks["type"]=="prominence") & (local_peaks["ID_EnergyPrice"]=="Price 4"), :],
        x="country",
        y="values",
        hue="year",
        palette=sns.color_palette(),
        order=order
    )
    plt.ylabel("average prominence of peaks \n per year (cent/kWh)")
    plt.legend(title="")
    plt.xticks(rotation=90)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Price_peak_prominence_cooling{COOLING_PERCENTAGE}.svg")
    print(f'local price peaks change from 2030 to 2050 by {local_peaks.loc[local_peaks["ID_EnergyPrice"]=="Price 4", :].groupby("year")["values"].sum().reset_index().pct_change().dropna()["values"].values[0]*100} %')

    merged_price_props["ID_EnergyPrice"] = merged_price_props["ID_EnergyPrice"].map(add_price_information())
    fig, axes = plt.subplots(figsize=(16,6), nrows=1, ncols=2, sharex=True, sharey=True)

    sns.scatterplot(
        data=merged_price_props.loc[(merged_price_props["type"]=="prominence") & (merged_price_props["year"]==2030), :],
        x="country",
        y="prominence_per_mean",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        style="ID_EnergyPrice",
        legend=True,
        ax=axes[0],
    )
    sns.scatterplot(
        data=merged_price_props.loc[(merged_price_props["type"]=="prominence") & (merged_price_props["year"]==2050), :],
        x="country",
        y="prominence_per_mean",
        hue="ID_EnergyPrice",
        palette=sns.color_palette(),
        style="ID_EnergyPrice",
        legend=False,
        ax=axes[1],
    )   
    axes[0].legend(title="")
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel("average prominence of price peaks \n over mean price per year (-)", fontsize=18)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.set_ylim(0, 3)

    axes[0].set_title("2030")
    axes[1].set_title("2050")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Price_peak_prominence_over_mean_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()

    def plot_price_prop(column_name, price_df, y_label, hue, ):
        order = price_df.groupby("country")[column_name].mean().sort_values().index
        g = sns.FacetGrid(price_df, col="year", col_wrap=2, height=5, aspect=1.5, sharex=True, sharey=True)
        # Map the barplot to each facet
        g.map_dataframe(
            sns.barplot,
            x="country",
            y=column_name,
            hue="ID_EnergyPrice",
            dodge=True,  # Ensures bars are grouped within x-axis categories
            hue_order=None, 
            order=order,
            palette=sns.color_palette()
        )

        # Adjust plot aesthetics
        g.set_axis_labels("Country", f"{y_label}")
        g.add_legend(loc="upper center")
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Show the plot
        plt.tight_layout()
        plt.savefig(SAVING_PATH / f"Price_analysis_{column_name}_cooling{COOLING_PERCENTAGE}.svg")
        plt.close()
    
    plot_price_prop("mean", price_props, "mean price (cent/kWh)")
    plot_price_prop("std" , price_props, "standard deviation of the price (cent/kWh)")
    plot_price_prop("mean_absolute_successive_difference", price_props, "MSSD of electricity price (cent/kWh)")

    price_props_4 = price_props.loc[price_props["ID_EnergyPrice"]=="Price 4", :]
    plot_values_on_EU_map(price_props_4, columns2plot="mean_absolute_successive_difference", y_label="MSSD (cent/kWh)")
    # plot_price_prop("max")
    # plot_price_prop("min")

    fig = plt.figure(figsize=(8,6))
    sns.barplot(
        price_props.loc[price_props["ID_EnergyPrice"]=="Price 4", :],
        x="country",
        y="frequency_of_changes",
        hue="year",
        palette=sns.color_palette(),
        order=price_props.loc[price_props["ID_EnergyPrice"]=="Price 2", :].groupby("country")["frequency_of_changes"].mean().sort_values().index
    )
    plt.ylabel("frequency of prices changing per year (-)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Price_analysis_frequency_change_cooling{COOLING_PERCENTAGE}.svg")

    print(f'price frequency changes from 2030 to 2050 by {price_props.loc[price_props["ID_EnergyPrice"]=="Price 2", :].groupby("year")["frequency_of_changes"].sum().reset_index().pct_change().dropna()["frequency_of_changes"].values[0]*100} %')

    # plot the daily mean price profile:
    daily_mean_price = loads[["Hour", "price (cent/kWh)", "ID_EnergyPrice", "year", "country"]].copy()
    daily_mean_price = daily_mean_price.loc[daily_mean_price["ID_EnergyPrice"]=="Price 4", :]
    daily_mean_price["day hour"] = (daily_mean_price["Hour"]-1) % 24 + 1
    daily_mean_price = daily_mean_price.groupby(["year", "country", "day hour"])["price (cent/kWh)"].median().reset_index()

    g = sns.FacetGrid(daily_mean_price, col="year", col_wrap=2, sharey=True, sharex=True, height=6, aspect=1)
    g.map_dataframe(
        sns.lineplot,
        x="day hour",
        y="price (cent/kWh)",
        hue="country",
        linewidth = 0.5,
        
    )
    plt.ylabel("median hourly electricity price per day (cent/kWh)")
    plt.xlabel("hour of the day")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Daily_median_price_all_EU_countries.svg")
    plt.close()

def calculate_price_correlations(loads: pd.DataFrame, national: pd.DataFrame):
    input_variables = []
    output_variables = []

    merged = pd.merge(left=loads, right=national[["country", "demand MW", "year", "Hour"]], on=["year", "Hour", "country"])
    merged["demand_opt"] = merged["demand MW"] - merged["ref_grid_demand_stock_MW"] + merged["opt_grid_demand_stock_MW"]
    demand_peaks = merged.groupby(["year", "country", "ID_EnergyPrice"])[["demand MW", "demand_opt"]].max().reset_index()
    demand_peaks["national demand peak increase (%)"] = (demand_peaks["demand_opt"] - demand_peaks["demand MW"]) / demand_peaks["demand MW"] * 100  # %
    demand_peaks = demand_peaks[["country", "year", "ID_EnergyPrice", "national demand peak increase (%)"]].copy()
    output_variables.append("national demand peak increase (%)")

    # national demand
    national_demand = merged.groupby(["year", "country", "ID_EnergyPrice"])["demand MW"].sum().reset_index()
    national_demand.rename(columns={"demand MW": "national demand MWh"}, inplace=True)
    merged2 = pd.merge(left=demand_peaks, right=national_demand, on=["country", "year", "ID_EnergyPrice"])
    input_variables.append("national demand MWh")

    # load factor
    groups = loads.groupby(["year", "country", "ID_EnergyPrice"])
    factor_ref = groups[["ref_grid_demand_stock_MW", "price (cent/kWh)"]].apply(
        lambda g: (g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "ref_grid_demand_stock_MW"].sum() - g.loc[g["price (cent/kWh)"] >=g["price (cent/kWh)"].quantile(0.75), "ref_grid_demand_stock_MW"].sum()) / (g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "ref_grid_demand_stock_MW"].sum() + g.loc[g["price (cent/kWh)"] >=g["price (cent/kWh)"].quantile(0.75), "ref_grid_demand_stock_MW"].sum())
    ).reset_index(name="flexibility factor reference")
    factor_opt = groups[["opt_grid_demand_stock_MW", "price (cent/kWh)"]].apply(
        lambda g: (g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "opt_grid_demand_stock_MW"].sum() - g.loc[g["price (cent/kWh)"] >=g["price (cent/kWh)"].quantile(0.75), "opt_grid_demand_stock_MW"].sum()) / (g.loc[g["price (cent/kWh)"] <= g["price (cent/kWh)"].quantile(0.25), "opt_grid_demand_stock_MW"].sum() + g.loc[g["price (cent/kWh)"] >=g["price (cent/kWh)"].quantile(0.75), "opt_grid_demand_stock_MW"].sum())
    ).reset_index(name="flexibility factor prosumager")
    load_factor = pd.merge(right=factor_ref, left=factor_opt, on=["year", "country", "ID_EnergyPrice"])
    load_factor["flexibility factor change"] = load_factor["flexibility factor prosumager"] - load_factor["flexibility factor reference"]
    load_factor = load_factor[["country", "year", "ID_EnergyPrice", "flexibility factor reference", "flexibility factor prosumager", "flexibility factor change"]].copy()
    merged3 = pd.merge(left=merged2, right=load_factor, on=["country", "year", "ID_EnergyPrice"])
    [output_variables.append(x) for x in ["flexibility factor reference", "flexibility factor prosumager", "flexibility factor change"]]

    # change in total elec demand of prosumagers
    demand = loads.groupby(["country", "year", "ID_EnergyPrice"])[["opt_grid_demand_stock_MW", "ref_grid_demand_stock_MW"]].sum().reset_index()
    demand["prosumager demand change (%)"] = (demand["opt_grid_demand_stock_MW"] - demand["ref_grid_demand_stock_MW"]) / demand["ref_grid_demand_stock_MW"] * 100  #%
    demand = demand[["country", "year", "ID_EnergyPrice", "prosumager demand change (%)"]].copy()
    merged4 = pd.merge(left=merged3, right=demand, on=["country", "year", "ID_EnergyPrice"])
    output_variables.append("prosumager demand change (%)")

    # price properties
    loads['avg price change magnitude'] = loads['price (cent/kWh)'].diff().abs()
    price_props = loads.groupby(["ID_EnergyPrice", "year", "country"])["price (cent/kWh)"].agg(['mean', 'std', 'max', 'min', 'var']).reset_index()
    price_props.rename(columns={"mean": 'price mean', 'std': "price std" , 'max': "price max", 'min': "price min", 'var': "price var"}, inplace=True)
    p = loads.groupby(["ID_EnergyPrice", "year", "country"])["avg price change magnitude"].mean().reset_index()
    price = pd.merge(left=price_props, right=p, on=["country", "year", "ID_EnergyPrice"])
    copy_loads = loads.copy()
    copy_loads["day"] = (copy_loads["Hour"]-1) // 24 + 1
    copy_loads = copy_loads.groupby(["ID_EnergyPrice", "year", "country", "day"])["price (cent/kWh)"].agg(["max", "min"]).reset_index()
    copy_loads["avg daily peak to peak"] = copy_loads["max"] - copy_loads["min"]
    peak_to_peak = copy_loads.groupby(["ID_EnergyPrice", "year", "country"])["avg daily peak to peak"].mean().reset_index()
    price = pd.merge(left=price, right=peak_to_peak, on=["country", "year", "ID_EnergyPrice"])
    merged5 = pd.merge(left=merged4, right=price, on=["country", "year", "ID_EnergyPrice"])
    [input_variables.append(x) for x in ['price mean', 'price std', 'price max', 'price min', 'price var', 'avg daily peak to peak']];
    
    # correlation with outside temperature,  number HPs, heat demand kWh/m2, nr heat pump / total elec demand on country
    heat_demand = Cp.create_national_heat_demand_df(percentage_cooling=COOLING_PERCENTAGE)
    heat_demand["ID_EnergyPrice"] = heat_demand["ID_EnergyPrice"].map({1: "Price 1", 2: "Price 2"})
    heat_demand["year"] = heat_demand["year"].astype(int)
    merged6 = pd.merge(left=merged5, right=heat_demand, on=["country", "year", "ID_EnergyPrice"])
    [output_variables.append(x) for x in ["Heating ref demand (kWh/m2)", "Heating opt demand (kWh/m2)"]];

    hp_numbers = Cp.create_number_of_HPs_df(COOLING_PERCENTAGE)
    merged7 = pd.merge(left=merged6, right=hp_numbers, on=["country", "year", "ID_EnergyPrice"])
    input_variables.append("number of HPs in country")

    # number of HPs per national demand
    merged7["number HPs per national demand"] = merged7["number of HPs in country"] / merged7["national demand MWh"]
    input_variables.append("number HPs per national demand")

    # include the GSC relative
    gsc = calculate_GSC(loads=loads)
    gsc = gsc.loc[gsc["ID_EnergyPrice"]!="reference", :]
    merged8 = pd.merge(left=merged7, right=gsc, on=["country", "year", "ID_EnergyPrice"])
    output_variables.append("GSC_rel")

    # include outside temperature
    temp_df = Cp.create_national_temperature_df()
    mean_temp = temp_df.groupby(["country","year"])["temperature"].mean().reset_index()
    mean_temp["year"] = mean_temp["year"].astype(int)
    merged9 = pd.merge(left=merged8, right=mean_temp, on=["country", "year"])
    input_variables.append("temperature")

    # include change in peak to peak demand
    merged["day"] = (merged["Hour"]-1) // 24 + 1
    peak_to_peak_ref = merged.groupby(["year", "country", "ID_EnergyPrice", "day"])["demand MW"].agg(lambda x: x.max() - x.min()).reset_index().rename(columns={"demand MW": "peak to peak ref"})
    peak_to_peak_opt = merged.groupby(["year", "country", "ID_EnergyPrice", "day"])["demand_opt"].agg(lambda x: x.max() - x.min()).reset_index().rename(columns={"demand_opt": "peak to peak opt"})
    peak_merge = pd.merge(left=peak_to_peak_ref, right=peak_to_peak_opt, on=["year", "country", "day", "ID_EnergyPrice"])
    peak_merge["peak to peak change"] = peak_merge["peak to peak opt"] - peak_merge["peak to peak ref"]
    peak_merge["daily peak to peak change (%)"] = peak_merge["peak to peak change"] / peak_merge["peak to peak ref"] * 100
    merged10 = pd.merge(left=merged9, right=peak_merge[["country", "year", "ID_EnergyPrice", "daily peak to peak change (%)"]], on=["country", "year", "ID_EnergyPrice"])
    output_variables.append("daily peak to peak change (%)")
    
    # inlcude daily load factor
    load_factor_opt = pd.DataFrame(merged.groupby(["country", "year", "ID_EnergyPrice", "day"])["demand_opt"].mean() / merged.groupby(["country", "year", "ID_EnergyPrice", "day"])["demand_opt"].max())
    load_factor_opt.reset_index(inplace=True)
    load_factor_opt.rename(columns={"demand_opt": "load factor opt"}, inplace=True)

    load_factor_ref = pd.DataFrame(merged.groupby(["country", "year", "ID_EnergyPrice", "day"])["demand MW"].mean() / merged.groupby(["country", "year", "ID_EnergyPrice", "day"])["demand MW"].max())
    load_factor_ref.reset_index(inplace=True)
    load_factor_ref = load_factor_ref[load_factor_ref["ID_EnergyPrice"]=="Price 1"].copy()
    load_factor_ref.rename(columns={"demand MW": "load factor ref"}, inplace=True)
    load_factor = pd.merge(left=load_factor_opt, right=load_factor_ref, on=["country", "year", "ID_EnergyPrice", "day"])
    load_factor = load_factor.groupby(["country", "year", "ID_EnergyPrice"])[["load factor opt", "load factor ref"]].mean().reset_index()  # take the mean over all days
    merged11 = pd.merge(left=merged10, right=load_factor, on=["country", "year", "ID_EnergyPrice"])
    output_variables.append("load factor opt")
    output_variables.append("load factor ref")

    # normalize
    # numeric_columns = ['mean', 'std', 'max', 'min', 'national demand peak increase (%)', "prosumager demand change (%)", "flexibility factor reference", "flexibility factor prosumager", "flexibility factor change"]
    # df[numeric_columns] = df[numeric_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    correlation_results = pd.DataFrame()
    # Loop over columns and calculate correlation with target columns
    for column in output_variables:
        # Calculate correlation between each column and all target columns
        correlation_results[column] = merged11[input_variables].apply(lambda x: x.corr(merged11[column]))
    
    sns.heatmap(
        data=correlation_results,
        cmap=sns.color_palette("Spectral_r", as_cmap=True),
        vmin=-1,
        vmax=1,
        annot=False,
    )

    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Heatmap_correlation_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()


    # number of HPs per national demand
    n_order = merged7.groupby("country")["number HPs per national demand"].mean().sort_values().index
    sns.barplot(
        data=merged7,
        x="country",
        y="number HPs per national demand",
        palette=sns.color_palette(),
        hue="year",
        order=n_order
    )
    plt.ylabel("number of heat pumps divided by national demand ([-]/MWh)")
    plt.xticks(rotation=90)
    plt.savefig(SAVING_PATH / f"Number_HPs_per_national_demand_cooling{COOLING_PERCENTAGE}.svg")
    plt.close()


    numbers_order = hp_numbers.groupby("country")["number of HPs in country"].mean().sort_values().index
    hp_numbers["number of HPs in country"] = hp_numbers["number of HPs in country"] / 1_000_000
    sns.barplot(
        data=hp_numbers,
        x="country",
        y="number of HPs in country",
        hue="year",
        palette=sns.color_palette(),
        order=numbers_order
    )   
    plt.ylabel("number of heat pump heated buildings (mio.)")
    plt.legend(title="")
    plt.xticks(rotation=90)
    plt.savefig(SAVING_PATH / f"number_of_HPs.svg")
    plt.close()

    # plot HP capacity over national demand


    heat_order = heat_demand.groupby("country")["Heating ref demand (kWh/m2)"].mean().sort_values().index
    sns.barplot(
        data=heat_demand,
        x="country",
        y="Heating ref demand (kWh/m2)",
        hue="year",
        palette=sns.color_palette(),
        order=heat_order
    )   
    plt.ylabel("average specific heating demand (kWh/m$^2$")
    plt.xticks(rotation=90)
    plt.savefig(SAVING_PATH / f"heat_demand_avg.svg")
    plt.close()

    temperature_order = mean_temp.groupby("country")["temperature"].mean().sort_values().index
    sns.barplot(
        data=mean_temp,
        x="country",
        y="temperature",
        palette=sns.color_palette(),
        hue="country",
        order=temperature_order
    )   
    plt.ylabel("average yearly temperature (°C)")
    plt.xticks(rotation=90)
    plt.savefig(SAVING_PATH / f"temperature_avg.svg")
    plt.close()
    
def compare_heat_demand_with_invert_2020():
    df = Cp.read_ref_2020_national_heat_demand(percentage_cooling=0.1)
    df["year"] = df["year"].astype(int)
    df["country"] = df["country"].astype(str)
    df["ID_EnergyPrice"] = df["ID_EnergyPrice"].astype(int)
    df = df.loc[df["ID_EnergyPrice"]==1, :].copy()
    df["source"] = "FLEX"
    
    invert = pd.read_csv(SAVING_PATH.parent / "hwb_Invert_HP_heated_buildings.csv")
    invert = invert.loc[invert["year"]==2020, :].copy()
    invert["source"] = "Invert"
    invert.rename(columns={"avg_hwb": "Heating ref demand (kWh/m2)"}, inplace=True)

    merged = pd.concat([df, invert], axis=0).reset_index(drop=True)
    sns.barplot(
        data=merged,
        x="country",
        y="Heating ref demand (kWh/m2)",
        hue="source",
        palette=sns.color_palette()
    )
    plt.ylabel("average specific heating demand (kWh/m$^2$)")
    plt.xticks(rotation=90)
    plt.savefig(SAVING_PATH / f"comparison_heat_demand_2020.svg")
    plt.close()

def plot_daily_load_factor(loads: pd.DataFrame, national: pd.DataFrame):
    df_national = pd.merge(left=national, right=loads, on=["country", "year", "Hour"])
    df_national["day"] = (df_national["Hour"]-1) // 24 + 1
    df_national["demand opt"] = df_national["opt_grid_demand_stock_MW"] - df_national["ref_grid_demand_stock_MW"] + df_national["demand MW"]
    
    def calculate_load_factor(df: pd.DataFrame, opt_column: str, ref_column: str):
        # load factor is the mean demand divided by the max demand in a day
        df["day"] = (df["Hour"]-1) // 24 + 1
        load_factor_opt = pd.DataFrame(df.groupby(["country", "year", "ID_EnergyPrice", "day"])[opt_column].mean() / df.groupby(["country", "year", "ID_EnergyPrice", "day"])[opt_column].max())
        load_factor_opt.reset_index(inplace=True)
        load_factor_opt.rename(columns={opt_column: "load factor"}, inplace=True)

        load_factor_ref = pd.DataFrame(df.groupby(["country", "year", "ID_EnergyPrice", "day"])[ref_column].mean() / df.groupby(["country", "year", "ID_EnergyPrice", "day"])[ref_column].max())
        load_factor_ref.reset_index(inplace=True)
        load_factor_ref = load_factor_ref[load_factor_ref["ID_EnergyPrice"]=="Price 1"].copy()
        load_factor_ref.rename(columns={ref_column: "load factor"}, inplace=True)
        load_factor_ref.loc[:, "ID_EnergyPrice"] = "reference"

        merged = pd.concat([load_factor_opt, load_factor_ref], axis=0).reset_index(drop=True)
        return merged
    
    merged = calculate_load_factor(df=loads, opt_column="opt_grid_demand_stock_MW", ref_column="ref_grid_demand_stock_MW")
    merged_national = calculate_load_factor(df=df_national, opt_column="demand opt", ref_column="demand MW")
    order = merged.groupby("country")["load factor"].mean().sort_values().index
    g = sns.FacetGrid(merged, col="ID_EnergyPrice", col_wrap=3, height=5, aspect=1.5, sharex=True, sharey=True)
    g.map_dataframe(
        sns.boxplot,
        x="country",
        y="load factor",
        hue="year",
        palette=sns.color_palette(),
        order=order,
        showfliers=False
    )
    plt.ylabel("daily load factor")
    for ax in g.axes.flat:
        ax.set_xticklabels(list(order), rotation=90)
    g.set_titles("{col_name}")
    g.add_legend(title="year")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"daily_load_factor_residential.svg")
    plt.close()

    # plot it on eu level by just taking the daily data of all countries
    sns.boxplot(
        data=merged,
        x="load factor",
        y="ID_EnergyPrice",
        palette=sns.color_palette(),
    )
    plt.xlabel("daily load factor of residential demand across countries")
    plt.ylabel("energy price scenario")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"daily_load_factor_EU_residential.svg")
    plt.close()

    # load factor for total demand
    g = sns.FacetGrid(merged_national, col="ID_EnergyPrice", col_wrap=3, height=5, aspect=1.5, sharex=True, sharey=True)
    g.map_dataframe(
        sns.boxplot,
        x="country",
        y="load factor",
        hue="year",
        palette=sns.color_palette(),
        order=order,
        showfliers=False
    )
    plt.ylabel("daily load factor")
    for ax in g.axes.flat:
        ax.set_xticklabels(list(order), rotation=90)
    g.set_titles("{col_name}")
    g.add_legend(title="year")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"daily_load_factor_total_demand.svg")
    plt.close()

    sns.boxplot(
        data=merged_national,
        x="load factor",
        y="ID_EnergyPrice",
        palette=sns.color_palette(),
    )
    plt.xlabel("daily load factor of total demand across countries")
    plt.ylabel("energy price scenario")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"daily_load_factor_EU_total_demand.svg")
    plt.close()

def plot_country_specific_heating_demand(percentage_cooling):
    df = Cp.create_national_heat_demand_df(percentage_cooling=percentage_cooling)
    df = df.loc[df["ID_EnergyPrice"]==1, :].copy()

    plot_values_on_EU_map(plot_df=df, columns2plot="Heating ref demand (kWh/m2)", y_label="specific heating demand (kWh/(m$^2$*year))")

def plot_PV_installations_and_size_distribution():
    path2_projects = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects")

    projects = [p for p in path2_projects.glob("*") if p.is_dir() and len(p.name)==18 and "analysis" not in p.name and "Murcia" not in p.name] 
    invert_files = [f / "input" /  f"INVERT_{f.name.replace(f'_{PROJECT_ACRONYM}','')}.csv" for f in projects]
    dfs = []
    for file in invert_files:
        if "MLT" in file.name or "CYP" in file.name:
            continue
        df = pd.read_csv(file)
        pv_columns = [c for c in df.columns if "PV" in c and not "PV_number_of_0_m2" in c]
        no_pv_number = df["PV_number_of_0_m2"].sum()
        pv_df = pd.DataFrame(df[pv_columns].sum()).reset_index().rename(columns={"index": "size m2", 0:"number"})
        pv_df["size m2"] = pv_df["size m2"].map(lambda x: x.split("_")[-2])
        pv_df["country"] = file.name.split("_")[-2]
        pv_df["year"] = file.name.split("_")[-1].replace(".csv", "")
        pv_df["source"] = "Invert"
        dfs.append(pv_df)
    pv_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    pv_df["size m2"] = pv_df["size m2"].astype(int)
    pv_df["number"] = pv_df["number"].astype(float)
    pv_df["country"] = pv_df["country"].astype(str)
    pv_df["year"] = pv_df["year"].astype(int)
    
    
    size_df = pv_df.groupby(["size m2", "year"])["number"].sum().reset_index()
    size_df["number"] = size_df["number"] / 1_000_000
    size_df["peak power (kWp)"] = size_df["size m2"] / 8

    agg = size_df.groupby("year").agg({
    "number": "median",
    "peak power (kWp)": "median"
    }).reset_index()
    plt.rcParams.update({
        'axes.labelsize': 16,    # x and y axis labels
        'axes.titlesize': 16,    # subplot titles
        'xtick.labelsize': 16,   # x-axis tick labels
        'ytick.labelsize': 16,   # y-axis tick labels
        'legend.fontsize': 16,   # legend text size
    })    
    sns.scatterplot(
        data=size_df,
        y="peak power (kWp)",
        x="number",
        hue="year",
        palette=sns.color_palette(),
        alpha=0.5,
    )

    plt.ylim(0,50)
    plt.ylabel("size of PV installations (kWp)")
    plt.xlabel("number of PV installations on \n residential buildings with HP (mio.)")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"PV_size_distribution.svg")
    plt.show()
    plt.close()

    number_df = pv_df.groupby(["country", "year"])["number"].sum().reset_index()
    number_df["number"] = number_df["number"] / 1_000_000
    order = number_df.groupby("country")["number"].mean().sort_values().index
    plt.rcParams.update({
        'axes.labelsize': 14,    # x and y axis labels
        'axes.titlesize': 14,    # subplot titles
        'xtick.labelsize': 14,   # x-axis tick labels
        'ytick.labelsize': 14,   # y-axis tick labels
        'legend.fontsize': 14,   # legend text size
    })
    sns.barplot(
        data=number_df,
        x="country",
        y="number",
        hue="year",
        palette=sns.color_palette(),
        order=order
    )
    plt.xticks(rotation=90)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.ylabel("number of PV installations on \n residential buildings with HP (mio.)")
    plt.legend(title="", loc="upper left")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"PV_number_of_installations.svg")
    plt.show()
    plt.close()

def plot_price_quariles(loads: pd.DataFrame):
    price = loads.loc[(loads["ID_EnergyPrice"]=="Price 1") & (loads["year"]==2020) & (loads["country"]=="AUT"), "price (cent/kWh)"].reset_index()
    q1 = np.percentile(price["price (cent/kWh)"], 25)  # 1st quartile (25%)
    q4 = np.percentile(price["price (cent/kWh)"], 75)
    sns.lineplot(
        price,
        y="price (cent/kWh)",
        x="index",
        linewidth=0.5,
        label="price (cent/kWh)"
    )
    plt.axhline(q1, color="green", linestyle="--", label="1st Quartile", alpha=0.5)
    plt.axhline(q4, color="red", linestyle="--", label="4th Quartile", alpha=0.5)
    ax = plt.gca()
    min, max = ax.get_ylim()
    plt.fill_between(range(len(price)), min, q1, color="green", alpha=0.1)
    plt.fill_between(range(len(price)), q4, max, color="red", alpha=0.1)

    plt.xlabel("hour")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"price_quartiles_visualized_aut_2019.svg")
    plt.show()


def show_installed_HP_capacity_over_mean_national_demand(national: pd.DataFrame):
    df = pd.read_csv(SAVING_PATH.parent / "Installed_HP_capacity_per_country.csv", sep=";")
    df = df.rename(columns={"Unnamed: 0": "year"}).melt(id_vars=["year"], var_name="country", value_name="installed capacity (MW)")
    df["installed (GW)"] = df["installed capacity (MW)"] / 1_000
    national_mean = national.groupby(["country", "year"])["demand MW"].mean().reset_index()
    merged = pd.merge(left=df, right=national_mean, on=["country", "year"])
    merged["capacity per mean demand"] = merged["installed capacity (MW)"] / merged["demand MW"] * 100
    plot_values_on_EU_map(plot_df=merged, columns2plot="capacity per mean demand", y_label="installed HP capacity per mean grid demand (%)")
    
    order = df.groupby("country")["installed (GW)"].mean().sort_values().index
    sns.barplot(
        df,
        x="country",
        y="installed (GW)",
        hue="year",
        palette=sns.color_palette(),
        order=order
    )
    plt.xlabel("country")
    plt.legend(title="", loc="upper left")
    plt.xticks(rotation=90)
    plt.ylabel("installed heat pump capacity (GW)")
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Installed_HP_capacity.svg")
    plt.show()
    plt.close()

def show_shifted_demand_over_year_in_days_stacked(loads):
    loads["day"] = (loads["Hour"]-1) // 24 + 1
    loads["diff"] =  loads["opt_grid_demand_stock_MW"] - loads["ref_grid_demand_stock_MW"]
    loads["demand increase MW"] = loads["diff"].clip(lower=0)
    loads["demand decrease MW"] = loads["diff"].clip(upper=0)

    # import the average yearly temperature per country:
    temp_df = Cp.create_national_temperature_df()
    mean_temp = temp_df.groupby(["country","year"])["temperature"].mean().reset_index()
    mean_temp["year"] = mean_temp["year"].astype(int)
    color_dict = plot_values_on_EU_map(mean_temp, columns2plot="temperature", y_label="average yearly temperatue °C")

    df = loads.groupby(["country", "year", "ID_EnergyPrice", "day"])[["demand increase MW", "demand decrease MW"]].sum().reset_index()
    df_small = df.loc[df["ID_EnergyPrice"]=="Price 4", :]
    df_small[["demand increase GWh", "demand decrease GWh"]] = df_small[["demand increase MW", "demand decrease MW"]] / 1_000
    pivot_df_increase = df_small.pivot_table(
        index=['year', "day"],
        columns='country',
        values='demand increase GWh',
        aggfunc='sum'
    ).reset_index()
    pivot_df_decrease = df_small.pivot_table(
        index=['year', "day"],
        columns='country',
        values='demand decrease GWh',
        aggfunc='sum'
    ).reset_index()

    fig, axes = plt.subplots(figsize=(12, 5), ncols=2, nrows=1, sharey=True)
    for ax, year in zip(axes, [2030, 2050]):
        year_df_increase = pivot_df_increase.loc[pivot_df_increase["year"]==year, :]
        x_axis=year_df_increase["day"].to_numpy()
        country_columns_sorted = [c for c in color_dict[year].keys()]
        y_data = year_df_increase[country_columns_sorted].to_numpy().T
        country_colors = [color_dict[year].get(country, "#cccccc") for country in country_columns_sorted]

        ax.stackplot(
            x_axis,
            y_data,
            labels=country_columns_sorted,
            colors=country_colors,
            # edgecolor='black',
            linewidth=0.5
        )

        year_df_decrease = pivot_df_decrease.loc[pivot_df_decrease["year"]==year, :]
        x_axis=year_df_decrease["day"].to_numpy()
        country_columns_sorted = [c for c in color_dict[year].keys()]
        y_data = year_df_decrease[country_columns_sorted].to_numpy().T

        ax.stackplot(
            x_axis,
            y_data,
            labels=country_columns_sorted,
            colors=country_colors,
            # edgecolor='black',
            linewidth=0.5
        )

        ax.set_title(f"{year}")
        ax.set_xlabel("days")
        ax.set_ylabel("daily shifted demand (GWh)")
        ax.set_ylim(-200, 200)
    plt.legend().remove()
    plt.savefig(SAVING_PATH / f"Daily_Demand_shift_{COOLING_PERCENTAGE}_cooling.svg")
    plt.show()
    plt.close()


def plot_shifted_electricity_per_appliance_EU(price_id: int):
    def plot_stackplot(df, columns_positive, columns_negative, color_dict_positive, color_dict_negative, legend_elements, hp: str):
        fig, axes = plt.subplots(figsize=(12, 5), ncols=2, nrows=1, sharey=True)
        for ax, year in zip(axes, [2030, 2050]):
            year_df_increase = df.loc[df["year"]==year, columns_positive+["day"]]
            x_axis=year_df_increase["day"].to_numpy()
            y_data = year_df_increase[columns_positive].to_numpy().T
            type_colors = [color_dict_positive.get(name, "#cccccc") for name in columns_positive]

            ax.stackplot(
                x_axis,
                y_data,
                labels=columns_positive,
                colors=type_colors,
                # edgecolor='black',
                linewidth=0.5
            )

            year_df_decrease = df.loc[df["year"]==year, columns_negative]
            y_data = year_df_decrease[columns_negative].to_numpy().T
            type_colors_negative = [color_dict_negative.get(name, "#cccccc") for name in columns_negative]

            ax.stackplot(
                x_axis,
                y_data,
                labels=columns_negative,
                colors=type_colors_negative,
                # edgecolor='black',
                linewidth=0.5
            )


            ax.set_title(f"{year}")
            ax.set_xlabel("days")
            ax.set_ylabel("daily shifted demand (GWh$_{el}$)")
            ax.set_ylim(-200,200)
        
            ax.legend(handles=legend_elements, loc='lower center')
        
        plt.tight_layout()
        plt.savefig(SAVING_PATH / f"Daily_electric_demand_shift_Appliances_price{price_id}_{COOLING_PERCENTAGE}_cooling_{hp}.svg", bbox_inches='tight')
        plt.show()
        plt.close()

    df = Cp.calculate_shifted_electricity_per_appliance_type(COOLING_PERCENTAGE, price_id=price_id)
    df.reset_index(inplace=True)
    
    df["day"] = (df["Hour"]-1) // 24 + 1

    columns_positive = [
        "thermal_mass_shift_increase",  
        "DHW_shift_increase",  
        "Buffer_shift_increase",
        "thermal_mass_shift_cooling_increase",
    ]
    columns_negative = [
        "thermal_mass_shift_decrease", 
        "DHW_shift_decrease", 
        "Buffer_shift_decrease",
        "thermal_mass_shift_cooling_decrease",
    ]
    grouped = df.groupby(["ID_EnergyPrice", "year", "day"])[columns_positive+columns_negative].sum()
    grouped[[c for c in grouped.columns]] =  grouped[[c for c in grouped.columns]] / 1_000 
    
    df_new = grouped.reset_index()
    df_new["year"] = df_new["year"].astype(int)

    color_dict_positive = {col: sns.color_palette("Paired")[i*2] for i, col in enumerate(columns_positive)}
    color_dict_negative = {col: sns.color_palette("Paired")[i*2-1] for i, col in enumerate(columns_negative, start=1)}
    # Create custom legend with 3 patches
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative["thermal_mass_shift_decrease"], label="Thermal Mass"),
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative["DHW_shift_decrease"], label="DHW Tank"),
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative["Buffer_shift_decrease"], label="Heating Tank"),
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative["thermal_mass_shift_cooling_decrease"], label="Cooling")
    ]
    plot_stackplot(df_new, columns_positive, columns_negative, color_dict_positive, color_dict_negative, legend_elements, hp="")
    total_sum = df_new.groupby("year")[columns_positive].sum()
    print(f"\nPercentages of total shifted electricity demand in Price scenario {price_id}:")
    for year in [2030, 2050]:
        year_data = total_sum.loc[year]
        year_percentages = (year_data / year_data.sum() * 100).round(2)
        print(f"\nYear {year}:")
        for col, percentage in year_percentages.items():
            print(f"{col}: {percentage}%  -  {year_data[col]}GWh")

    # make the same plot only for HP heating and HP DHW and Cooling
    columns_positive_HP = [
        "space_cooling_shift_increase",
        "heating_HP_shift_increase",
        "DHW_HP_shift_increase",
    ]
    columns_negative_HP = [
        "space_cooling_shift_decrease",
        "heating_HP_shift_decrease",
        "DHW_HP_shift_decrease",
    ]
    grouped_HP = df.groupby(["ID_EnergyPrice", "year", "day"])[columns_positive_HP+columns_negative_HP].sum()
    grouped_HP[[c for c in grouped_HP.columns]] =  grouped_HP[[c for c in grouped_HP.columns]] / 1_000 
    df_new_HP = grouped_HP.reset_index()
    df_new_HP["year"] = df_new_HP["year"].astype(int)

    color_dict_positive_HP = {col: sns.color_palette("Paired")[i*2] for i, col in enumerate(columns_positive_HP)}
    color_dict_negative_HP = {col: sns.color_palette("Paired")[i*2-1] for i, col in enumerate(columns_negative_HP, start=1)}
    # Create custom legend with 3 patches
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative_HP["heating_HP_shift_decrease"], label="Heating"),
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative_HP["DHW_HP_shift_decrease"], label="DHW"),
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative_HP["space_cooling_shift_decrease"], label="Cooling"),
    ]
    plot_stackplot(df_new_HP, columns_positive_HP, columns_negative_HP, color_dict_positive_HP, color_dict_negative_HP, legend_elements, hp="HP")

    # Calculate percentages for each year
    total_sum_HP = df_new_HP.groupby("year")[columns_positive_HP].sum()
    print(f"\nPercentages of total shifted electricity demand in Price scenario {price_id} for HP:")
    for year in [2030, 2050]:
        year_data = total_sum_HP.loc[year]
        year_percentages = (year_data / year_data.sum() * 100).round(2)
        print(f"\nYear {year}:")
        for col, percentage in year_percentages.items():
            print(f"{col}: {percentage}%  -  {year_data[col]}GWh")

def plot_shifted_demand_per_appliance_EU(price_id: int):
    df = Cp.calculate_shifted_energy_per_appliance_type(COOLING_PERCENTAGE, price_id=price_id)
    df.reset_index(inplace=True)
    
    df["day"] = (df["Hour"]-1) // 24 + 1

    columns_positive = [
        "thermal_mass_shift_increase",  
        "DHW_shift_increase",  
        "Buffer_shift_increase",
        "space_cooling_shift_increase",
    ]
    columns_negative = [
        "thermal_mass_shift_decrease", 
        "DHW_shift_decrease", 
        "Buffer_shift_decrease",
        "space_cooling_shift_decrease",
    ]
    grouped = df.groupby(["ID_EnergyPrice", "year", "day"])[columns_positive+columns_negative].sum()
    grouped[[c for c in grouped.columns]] =  grouped[[c for c in grouped.columns]] / 1_000 / 1_000
    
    df_new = grouped.reset_index()
    df_new["year"] = df_new["year"].astype(int)

    color_dict_positive = {col: sns.color_palette("Paired")[i*2] for i, col in enumerate(columns_positive)}
    color_dict_negative = {col: sns.color_palette("Paired")[i*2-1] for i, col in enumerate(columns_negative, start=1)}


    fig, axes = plt.subplots(figsize=(12, 5), ncols=2, nrows=1, sharey=True)
    for ax, year in zip(axes, [2030, 2050]):
        year_df_increase = df_new.loc[df_new["year"]==year, columns_positive+["day"]]
        x_axis=year_df_increase["day"].to_numpy()
        y_data = year_df_increase[columns_positive].to_numpy().T
        type_colors = [color_dict_positive.get(name, "#cccccc") for name in columns_positive]

        ax.stackplot(
            x_axis,
            y_data,
            labels=columns_positive,
            colors=type_colors,
            # edgecolor='black',
            linewidth=0.5
        )

        year_df_decrease = df_new.loc[df_new["year"]==year, columns_negative]
        y_data = year_df_decrease[columns_negative].to_numpy().T
        type_colors_negative = [color_dict_negative.get(name, "#cccccc") for name in columns_negative]

        ax.stackplot(
            x_axis,
            y_data,
            labels=columns_negative,
            colors=type_colors_negative,
            # edgecolor='black',
            linewidth=0.5
        )


        ax.set_title(f"{year}")
        ax.set_xlabel("days")
        ax.set_ylabel("daily shifted demand (TWh$_{th}$)")
        ax.set_ylim(-0.8,0.8)
    
        # Create custom legend with 3 patches
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative["thermal_mass_shift_decrease"], label="Thermal Mass"),
            plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative["DHW_shift_decrease"], label="DHW Tank"),
            plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative["Buffer_shift_decrease"], label="Heating Tank"),
            plt.Rectangle((0,0), 1, 1, facecolor=color_dict_negative["space_cooling_shift_decrease"], label="Cooling")
        ]
        ax.legend(handles=legend_elements, loc='lower center')
    
    plt.tight_layout()
    plt.savefig(SAVING_PATH / f"Daily_Demand_shift_Appliances_price{price_id}_{COOLING_PERCENTAGE}_cooling.svg", bbox_inches='tight')
    plt.show()
    plt.close()

    # Calculate percentages for each year
    total_sum = df_new.groupby("year")[columns_positive].sum()
    print(f"\nPercentages of total shifted demand in Price scenario {price_id}:")
    for year in [2030, 2050]:
        year_data = total_sum.loc[year]
        year_percentages = (year_data / year_data.sum() * 100).round(2)
        print(f"\nYear {year}:")
        for col, percentage in year_percentages.items():
            print(f"{col}: {percentage}%  -  {year_data[col]}TWh")


def plot_daily_load_comparison_between_peak_and_fees(percentage_cooling):
    df_fees = load_project_files(percentage_cooling, project_acronym="grid_fees")
    df_peak = load_project_files(percentage_cooling, project_acronym="peak_price")
    daily_load_fees = df_fees.copy()
    daily_load_fees["hourofday"] = (daily_load_fees["Hour"]-1) % 24 + 1
    daily_load_fees = daily_load_fees.groupby(["year", "country", "ID_EnergyPrice", "hourofday"])["opt_grid_demand_stock_MW"].mean().reset_index()

    daily_load_peak = df_peak.copy()
    daily_load_peak["hourofday"] = (daily_load_peak["Hour"]-1) % 24 + 1
    daily_load_peak = daily_load_peak.groupby(["year", "country", "ID_EnergyPrice", "hourofday"])["opt_grid_demand_stock_MW"].mean().reset_index()
    # Create plotly subplot comparing daily load profiles
    years = sorted(daily_load_fees["year"].unique())
    countries = sorted(daily_load_fees["country"].unique())
    prices = sorted(daily_load_fees["ID_EnergyPrice"].unique())
    
    # Create subplot figure with columns for years and rows for prices
    fig = make_subplots(
        rows=len(prices),
        cols=len(years),
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=[f"Year {year}" for year in years] * len(prices),
        row_titles=[f"{price}" for price in prices],
    )
    
    # Create color mapping for countries
    country_colors = {country: color for country, color in zip(countries, cycle(qualitative.Plotly))}
    
    # Add traces for each year, price, country, and scenario
    for i, price in enumerate(prices):
        for j, year in enumerate(years):
            row_num = i + 1
            col_num = j + 1
            
            # Filter data for this year and price scenario
            year_data_fees = daily_load_fees[(daily_load_fees["year"] == year) & 
                                           (daily_load_fees["ID_EnergyPrice"] == price)]
            year_data_peak = daily_load_peak[(daily_load_peak["year"] == year) & 
                                           (daily_load_peak["ID_EnergyPrice"] == price)]
            
            for country in countries:
                country_data_fees = year_data_fees[year_data_fees["country"] == country]
                country_data_peak = year_data_peak[year_data_peak["country"] == country]
                
                if not country_data_fees.empty:
                    # Add solid line for grid fees scenario
                    fig.add_trace(
                        go.Scatter(
                            x=country_data_fees["hourofday"],
                            y=country_data_fees["opt_grid_demand_stock_MW"],
                            mode="lines",
                            name=f"{country} - Grid Fees",
                            line=dict(color=country_colors[country], dash="solid"),
                            legendgroup=country,
                            showlegend=(i == 0 and j == 0),  # Only show legend for first subplot
                        ),
                        row=row_num,
                        col=col_num,
                    )
                
                if not country_data_peak.empty:
                    # Add dashed line for peak pricing scenario
                    fig.add_trace(
                        go.Scatter(
                            x=country_data_peak["hourofday"],
                            y=country_data_peak["opt_grid_demand_stock_MW"],
                            mode="lines",
                            name=f"{country} - Peak Price",
                            line=dict(color=country_colors[country], dash="dash"),
                            legendgroup=country,
                            showlegend=(i == 0 and j == 0),  # Only show legend for first subplot
                        ),
                        row=row_num,
                        col=col_num,
                    )
    
    # Update layout
    fig.update_layout(
        height=300 * len(prices),  # Adjust height based on number of price scenarios
        width=1000 * len(years),    # Adjust width based on number of years
        title="Daily Load Profiles: Grid Fees vs Peak Pricing",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    # Update x and y axis labels
    # Add x-axis labels only to bottom row
    for j in range(1, len(years) + 1):
        fig.update_xaxes(title_text="Hour of Day", row=len(prices), col=j)
    
    # Add y-axis labels only to first column
    for i in range(1, len(prices) + 1):
        fig.update_yaxes(title_text="Grid Demand (MW)", row=i, col=1)
    
    # Save the plot as HTML
    saving_path = SAVING_PATH.parent / f"daily_load_comparison_{PROJECT_ACRONYM}_cooling{COOLING_PERCENTAGE}.html"
    fig.write_html(saving_path)
    LOGGER.info(f"Saved plotly figure: daily_load_comparison_{PROJECT_ACRONYM}_cooling{COOLING_PERCENTAGE}.html")


def plot_peak_day_comparison_between_peak_and_fees(percentage_cooling):
    df_fees = load_project_files(percentage_cooling, project_acronym="grid_fees")
    df_peak = load_project_files(percentage_cooling, project_acronym="peak_price")
    
    def extract_peak_day_profiles(df):
        """Extract the 24-hour profile for the day with peak opt_grid_demand_stock_MW"""
        df["day"] = (df["Hour"] - 1) // 24 + 1
        df["hourofday"] = (df["Hour"] - 1) % 24 + 1
        
        peak_day_profiles = []
        
        for (country, year, price), group in df.groupby(["country", "year", "ID_EnergyPrice"]):
            # Find the hour with peak demand
            peak_hour = group.loc[group["opt_grid_demand_stock_MW"].idxmax(), "Hour"]
            peak_day = (peak_hour - 1) // 24 + 1
            
            # Extract the full 24-hour profile for that day
            day_profile = group[group["day"] == peak_day].copy()
            day_profile = day_profile.sort_values("hourofday")
            
            if len(day_profile) == 24:  # Ensure we have complete 24-hour data
                peak_day_profiles.append(day_profile)
        
        return pd.concat(peak_day_profiles, ignore_index=True) if peak_day_profiles else pd.DataFrame()
    
    # Extract peak day profiles for both scenarios
    peak_day_fees = extract_peak_day_profiles(df_fees)
    peak_day_peak = extract_peak_day_profiles(df_peak)
    
    if peak_day_fees.empty or peak_day_peak.empty:
        LOGGER.warning("No complete peak day profiles found")
        return
    
    # Create the plotly visualization
    years = sorted(peak_day_fees["year"].unique())
    countries = sorted(peak_day_fees["country"].unique())
    prices = sorted(peak_day_fees["ID_EnergyPrice"].unique())
    
    # Create subplot figure with columns for years and rows for prices
    fig = make_subplots(
        rows=len(prices),
        cols=len(years),
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=[f"Year {year}" for year in years] * len(prices),
        row_titles=[f"{price}" for price in prices],
    )
    
    # Create color mapping for countries
    country_colors = {country: color for country, color in zip(countries, cycle(qualitative.Plotly))}
    
    # Add traces for each year, price, country, and scenario
    for i, price in enumerate(prices):
        for j, year in enumerate(years):
            row_num = i + 1
            col_num = j + 1
            
            # Filter data for this year and price scenario
            year_data_fees = peak_day_fees[(peak_day_fees["year"] == year) & 
                                         (peak_day_fees["ID_EnergyPrice"] == price)]
            year_data_peak = peak_day_peak[(peak_day_peak["year"] == year) & 
                                         (peak_day_peak["ID_EnergyPrice"] == price)]
            
            for country in countries:
                country_data_fees = year_data_fees[year_data_fees["country"] == country]
                country_data_peak = year_data_peak[year_data_peak["country"] == country]
                
                if not country_data_fees.empty:
                    # Add solid line for grid fees scenario
                    fig.add_trace(
                        go.Scatter(
                            x=country_data_fees["hourofday"],
                            y=country_data_fees["opt_grid_demand_stock_MW"],
                            mode="lines",
                            name=f"{country} - Grid Fees",
                            line=dict(color=country_colors[country], dash="solid"),
                            legendgroup=country,
                            showlegend=(i == 0 and j == 0),  # Only show legend for first subplot
                        ),
                        row=row_num,
                        col=col_num,
                    )
                
                if not country_data_peak.empty:
                    # Add dashed line for peak pricing scenario
                    fig.add_trace(
                        go.Scatter(
                            x=country_data_peak["hourofday"],
                            y=country_data_peak["opt_grid_demand_stock_MW"],
                            mode="lines",
                            name=f"{country} - Peak Price",
                            line=dict(color=country_colors[country], dash="dash"),
                            legendgroup=country,
                            showlegend=(i == 0 and j == 0),  # Only show legend for first subplot
                        ),
                        row=row_num,
                        col=col_num,
                    )
    
    # Update layout
    fig.update_layout(
        height=300 * len(prices),  # Adjust height based on number of price scenarios
        width=1000 * len(years),    # Adjust width based on number of years
        title="Peak Day Load Profiles: Grid Fees vs Peak Pricing",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    # Update x and y axis labels
    # Add x-axis labels only to bottom row
    for j in range(1, len(years) + 1):
        fig.update_xaxes(title_text="Hour of Day", row=len(prices), col=j)
    
    # Add y-axis labels only to first column
    for i in range(1, len(prices) + 1):
        fig.update_yaxes(title_text="Grid Demand (MW)", row=i, col=1)
    
    # Save the plot as HTML
    saving_path = SAVING_PATH.parent / f"peak_day_load_comparison_{PROJECT_ACRONYM}_cooling{COOLING_PERCENTAGE}.html"
    fig.write_html(saving_path)
    LOGGER.info(f"Saved plotly figure: peak_day_load_comparison_{PROJECT_ACRONYM}_cooling{COOLING_PERCENTAGE}.html")


def compare_peak_pricing_with_fixed_grid_fees(percentage_cooling, national):
    df_fees = load_project_files(percentage_cooling, project_acronym="grid_fees")
    df_peak = load_project_files(percentage_cooling, project_acronym="peak_price")

    national_fees = pd.merge(left=df_fees, right=national[["country", "demand MW", "year", "Hour"]], on=["year", "Hour", "country"])
    national_fees["demand_opt"] = national_fees["demand MW"] - national_fees["ref_grid_demand_stock_MW"] + national_fees["opt_grid_demand_stock_MW"]
    demand_peaks_fees = national_fees.groupby(["year", "country", "ID_EnergyPrice"])[["demand MW", "demand_opt"]].max().reset_index()
    demand_peaks_fees["national_increase"] = (demand_peaks_fees["demand_opt"] - demand_peaks_fees["demand MW"]) / demand_peaks_fees["demand MW"] * 100 

    national_peak = pd.merge(left=df_peak, right=national[["country", "demand MW", "year", "Hour"]], on=["year", "Hour", "country"])
    national_peak["demand_opt"] = national_peak["demand MW"] - national_peak["ref_grid_demand_stock_MW"] + national_peak["opt_grid_demand_stock_MW"]
    demand_peaks_peak = national_peak.groupby(["year", "country", "ID_EnergyPrice"])[["demand MW", "demand_opt"]].max().reset_index()
    demand_peaks_peak["national_increase_peak"] = (demand_peaks_peak["demand_opt"] - demand_peaks_peak["demand MW"]) / demand_peaks_peak["demand MW"] * 100 

    merged = pd.merge(left=demand_peaks_fees, right=demand_peaks_peak[["country", "year", "ID_EnergyPrice", "national_increase_peak"]], on=["year", "country", "ID_EnergyPrice"])
    
    # plot_df = merged.copy()
    # plot_df["ID_EnergyPrice"] = plot_df["ID_EnergyPrice"].map(add_price_information())
    plt.rcParams.update({
        'axes.labelsize': 14,    # x and y axis labels
        'axes.titlesize': 14,    # subplot titles
        'xtick.labelsize': 14,   # x-axis tick labels
        'ytick.labelsize': 14,   # y-axis tick labels
        'legend.fontsize': 14,   # legend text size
    })
    # Create offset datasets for side-by-side plotting
    plot_df_peak = demand_peaks_peak.copy()
    plot_df_peak["ID_EnergyPrice"] = plot_df_peak["ID_EnergyPrice"].map(add_price_information())
    plot_df_fees = demand_peaks_fees.copy()
    plot_df_fees["ID_EnergyPrice"] = plot_df_fees["ID_EnergyPrice"].map(add_price_information())

    # Map years to sequential integers and then apply offset
    prices = plot_df_peak["ID_EnergyPrice"].unique()
    offsets = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    price_offsets_fees = {price:offsets[::2][i] for i, price in enumerate(prices)}
    price_offsets_peak = {price:offsets[1::2][i] for i, price in enumerate(prices)}

    plot_df_peak["price_offset"] = plot_df_peak["ID_EnergyPrice"].map(price_offsets_peak) 
    plot_df_peak.loc[plot_df_peak["year"] == 2030, "price_offset"] += 0
    plot_df_peak.loc[plot_df_peak["year"] == 2050, "price_offset"] += 4

    plot_df_fees["price_offset"] = plot_df_fees["ID_EnergyPrice"].map(price_offsets_fees) 
    plot_df_fees.loc[plot_df_fees["year"] == 2030, "price_offset"] += 0
    plot_df_fees.loc[plot_df_fees["year"] == 2050, "price_offset"] += 4
    merged_offset = pd.concat([plot_df_fees[["year", "ID_EnergyPrice", "national_increase", "price_offset"]], plot_df_peak[["year", "ID_EnergyPrice", "national_increase", "price_offset"]]], axis=0)
    merged_offset["price_offset"] = pd.to_numeric(merged_offset["price_offset"], errors="coerce")
    import matplotlib.path as mpath
    fig, ax = plt.subplots(figsize=(10, 8))
    # Add hatching to peak pricing boxes 
    bp2 = sns.boxplot(
        data=merged_offset,
        x="national_increase",
        y="price_offset",
        hue="ID_EnergyPrice",
        orient="h",
        palette=sns.color_palette(),
        ax=ax,
        dodge=False,
        width=0.9,
    )
    
    for i, thisbar in enumerate(bp2.artists):  # Access Rectangle objects
    # Get the corresponding price_offset for this box
        y_offset = merged_offset.iloc[i // len(merged_offset["national_increase"].unique())]["price_offset"]

        # Modify the position of each rectangle (box)
        thisbar.set_y(y_offset - thisbar.get_height() / 2)  # Center the box at the y_offset

    for i,thisbar in enumerate(bp2.patches):
        if i % 2 == 1:
            thisbar.set_hatch("///")
   
    legend_elements = []
    # Add price scenario colors
    for i, price in enumerate(prices):
        legend_elements.append(mpatches.Patch(color=sns.color_palette()[i], label=price))
    
    # Add scenario patterns
    legend_elements.extend([
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Grid Fees"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="Peak Pricing"),
    ])
    
    ax.legend(handles=legend_elements, title="", loc="lower right", ncol=1)
    ax.set_xlabel("National peak demand increase (%)")
    ax.set_ylabel("year")
    ax.set_yticks([0, 4])
    ax.set_yticklabels(["2030", "2050"])
    
    plt.tight_layout()
    plt.savefig(SAVING_PATH.parent / f"Peak_demand_comparison_national_EU_cooling_seperated{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

    # plot the peak reduction of the residential buildings
    peaks_fees = df_fees.groupby(["year", "country", "ID_EnergyPrice"])[["opt_grid_demand_stock_MW", "ref_grid_demand_stock_MW"]].max().reset_index()
    peaks_fees["peak_increase_fees"] = (peaks_fees["opt_grid_demand_stock_MW"] - peaks_fees["ref_grid_demand_stock_MW"]) / peaks_fees["ref_grid_demand_stock_MW"] * 100
    peaks_peak = df_peak.groupby(["year", "country", "ID_EnergyPrice"])[["opt_grid_demand_stock_MW", "ref_grid_demand_stock_MW"]].max().reset_index()
    peaks_peak["peak_increase_peak"] = (peaks_peak["opt_grid_demand_stock_MW"] - peaks_peak["ref_grid_demand_stock_MW"]) / peaks_peak["ref_grid_demand_stock_MW"] * 100
    merged_peaks = pd.merge(left=peaks_fees, right=peaks_peak[["year", "country", "ID_EnergyPrice", "peak_increase_peak"]], on=["year", "country", "ID_EnergyPrice"])
    

    merged_peaks["ID_EnergyPrice"] = merged_peaks["ID_EnergyPrice"].map(add_price_information())
     
    plt.rcParams.update({
        'axes.labelsize': 14,    # x and y axis labels
        'axes.titlesize': 14,    # subplot titles
        'xtick.labelsize': 14,   # x-axis tick labels
        'ytick.labelsize': 14,   # y-axis tick labels
        'legend.fontsize': 14,   # legend text size
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # Add hatching to peak pricing boxes 
    
    bp2 = sns.boxplot(
        data=merged_peaks,
        x="peak_increase_peak",
        y="year",
        hue="ID_EnergyPrice",
        orient="y",
        palette=sns.color_palette(),
        ax=ax,
        dodge=True,
        
    )
    for i,thisbar in enumerate(bp2.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch("///")

    bp = sns.boxplot(
        data=merged_peaks,
        x="peak_increase_fees",
        y="year",
        hue="ID_EnergyPrice",
        orient="y",
        palette=sns.color_palette(),
        ax=ax,
        dodge=True
    )
    
    legend_elements = []
    # Add price scenario colors
    for i, price in enumerate(merged_peaks["ID_EnergyPrice"].unique()):
        legend_elements.append(mpatches.Patch(color=sns.color_palette()[i], label=price))
    
    # Add scenario patterns
    legend_elements.extend([
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Grid Fees"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="Peak Pricing"),
    ])
    
    ax.legend(handles=legend_elements, title="", loc="lower center", ncol=1)
    ax.set_xlabel("Prosumager peak demand increase (%)")
    plt.tight_layout()
    plt.savefig(SAVING_PATH.parent / f"Peak_demand_comparison_residential_EU_cooling_seperated{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

    
    # shifted electricity
    def calc_shifted(loads):
        loads["shifted MW"] = loads["ref_grid_demand_stock_MW"] - loads["opt_grid_demand_stock_MW"]
        loads.loc[loads["shifted MW"] < 0, "shifted MW"] = 0

        shifted_df = loads.groupby(["year", "country", "ID_EnergyPrice"])[["shifted MW", "ref_grid_demand_stock_MW"]].sum().reset_index()
        shifted_df["shifted (%)"] = shifted_df["shifted MW"] / shifted_df["ref_grid_demand_stock_MW"] * 100
        
        x_order = shifted_df.groupby("country")["shifted (%)"].max().sort_values().index
        shifted_df["ID_EnergyPrice"] = shifted_df["ID_EnergyPrice"].map(add_price_information())
        return shifted_df

    shifted_fees = calc_shifted(df_fees)
    shifted_peak = calc_shifted(df_peak)
    shifted_peak.rename(columns={"shifted MW" : "shifted MW peak"}, inplace=True) 
    merged_shifted = pd.merge(left=shifted_fees, right=shifted_peak[["year", "country", "ID_EnergyPrice", "shifted MW peak"]], on=["year", "country", "ID_EnergyPrice"])
    merged_shifted["decrease in shifted energy GWh"] = (merged_shifted["shifted MW peak"] - merged_shifted["shifted MW"]) / 1_000 # GWh
    
    plt.figure(figsize=(10, 8))
    sns.boxplot(
        data=merged_shifted,
        x="decrease in shifted energy GWh",
        y="year",
        hue="ID_EnergyPrice",
        orient="y",
        palette=sns.color_palette(),
        
    )
    plt.legend()
    plt.xlabel("decrease in amount of shifted electricity by adding peak pricing in EU (GWh)")
    plt.tight_layout()
    plt.savefig(SAVING_PATH.parent / f"Shifted_electricity_comparison_EU_cooling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()

    plot_df = merged_shifted.copy()
    plot_df.drop(columns=["ref_grid_demand_stock_MW", "shifted (%)", "decrease in shifted energy GWh"], inplace=True)
    # plot_df = plot_df.melt(id_vars=["year", "country", "ID_EnergyPrice"], value_vars=["shifted MW", "shifted MW peak"], var_name="scenario", value_name="shifted")
    # plot_df["scenario"] = plot_df["scenario"].map({"shifted MW": "grid fees", "shifted MW peak": "grid fees + peak price"})
    # plot_df["shifted"] = plot_df["shifted"] / 1_000 / 1_000

    plot_df = plot_df.groupby(["year", "ID_EnergyPrice"])[["shifted MW", "shifted MW peak"]].sum().reset_index()
    plot_df[["shifted MW", "shifted MW peak"]] = plot_df[["shifted MW", "shifted MW peak"]] / 1_000_000

    
    x = np.arange(len(plot_df))  # one per energy price level
    width = 0.35
    prices = [f"{i} cent/kWh grid fees" for i in [0, 5, 10, 20, 40]]
    years = sorted(plot_df["year"].unique())
    price_colors = {price: sns.color_palette()[i] for i, price in enumerate(prices)}
    position_price = {price: -0.8 + i*0.4 for i, price in enumerate(prices)}
    x_labels = ["2030", "2050"]
    unique_groups = plot_df[["year", "ID_EnergyPrice"]].drop_duplicates()
    


    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in plot_df.iterrows():
        year=row["year"]
        id_price = row["ID_EnergyPrice"]
        shifted = row["shifted MW"]
        shifted_peak = row["shifted MW peak"]

        if year == 2030:
            position_year = 0
        else:
            position_year = 3

        position = position_price[id_price] + position_year

        ax.bar(position, height=shifted, color=price_colors[id_price], hatch="", width=0.2, edgecolor="black")
        ax.bar(position+0.2, height=shifted_peak, color=price_colors[id_price], hatch="//", width=0.2, edgecolor="black")

    ax.set_xticks([0, 3])
    ax.set_xticklabels(x_labels, rotation=0)
    legend_elements = [Patch(facecolor=price_colors[p], label=p) for p in prices] + [
        Patch(facecolor="white", hatch="///", label="grid fees + peak price", edgecolor="black"),
        Patch(facecolor="white", hatch="", label="grid fees", edgecolor="black"),
        ]

    plt.legend(title="", loc="upper left", handles=legend_elements)
    plt.xlabel("year")
    plt.ylabel("shifted electricity in the EU in TWh")
    plt.tight_layout()
    plt.savefig(SAVING_PATH.parent / f"Shifted_EU_absolute_grid_demand_coling{COOLING_PERCENTAGE}.svg")
    plt.show()
    plt.close()



def load_project_files(percentage_cooling, project_acronym):
    path_2_demand_file = Path(__file__).parent / f"EU27_loads_{project_acronym}_cooling-{percentage_cooling}.parquet.gzip"
    if not path_2_demand_file.exists():
        Cp.main(percentage_cooling)
    df = pd.read_parquet(Path(__file__).parent / f"EU27_loads_{project_acronym}_cooling-{percentage_cooling}.parquet.gzip")
    df["year"] = df["year"].astype(int)
    df["country"] = df["country"].astype(str)
    df["ID_EnergyPrice"] = df["ID_EnergyPrice"].astype(int)
    df["ID_EnergyPrice"] = df["ID_EnergyPrice"].map({i: f"Price {i}" for i in range(1, len(df["ID_EnergyPrice"].unique())+1)})
    return df

def main(percentage_cooling: float):
    df = load_project_files(percentage_cooling, project_acronym=PROJECT_ACRONYM)
    national_demand = Cp.get_national_demand_profiles()
    # compare_peak_pricing_with_fixed_grid_fees(percentage_cooling, national_demand)
    
    plot_shifted_electricity_per_appliance_EU(price_id=4)
    # plot_shifted_demand_per_appliance_EU(price_id=4)
    # plot_daily_load_factor(loads=df, national=national_demand)
    # show_shifted_demand_over_year_in_days_stacked(df)
    # show_installed_HP_capacity_over_mean_national_demand(national_demand)
    # analyse_prices(loads=df, national=national_demand)
    # analyse_peak_demand(loads=df, national=national_demand)
    # show_national_demand_increase_in_high_and_low_price_quantile(loads=df, national=national_demand) 
    # show_residential_demand_increase_in_high_and_low_price_quantile(loads=df) 
    # plot_shifted_electricity(loads=df)
    # plot_PV_self_consumption(loads=df, percentage_cooling=percentage_cooling, project_acronym=PROJECT_ACRONYM)
    # plot_daily_load_comparison_between_peak_and_fees  # creates html file with they mean daily load profiles for both scenarios
    # plot_peak_day_comparison_between_peak_and_fees  # creates html file with they peak day load profiles for both scenarios

    # show_average_day_profile(loads=df)
    # show_flexibility_factor(loads=df)
    # show_GSCrel_and_GSC_abs(loads=df)
    # plot_grid_demand_increase(loads=df, national=national_demand)
    # compare_heat_demand_with_invert_2020()

    # TODO plot relative price change (p2p?) and plot p2p of 2030 2050 prices so they are comparable with other study

    # calculate_price_correlations(loads=df, national=national_demand)
    # plot_load_factor(loads=df, national=national_demand, scenario="shiny happy")  # not working
    # plot_country_specific_heating_demand(COOLING_PERCENTAGE)

    # plot_PV_installations_and_size_distribution()
    # plot_price_quariles(loads=df) #not needed?
    # show_day_with_peak_deamand(loads=df, national=national_demand) # not done yet

    # plot_flexible_storage_efficiency(loads=df) # makes no sense


if __name__ == "__main__":
    COOLING_PERCENTAGE = 0.1
    PROJECT_ACRONYM = "grid_fees"  # grid_fees peak_price
    SAVING_PATH = Path(__file__).parent / f"figures_{PROJECT_ACRONYM}"
    SAVING_PATH.mkdir(exist_ok=True)
    main(
        percentage_cooling=COOLING_PERCENTAGE,
    )

    # TODO 
    # weather data from balmorel
    # hwb im csv checken GRC 2040
    # paper mit hooks lesen und ins paper einbauen
    #  add weighted GSC to correclation matrix
    # using 220 and the generation of renewables from ENTSO-E try see if a variable grid fee would enhance the results even more
    # or create more price scenarios where 1/4 of buildings have different grid fees
    # include shifted energy into analysis (total amount of shifted electricity)
    # daily load factor
    