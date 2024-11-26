import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlalchemy
from typing import List
import matplotlib.gridspec as gridspec




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