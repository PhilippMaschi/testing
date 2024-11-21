import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlalchemy
from typing import List




def plot_supply_and_demand_matching_over_price(price: np.array, s_d_match: dict, country: str, year: str):

    plt.plot(s_d_match.keys(), np.array([s for s in s_d_match.values()]) / 1_000)
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.hist(price*1_000, 100, alpha=0.5, color="green")
    ax.set_xlabel("electricity wholesale price (cent/kWh)")
    ax.set_ylabel("cumulated additional electricity usage below \n a price threshold (GWh)")
    ax2.set_ylabel("frequency of prices occuring during the year")
    ax2.tick_params(axis='y', colors='green')
    ax2.spines['right'].set_color('green')
    plt.savefig(Path(__file__).parent / "single_country_plots" / f"supply_demand_matching_{country}_{year}.png")
    plt.close()


    plt.scatter(x["price"] * 1_000, x["change in electricity demand"])
    plt.savefig(Path(__file__).parent / "single_country_plots" / f"supply_demand_matching_{country}_{year}.png")
    plt.close()


