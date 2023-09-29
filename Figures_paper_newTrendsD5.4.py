import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

EUROPEAN_COUNTRIES = {
    'AUT': 'Austria',  #
    'BEL': 'Belgium',  #
    'BGR': 'Bulgaria',  #
    'HRV': 'Croatia',  #
    "CYP": "Cyprus",
    'CZE': 'Czech Republic',
    'DNK': 'Denmark',  #
    'EST': 'Estonia',  #
    'FIN': 'Finland',  #
    'FRA': 'France',  #
    'DEU': 'Germany',  #
    'GRC': 'Greece',  #
    'HUN': 'Hungary',  #
    'IRL': 'Ireland',  #
    'ITA': 'Italy',  #
    'LVA': 'Latvia',  #
    'LTU': 'Lithuania',  #
    'LUX': 'Luxembourg',  #
    'MLT': 'Malta',  #
    'NLD': 'Netherlands',  #
    'POL': 'Poland',  #
    'PRT': 'Portugal',  #
    'ROU': 'Roumania',  #
    'SVK': 'Slovakia',  #
    'SVN': 'Slovenia',  #
    'ESP': 'Spain',  #
    'SWE': 'Sweden'  #
}

path_to_files = Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\NewTrends\Deliverables\D3.3\NewTrends_Decarb")

capacities = pd.read_csv(path_to_files / "Capacities_D5.4.csv", sep=";")
capacities["country"] = capacities["country"].replace(EUROPEAN_COUNTRIES)
load_factor = pd.read_csv(path_to_files / "Load Factor (%).csv", sep=";")
load_factor["price_id"] = load_factor["price_id"].replace({"optimized price 1": "prosumager",
                                                           "reference": "consumer"})

self_consumption = pd.read_csv(path_to_files / "PV self consumption (%).csv", sep=";")
self_consumption["price_id"] = self_consumption["price_id"].replace({"optimized price 1": "prosumager",
                                                           "reference": "consumer"})

mac = pd.read_csv(path_to_files / "MAC electricity price (centperkWh).csv", sep=";")
shifted_electricity = pd.read_csv(path_to_files / "shifted electricity (%).csv", sep=";")
shifted_electricity_mwh = pd.read_csv(path_to_files / "shifted electricity (MWh).csv", sep=";")
mean_electricity_price = pd.read_csv(path_to_files / "electricity price mean (centperkWh).csv", sep=";")


def merge_dfs(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if "price_id" in right.columns:
        dataframe = pd.merge(left=left, right=right, on=["country", "year", "price_id"], how="outer")
    else:
        dataframe = pd.merge(left=left, right=right, on=["country", "year"], how="outer")
    return dataframe


# X: total PV capacity, Y: PV self consumption
plot_df = merge_dfs(left=self_consumption, right=capacities)

plot_df["total storage capacity (GWh)"] = (plot_df["total_battery_capacity (kWh)"] +
                                           plot_df["total_dhw_capacity (kWh)"] +
                                           plot_df["total_buffer_capacity (kWh)"]) / 1_000_000

for df in [
    mac.drop(columns=["price_id"]),
    load_factor,
    shifted_electricity.drop(columns=["price_id"]),
    shifted_electricity_mwh.drop(columns=["price_id"]),
    mean_electricity_price.drop(columns=["price_id"])
]:
    plot_df = merge_dfs(left=plot_df, right=df)

plot_df["shifted electricity (TWh)"] = plot_df["shifted electricity (MWh)"] / 1_000_000
plot_df["PV capacity (GWp)"] = plot_df["total_pv_capacity (kWp)"] / 1_000_000
plot_df["Load Factor (%)"] = plot_df["Load Factor (%)"] * 100  # %

sns.scatterplot(data=plot_df.query(f"year != 2020"),
                x="PV capacity (GWp)",
                y="PV self consumption (%)",
                hue="year",
                style="consumer type",
                )
plt.show()

#
sns.scatterplot(data=plot_df.query(f"year != 2020"),
                x="total_battery_capacity (kWh)",
                y="PV self consumption (%)",
                hue="year",
                style="consumer type",
                )
plt.show()

sns.scatterplot(data=plot_df.query(f"year != 2020"),
                x="total storage capacity (GWh)",
                y="PV self consumption (%)",
                hue="year",
                style="consumer type",
                )
plt.show()

sns.scatterplot(data=plot_df.query(f"year != 2020"),
                x="MAC electricity price (cent/kWh)",
                y="Load Factor (%)",
                hue="year",
                style="consumer type"
                )
plt.show()

# MAC, Shifted electricity (%)
sns.scatterplot(data=plot_df.query(f"year != 2020"),
                x="MAC electricity price (cent/kWh)",
                y="shifted electricity (%)",
                hue="year",
                )
plt.show()

# MAC, Shifted electricity TWh
sns.scatterplot(data=plot_df.query(f"year != 2020"),
                x="MAC electricity price (cent/kWh)",
                y="shifted electricity (TWh)",
                hue="year",
                )
plt.show()

# mean electricity price, installed PV capacity:
sns.scatterplot(data=plot_df.query(f"year != 2020"),
                x="electricity price mean (cent/kWh)",
                y="PV capacity (GWp)",
                hue="year",
                )
plt.show()
