import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

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
pv_share = pd.read_csv(path_to_files / "PV share (%).csv", sep=";")
pv_share["price_id"] = pv_share["price_id"].replace({"1": "prosumager",
                                                     "reference_1": "consumer"})

primes_df = pd.read_excel(path_to_files / "PRIMES_Prosumager_data_0929.xlsx", engine="openpyxl")
primes_df[["Load Factor (%)", "PV share (%)", "shifted electricity (%)", "PV self consumption (%)"]] = primes_df[[
    "Load Factor (%)", "PV share (%)", "shifted electricity (%)", "PV self consumption (%)"]] * 100


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
    mean_electricity_price.drop(columns=["price_id"]),
    pv_share
]:
    plot_df = merge_dfs(left=plot_df, right=df)

plot_df["shifted electricity (TWh)"] = plot_df["shifted electricity (MWh)"] / 1_000_000
plot_df["PV capacity (GWp)"] = plot_df["total_pv_capacity (kWp)"] / 1_000_000
plot_df["Load Factor (%)"] = plot_df["Load Factor (%)"] * 100  # %
plot_df["consumer type"] = plot_df["price_id"]
plot_df["model"] = "FLEX"

primes_for_merge = pd.merge(left=primes_df,
                            right=plot_df[["country", "year", "consumer type", "MAC electricity price (cent/kWh)",
                                           "electricity price mean (cent/kWh)"]],
                            on=["country", "year", "consumer type"])
for_merge = plot_df[[name for name in primes_for_merge.columns]].query(f"year != 2020")

combined_df = pd.concat([for_merge, primes_for_merge], axis=0)
combined_df.to_csv(path_to_files / "data_for_D5.4_graphiks.csv", index=False, sep=";")

for_plot = combined_df.loc[combined_df.loc[:, "consumer type"] == "prosumager", :]

# prepare plot_df to merge with PRIMES:
# sns.set_palette("tab10")


def create_scatter_plot(data: pd.DataFrame, x: str, y: str, style="year", hue="model"):
    plot_name = f"{x.replace(r'/', 'per')}_over_{y}.png"
    path_to_figures = Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\NewTrends\Deliverables\D5.4 paper\Figures")
    data["model"] = data["model"].replace({"PRIMES": "PRIMES-Prosumager", "FLEX": "Invert/FLEX"})
    palette = sns.color_palette("hls", n_colors=2)
    if y == "shifted electricity (%)":
        sns.relplot(
            data=data,
            x=x,
            y=y,
            col="year",
            hue=style,
            # style="consumer type",  # style
            kind="scatter",
            palette=palette
        )
    else:
        sns.relplot(
            data=data,
            x=x,
            y=y,
            col="year",
            # size="PV capacity (GWp)",
            hue=style,
            # style="consumer type",  # style
            kind="scatter",
            palette=palette
        )
    matplotlib.rcParams.update({'font.size': 15})

    # sns.scatterplot(data=data.query(query_string),
    #                 x=x,
    #                 y=y,
    #                 hue=hue,
    #                 style=style,
    #                 palette=palette
    #                 )
    plt.savefig(path_to_figures / plot_name,
                # dpi=1200
                )
    plt.savefig(path_to_figures / plot_name.replace("png", "svg"))
    plt.show()


x_y_pairs = [("PV capacity (GWp)", "PV self consumption (%)"),
             ("PV capacity (GWp)", "Load Factor (%)"),
             ("PV capacity (GWp)", "shifted electricity (%)"),
             ("PV capacity (GWp)", "PV share (%)"),
             ("MAC electricity price (cent/kWh)", "PV self consumption (%)"),
             ("MAC electricity price (cent/kWh)", "Load Factor (%)"),
             ("MAC electricity price (cent/kWh)", "shifted electricity (%)"),
             ("MAC electricity price (cent/kWh)", "PV share (%)"),
             ("electricity price mean (cent/kWh)", "PV capacity (GWp)"),
             ("electricity price mean (cent/kWh)", "PV self consumption (%)"),
             ("shifted electricity (%)", "Load Factor (%)"),
             ("PV share (%)", "PV self consumption (%)")
             ]

for x, y in x_y_pairs:
    create_scatter_plot(data=for_plot,
                        x=x,
                        y=y,
                        style="model",
                        hue="year")
