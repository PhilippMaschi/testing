import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib.ticker import PercentFormatter


YEARS = [2020, 2030, 2040, 2050]

# Murcia scnearios
# Strong policy
heating_murcia_high_eff = {
    "Air HP": [0.2, 0.35, 0.6, 0.8],
    "Ground HP": [0, 0.02, 0.04, 0.06],
    "Direct electric heating": [0.39, 0.3, 0.2, 0.05],
    "Conventional": [0.19, 0.15, 0.07, 0.02],
}
devices_murcia_high_eff = {
    "AC": [0.5, 0.6, 0.8, 0.9],
    "DHW tank": [0.5, 0.55, 0.6, 0.65],
    "Buffer tank": [0, 0.05, 0.15, 0.25],
    "PV": [0.015, 0.1, 0.4, 0.6],
    "Battery": [0.1, 0.12, 0.2, 0.3],
    "Prosumager": [0, 0.1, 0.3, 0.5],
    }

# with weak policy
heating_murcia_moderate_eff = {
    "Air HP": [0.2, 0.3, 0.5, 0.7],
    "Ground HP": [0, 0.01, 0.02, 0.03],
    "Direct electric heating": [0.39, 0.32, 0.2, 0.1],
    "Conventional": [0.19, 0.16, 0.1, 0.05],
}
devices_murcia_moderate_eff = {
    "AC": [0.5, 0.65, 0.8, 0.95],
    "DHW tank": [0.5, 0.55, 0.6, 0.65],
    "Buffer tank": [0, 0.05, 0.15, 0.25],
    "PV": [0.015, 0.15, 0.3, 0.5],
    "Battery": [0.1, 0.12, 0.16, 0.25],
    "Prosumager": [0, 0.1, 0.2, 0.4],
    }


# Leeuwarden
# Strong policy
heating_leeuwarden_high_eff = {
    "Air HP": [0.04, 0.18, 0.5, 0.7],
    "Ground HP": [0, 0.05, 0.1, 0.15],
    "Direct electric heating": [0.02, 0.03, 0.02, 0.01],
    "Conventional": [0.9, 0.7, 0.34, 0.1],
} 
devices_leeuwarden_high_eff = {
    "AC": [0.2, 0.3, 0.5, 0.7],
    "DHW tank": [0.5, 0.55, 0.6, 0.65],
    "Buffer tank": [0, 0.05, 0.15, 0.25],
    "PV": [0.02, 0.15, 0.4, 0.6],
    "Battery": [0.1, 0.12, 0.2, 0.3],
    "Prosumager": [0, 0.1, 0.3, 0.5],
    }


# weak policy
heating_leeuwarden_moderate_eff = {
    "Air HP": [0.04, 0.18, 0.45, 0.6],
    "Ground HP": [0, 0.02, 0.06, 0.1],
    "Direct electric heating": [0.02, 0.03, 0.02, 0.02],
    "Conventional": [0.9, 0.73, 0.43, 0.24],
} 
devices_leeuwarden_moderate_eff = {
    "AC": [0.2, 0.35, 0.6, 0.8],
    "DHW tank": [0.5, 0.55, 0.6, 0.65],
    "Buffer tank": [0, 0.05, 0.15, 0.25],
    "PV": [0.02, 0.1, 0.3, 0.5],
    "Battery": [0.1, 0.12, 0.16, 0.25],
    "Prosumager": [0, 0.1, 0.2, 0.4],
    }



def add_no_heating_to_heating_dict(heating_dict: dict) -> dict:
    no_heating = []
    for i, year in enumerate(YEARS):
        a = 1
        for lst in [lst for name, lst in heating_dict.items()]:
            a -= lst[i]
        no_heating.append(a)

    heating_dict["No heating"] = no_heating
    return heating_dict



def line_plot(dictionary: dict, graphik_name: str, stacked: bool):
    if stacked:
        kind="area"
        alpha=0.4
        linestyles=['-', '-', '-', '-', '-', ]
    else:
        kind="line"
        alpha=1
        linestyles = ['-', '--', '-.', ':', '-D', '-x']
    # Create a DataFrame
    df = pd.DataFrame(dictionary, index=YEARS)
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd'   # purple
    ]
    # Plotting the DataFrame
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind=kind, stacked=stacked, ax=ax, alpha=alpha, style=linestyles)  # Adjust transparency with alpha

    # Adding labels and title
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('Percentage', fontsize=18)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(YEARS, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize='large')

    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") /
                f"{graphik_name}.png")
    plt.close()



line_plot(add_no_heating_to_heating_dict(heating_murcia_high_eff), graphik_name="Murcia_high_eff_heating_systems", stacked=True)
line_plot(add_no_heating_to_heating_dict(heating_murcia_moderate_eff), graphik_name="Murcia_moderate_eff_heating_systems", stacked=True)

line_plot(devices_murcia_high_eff, graphik_name="Devices_murcia_high_eff", stacked=False)
line_plot(devices_murcia_moderate_eff, graphik_name="Devices_murcia_low_eff", stacked=False)



line_plot(add_no_heating_to_heating_dict(heating_leeuwarden_high_eff), graphik_name="Leeuwarden_strong_policy_heating_systems", stacked=True)
line_plot(add_no_heating_to_heating_dict(heating_leeuwarden_moderate_eff), graphik_name="Leeuwarden_weak_policy_heating_systems", stacked=True)

line_plot(devices_leeuwarden_high_eff, graphik_name="Devices_Leeuwarden_Strong_policy", stacked=False)
line_plot(devices_leeuwarden_moderate_eff, graphik_name="Devices_Leeuwarden_Weak_policy", stacked=False)

def calculate_percentage_of_categorial_data(df, column):
    new_df = pd.DataFrame()
    for region, group in df.groupby(['region']):
        it = pd.DataFrame(group.value_counts(column), columns=["numbers"]).reset_index()
        it["percentage"] = it["numbers"] / it["numbers"].sum()
        it["region"] = region
        it.drop(columns="numbers", inplace=True)
        new = it.pivot(columns="type", values="percentage", index="region")
        new_df = pd.concat([new_df, new], axis=0)
    return new_df



def show_building_attributes():
    murcia_df = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") / f"2020_high_eff_combined_building_df_Leeuwarden_non_clustered.xlsx", 
                          engine="openpyxl")
    murcia_df["region"] = "Murcia"
    leeuwarden_df = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") / f"2020_high_eff_combined_building_df_Murcia_non_clustered.xlsx", 
                          engine="openpyxl")
    leeuwarden_df["region"] = "Leeuwarden"
    
    common_columns = murcia_df.columns.intersection(leeuwarden_df.columns)
    df = pd.concat([murcia_df[common_columns], leeuwarden_df[common_columns]], axis=0)

    df.boxplot(by="region", column="area")
    plt.ylabel("floor area (m$^2$)")
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "ECEMF_building_areas_boxplot.png")
    plt.close()

    category1_pct = calculate_percentage_of_categorial_data(df, 'type')
    category1_pct.plot(kind="bar", stacked=True)
    plt.ylabel("percentage")
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "ECEMF_SFH_MFH_percentage_barplot.png")
    plt.close()

show_building_attributes()


