import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib.ticker import PercentFormatter, FuncFormatter
import matplotlib
import plotly.express as px
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.lines import Line2D

YEARS = [2020, 2030, 2040, 2050]

# data that is the same for both case studies and for both policy scenarios:
SAME_FOR_ALL = {
    "DHW tank": [0.5, 0.55, 0.6, 0.65],
    "Buffer tank": [0, 0.05, 0.15, 0.25],
    "Battery weak policy": [0.1, 0.12, 0.16, 0.25],
    "Battery strong policy": [0.1, 0.12, 0.2, 0.3],
    "Prosumager low": [0, 0.05, 0.1, 0.2],
    "Prosumager medium": [0, 0.1, 0.3, 0.5],
    "Prosumager high": [0, 0.15, 0.4, 0.8],
}

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
    "PV": [0.015, 0.15, 0.4, 0.6],
    "Battery": [0.1, 0.12, 0.2, 0.3],
    "Prosumager low": [0, 0.05, 0.1, 0.2],
    "Prosumager medium": [0, 0.1, 0.3, 0.5],
    "Prosumager high": [0, 0.15, 0.4, 0.8],
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
    "PV": [0.015, 0.1, 0.3, 0.5],
    "Battery": [0.1, 0.12, 0.16, 0.25],
    "Prosumager low": [0, 0.05, 0.1, 0.2],
    "Prosumager medium": [0, 0.1, 0.3, 0.5],
    "Prosumager high": [0, 0.15, 0.4, 0.8],
    }

Low = [0, 0.05, 0.1, 0.2]
Medium = [0, 0.1, 0.3, 0.5]
High = [0, 0.15, 0.4, 0.8]
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
    "Prosumager low": [0, 0.05, 0.1, 0.2],
    "Prosumager medium": [0, 0.1, 0.3, 0.5],
    "Prosumager high": [0, 0.15, 0.4, 0.8],

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
    "Prosumager low": [0, 0.05, 0.1, 0.2],
    "Prosumager medium": [0, 0.1, 0.3, 0.5],
    "Prosumager high": [0, 0.15, 0.4, 0.8],
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


def plot_params_same_for_all():
    df = pd.DataFrame(SAME_FOR_ALL, index=YEARS)
    prosumer_columns = [name for name in df.columns if "prosumager" in name.lower()]
    linestyles = [':o', '-.o', '--d', '-x', ]
    fig, ax = plt.subplots(figsize=(10, 6))
    years = df.index.to_numpy()
    bar_width = 2
    ax.set_xticks(years)
    colors = ['darkturquoise', 'blue', 'indianred', 'tomato']
    df.drop(columns=prosumer_columns).plot(kind="line", stacked=False, ax=ax, alpha=1, style=linestyles, color=colors)  # Adjust transparency with alpha
    ax.bar(years - bar_width, df['Prosumager low'], width=bar_width, alpha=0.5, label='Prosumager low', color="greenyellow")
    ax.bar(years, df['Prosumager medium'], width=bar_width, alpha=0.5, label='Prosumager medium', color="limegreen")
    ax.bar(years + bar_width, df['Prosumager high'], width=bar_width, alpha=0.5, label='Prosumager high', color="darkgreen")
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('Percentage of buildings equiped \n with certain technology', fontsize=18)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(YEARS, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize='large')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") /
                f"Scenario_independent_parameters_ecemf.svg")
    plt.close()

def show_pv_and_ac_change_over_all_scenarios():
    new_dict = {
        "AC strong policy Murcia": [value for _, value in devices_murcia_high_eff.items() if "AC" in _][0],
        "AC weak policy Murcia": [value for _, value in devices_murcia_moderate_eff.items() if "AC" in _][0],
        "AC strong policy Leeuwarden": [value for _, value in devices_leeuwarden_high_eff.items() if "AC" in _][0],
        "AC weak policy Leeuwarden": [value for _, value in devices_leeuwarden_moderate_eff.items() if "AC" in _][0],
        "PV strong policy Leeuwarden": [value for _, value in devices_leeuwarden_high_eff.items() if "PV" in _][0],
        "PV weak policy Leeuwarden": [value for _, value in devices_leeuwarden_moderate_eff.items() if "PV" in _][0],
        "PV strong policy Murcia": [value for _, value in devices_murcia_high_eff.items() if "PV" in _][0],
        "PV weak policy Murcia": [value for _, value in devices_murcia_moderate_eff.items() if "PV" in _][0],
    }


    def create_df(dictionary):
        df_ = pd.DataFrame(dictionary, index=YEARS).reset_index().rename(columns={"index": "year"}).melt(var_name="policy scenario", value_name="percentage", id_vars="year")
        df_["region"] = df_["policy scenario"].apply(lambda x: "Murcia" if "Murcia" in x else "Leeuwarden")
        df_["technology"] = df_["policy scenario"].apply(lambda x: "PV" if "PV" in x else "AC")
        df_["policy scenario"] = df_["policy scenario"].apply(lambda x: x.replace(" Leeuwarden", "").replace("PV ", "").replace(" Murcia", "").replace("AC ", ""))
        return df_

    df = create_df(new_dict)

    fig = px.bar(data_frame=df,
            x="year",
            y="percentage",
            color="technology",
            pattern_shape="policy scenario",
            barmode="group",
            facet_col="region",
            color_discrete_sequence=px.colors.qualitative.G10
            )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
    yaxis=dict(
        tickformat=".0%",
        title="Percentage of buildings equiped with certain technology"
    ))
    fig.show()
    fig.write_image(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") /
                f"PV_and_AC_percentages_ecemf.svg", engine="kaleido")


def line_plot(dictionary: dict, graphik_name: str, stacked: bool):
    if stacked:
        kind="area"
        alpha=0.4
        linestyles=['-', '-', '-', '-', '-', ]
    else:
        kind="line"
        alpha=1
        linestyles = [':o', '--', '-.', '-x', '--D',]
    # Create a DataFrame
    df = pd.DataFrame(dictionary, index=YEARS)
    prosumer_columns = [name for name in df.columns if "prosumager" in name.lower()]
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd'   # purple
    ]
    # Plotting the DataFrame
    fig, ax = plt.subplots(figsize=(10, 6))
    if stacked:
        df.plot(kind=kind, stacked=stacked, ax=ax, alpha=alpha, style=linestyles)  # Adjust transparency with alpha

    else:
        years = df.index.to_numpy()
        bar_width = 2
        ax.set_xticks(years)
        df.drop(columns=prosumer_columns).plot(kind=kind, stacked=stacked, ax=ax, alpha=alpha, style=linestyles)  # Adjust transparency with alpha
        ax.bar(years - bar_width, df['Prosumager low'], width=bar_width, alpha=0.5, label='Prosumager low')
        ax.bar(years, df['Prosumager medium'], width=bar_width, alpha=0.5, label='Prosumager medium')
        ax.bar(years + bar_width, df['Prosumager high'], width=bar_width, alpha=0.5, label='Prosumager high')
 
        # ax2 = ax.twinx()
        # ax2.set_xticks(df.index)
        # df[prosumer_columns].plot(kind="bar", ax=ax2, alpha=alpha,)  # Adjust transparency with alpha
        
    # Adding labels and title
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('Percentage share of heating systems (%)', fontsize=18)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(YEARS, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize='large')

    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") /
                f"{graphik_name}.svg")
    plt.close()



# 

def calculate_percentage_of_categorial_data(df, column):
    new_df = pd.DataFrame()
    for region, group in df.groupby('region'):
        it = pd.DataFrame(group.value_counts(column), columns=["numbers"]).reset_index()
        it["percentage"] = it["numbers"] / it["numbers"].sum()
        it["region"] = region
        it.drop(columns="numbers", inplace=True)
        new = it.pivot(columns="type", values="percentage", index="region")
        new_df = pd.concat([new_df, new], axis=0)
    return new_df



def show_building_attributes():
    # TODO these are the wrong files, they get cleaned again when creating the scenarios
    murcia_df = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") / f"2020_H_combined_building_df_Leeuwarden_non_clustered.xlsx", 
                          engine="openpyxl")
    murcia_df["region"] = "Murcia"
    leeuwarden_df = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") / f"2020_H_combined_building_df_Murcia_non_clustered.xlsx", 
                          engine="openpyxl")
    leeuwarden_df["region"] = "Leeuwarden"
    
    common_columns = murcia_df.columns.intersection(leeuwarden_df.columns)
    df = pd.concat([murcia_df[common_columns], leeuwarden_df[common_columns]], axis=0)

    df.boxplot(by="region", column="area")
    plt.suptitle("")
    plt.title("")
    plt.ylabel("floor area (m$^2$)")
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "ECEMF_building_areas_boxplot.png")
    plt.close()

    df.boxplot(by="region", column="percentage attached surface area")
    plt.suptitle("")
    plt.title("")
    plt.ylabel("percentage of attached wall area")
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "ECEMF_attached_wall_boxplot.png")
    plt.close()

    category1_pct = calculate_percentage_of_categorial_data(df, 'type')
    category1_pct.plot(kind="bar", stacked=True)
    plt.ylabel("percentage share")
    plt.xticks(rotation="horizontal")
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "ECEMF_SFH_MFH_percentage_barplot.png")
    plt.close()


def prepare_final_df(path_2_file, sheetname):
    df = pd.read_excel(path_2_file, sheet_name=sheetname, header=[0, 1, 2])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Reshape the DataFrame
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    melted_df = pd.melt(df, id_vars=['Policy_Prosumagers_EVs'], var_name='policy_prosumager_ev', value_name='percentage increase (%)')
    melted_df[['Policy scenario', 'Prosumager scenario', "EV scenario"]] = melted_df['policy_prosumager_ev'].str.split('_', expand=True)
    melted_df.drop(columns="policy_prosumager_ev", inplace=True)
    melted_df['Policy scenario'] = melted_df['Policy scenario'].replace({"Weak": "Weak Policy", "Strong": "Strong Policy"})
    melted_df['Prosumager scenario'] = melted_df['Prosumager scenario'].replace({"High": "Prosumager-High", "Medium":  "Prosumager-Medium", "Low":  "Prosumager-Low"})

    final_df = melted_df.rename(columns={"Policy_Prosumagers_EVs": "year"})
    final_df = final_df.loc[final_df.loc[:, "year"] != 2020, :]
    final_df = final_df.sort_values("year", ascending=False)
    final_df["percentage increase (%)"] = final_df["percentage increase (%)"] /100
    final_df['year'] = final_df['year'].astype(str)
    return final_df


def plot_murcia_results():
    path2data = Path(__file__).parent / "Comillas_results_Murica.xlsx"
    for name in ["Low Voltage", "Medium Voltage", "Transformers", "Power losses", "LV MV grid"]:
        final_df = prepare_final_df(path_2_file=path2data, sheetname=name)
        barplot_comillas_results(df=final_df, region="Murcia", data_path=path2data, name=name)

def calculate_diff_between_years(df):
    unique_groups = df[['Policy scenario', 'Prosumager scenario', "Ev scenario"]].drop_duplicates()
    new_dfs = []
    for i, (policy, prosumager, ev) in enumerate(unique_groups.values):
        group_df = df[(df['Policy scenario'] == policy) & (df['Prosumager scenario'] == prosumager) & (df['Ev scenario'] == ev)]
        new = group_df.copy()

        for years in [[2020, 2030], [2030, 2040], [2040, 2050]]:
            new.loc[group_df["Year"]==years[1], "Load in (mw)"] = abs(group_df.loc[group_df["Year"]==years[1], "Load in (mw)"].values[0] - group_df.loc[group_df["Year"]==years[0], "Load in (mw)"].values[0])
        new_dfs.append(new)
    return pd.concat(new_dfs)

def barplot_peak_demand(df: pd.DataFrame, region: str, data_path: Path, name: str):
    matplotlib.rc("font", **{"size": 28})
    # Define unique groups for policy and prosumager scenarios
    df.columns = [i.capitalize() for i in df.columns]
    
    df_plot = df.copy()#calculate_diff_between_years(df)
    unique_groups = df_plot[['Policy scenario', 'Prosumager scenario', "Ev scenario"]].drop_duplicates()

    palette = sns.color_palette("pastel6", len(df_plot['Year'].unique()))

    x_labels = ["low", "medium", "high", "low", "medium", "high"]
    color_year = {2020: palette[3], 2030: palette[1], 2040: palette[2], 2050: palette[0]}
    position_year = {2020: -0.6, 2030: -0.2, 2040: 0.2, 2050: 0.6}
    fig, ax = plt.subplots(figsize=(20, 16))
    for i, (policy, prosumager, ev) in enumerate(unique_groups.values):
        group_df = df_plot[(df_plot['Policy scenario'] == policy) & (df_plot['Prosumager scenario'] == prosumager) & (df_plot['Ev scenario'] == ev)]
        bottom = 0
        for j, year in enumerate(group_df['Year'].unique()):
            data = group_df[group_df['Year'] == year]
            
            if "low" in prosumager.lower():

                if "no" in ev.lower():
                    position = 0 - 0.3
                else:
                    position = 6 + 0.3
            elif "medium" in prosumager.lower():

                if "no" in ev.lower():
                    position = 2 - 0.3
                else:
                    position = 8 + 0.3

            else:

                if "no" in ev.lower():
                    position = 4 - 0.3
                else:
                    position = 10 + 0.3
                    
            if "weak" in policy.lower():
                hatch = "//"
            else:
                position += 0.2
                hatch = ""

            
            color = mcolors.to_rgba(color_year[year], alpha=1)
            # color_year[j] = year
            ax.bar(position+position_year[year], data['Load in (mw)'].values[0], color=color, edgecolor='black', label=year if i == 0 else "", width=0.2, hatch=hatch, bottom=0)
            bottom += data['Load in (mw)'].values[0]

    # Add labels and title
    ax.set_xticks([-0.2, 1.8, 3.8, 6.4, 8.4, 10.4])
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel("Prosumager scenario")

    ax.set_ylabel(name)
    legend_elements = [
        Patch(facecolor=color_year[2050], label="2050"),
        Patch(facecolor=color_year[2040], label="2040"),
        Patch(facecolor=color_year[2030], label="2030"),
        Patch(facecolor=color_year[2020], label="2020"),
        Patch(facecolor="white", hatch="///", label="weak Policy", edgecolor="black"),
        Patch(facecolor="white", hatch="", label="strong Policy", edgecolor="black"),
        ]
    ax.legend(handles=legend_elements)#, loc='upper left')#, bbox_to_anchor=(1.05, 1))

    ax2 = ax.twiny()
    ax2.set_xticks([0.25, 0.75])
    ax2.set_xticklabels(["without EV", "with EV"])

    plt.tight_layout()
    fig.savefig(data_path / f"{name.replace(' ', '_')}_{region}.svg")
    plt.show()


def barplot_comillas_results(df: pd.DataFrame, region: str, data_path: Path, name: str):
    matplotlib.rc("font", **{"size": 28})
    # Define unique groups for policy and prosumager scenarios
    unique_groups = df[['Policy scenario', 'Prosumager scenario', "EV scenario"]].drop_duplicates()
    # Define a color palette
    palette = sns.color_palette("pastel6", len(df['year'].unique()))

    x_labels = ["low", "medium", "high", "low", "medium", "high"]
    color_year = {2030: palette[1], 2040: palette[2], 2050: palette[0]}
    position_year = {2020: -0.6, 2030: -0.2, 2040: 0.2, 2050: 0.6}
    # Plot each year as a separate bar, stacked by the 'percentage increase (%)'
    fig, ax = plt.subplots(figsize=(20, 16))
    for i, (policy, prosumager, ev) in enumerate(unique_groups.values):
        group_df = df[(df['Policy scenario'] == policy) & (df['Prosumager scenario'] == prosumager) & (df['EV scenario'] == ev)]

        # weak policy is in 0-5, strong policy in positions 6-11
        for j, year in enumerate(group_df['year'].unique()):
            data = group_df[group_df['year'] == year]
            
            if "low" in prosumager.lower():
                if "no" in ev.lower():
                    position = 0 - 0.3
                else:
                    position = 6 + 0.3
            elif "medium" in prosumager.lower():

                if "no" in ev.lower():
                    position = 2 - 0.3
                else:
                    position = 8 + 0.3
            else:

                if "no" in ev.lower():
                    position = 4 - 0.3
                else:
                    position = 10 + 0.3
            
            if "weak" in policy.lower():
                hatch = "//"
            else:
                position += 0.2
                hatch = ""

            color = mcolors.to_rgba(palette[j], alpha=1)
            ax.bar(position+position_year[int(year)], data['percentage increase (%)'].values[0], bottom=0, color=color, edgecolor='black', label=year if i == 0 else "", width=0.2, hatch=hatch)

    # Add labels and title
    ax.set_xticks([0, 2, 4, 6.6, 8.6, 10.6])

    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel("Prosumager scenario")
    if "Voltage" in name:
        add = "lines"
    else:
        add = ""
    if "loss" in name.lower():
        y_label = "Incremental increase in energy losses (%)"
    elif "grid" in name.lower():
        y_label = "Incremental cost increase in distribution grid (%)"
    else:
        y_label = f'Incremental cost increase in {name} {add} (%)'.replace("MV-LV", "")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    legend_elements = [
        Patch(facecolor=palette[0], label="2050"),
        Patch(facecolor=palette[1], label="2040"),
        Patch(facecolor=palette[2], label="2030"),
        Patch(facecolor="white", hatch="///", label="weak Policy", edgecolor="black"),
        Patch(facecolor="white", hatch="", label="strong Policy", edgecolor="black"),
        ]
    ax.legend(handles=legend_elements, loc='upper left')#, bbox_to_anchor=(1.05, 1))

    ax2 = ax.twiny()
    ax2.set_xticks([0.25, 0.75])
    ax2.set_xticklabels(["without EV", "with EV"])
    plt.tight_layout()
    plt.savefig(data_path.parent / f"{region}_results_{name}.svg".replace(" ", "_"))
    plt.close()

def plot_leeuwarden_results():
    path2data = Path(__file__).parent / "Comillas_results_Leeuwarden.xlsx"
    for name in ["Low Voltage", "Medium Voltage", "High Voltage", "MV-LV Transformers", "HV-MV substations", "Power losses", "LV MV grid"]:
        final_df = prepare_final_df(path_2_file=path2data, sheetname=name)
        barplot_comillas_results(df=final_df, region="Leeuwarden", data_path=path2data, name=name)


# def grab_ev_peaks():
#     murcia = pd.read_csv(Path(__file__).parent / "EV_aggregated_profiles_Murcia.csv")
#     leeuwarden = pd.read_csv(Path(__file__).parent / "EV_aggregated_profiles_Leeuwarden.csv")
#     df = pd.concat([murcia, leeuwarden]).rename(columns={"col_region": "Region", "col_policy": "policy scenario", "col_prosum": "prosumager scenario", "col_year": "year", "ev_peak": "ev peak (MW)"})
#     df.loc[:, "ev peak (MW)"] = df.loc[:, "ev peak (MW)"]  / 1_000
#     columns_2_drop = [col for col in df.columns if "profiles" in col]
#     df.drop(columns=columns_2_drop, inplace=True)
#     return df

def grab_ev_demand_at_feed_day():
    murcia = pd.read_csv(Path(__file__).parent / "EV_aggregated_profiles_Murcia.csv")
    leeuwarden = pd.read_csv(Path(__file__).parent / "EV_aggregated_profiles_Leeuwarden.csv")
    df = pd.concat([murcia, leeuwarden]).rename(columns={"col_region": "Region", "col_policy": "policy scenario", "col_prosum": "prosumager scenario", "col_year": "year", "ev_peak": "ev load (MW)"})
    columns_2_keep = [col for col in df.columns if not "profiles" in col]
    columns = []
    for col in df.columns:
        if "profile" in col:
            if 25 <= int(col.split("_")[-1]) <=48:
                columns.append(col)
    df_feed = df.loc[:, columns_2_keep + columns].copy()
    df_feed.drop(columns=["ev load (MW)"], inplace=True)  # drop the single peaks
    # change the name of the hour columns:
    replace_column_names_dict = {}
    for i, col in enumerate(columns, start=1):
        replace_column_names_dict[col] = i
    df_feed.rename(columns=replace_column_names_dict, inplace=True)

    id_columns = [col for col in columns_2_keep if col != "ev load (MW)"]
    df_feed_final = df_feed.melt(id_vars=id_columns, var_name="hours", value_name="load in (MW)")
    df_feed_final.loc[:, "load in (MW)"] = df_feed_final.loc[:, "load in (MW)"]  / 1_000

    # add 2020 values with 0
    df_2020 = df_feed_final.query("year == 2030").copy()
    df_2020.loc[:, "year"] = 2020
    df_2020.loc[:, "load in (MW)"] = 0

    df_final = pd.concat([df_feed_final, df_2020])
    return df_final

def grab_ev_demand_at_demand_day():
    murcia = pd.read_csv(Path(__file__).parent / "EV_aggregated_profiles_Murcia.csv")
    leeuwarden = pd.read_csv(Path(__file__).parent / "EV_aggregated_profiles_Leeuwarden.csv")
    df = pd.concat([murcia, leeuwarden]).rename(columns={"col_region": "Region", "col_policy": "policy scenario", "col_prosum": "prosumager scenario", "col_year": "year", "ev_peak": "ev load (MW)"})
    columns_2_keep = [col for col in df.columns if not "profiles" in col]
    columns = []
    for col in df.columns:
        if "profile" in col:
            if 1 <= int(col.split("_")[-1]) <=24:
                columns.append(col)
    df_feed = df.loc[:, columns_2_keep + columns].copy()
    df_feed.drop(columns=["ev load (MW)"], inplace=True)  # drop the single peaks
    # change the name of the hour columns:
    replace_column_names_dict = {}
    for i, col in enumerate(columns, start=1):
        replace_column_names_dict[col] = i
    df_feed.rename(columns=replace_column_names_dict, inplace=True)

    id_columns = [col for col in columns_2_keep if col != "ev load (MW)"]
    df_feed_final = df_feed.melt(id_vars=id_columns, var_name="hours", value_name="load in (MW)")
    df_feed_final.loc[:, "load in (MW)"] = df_feed_final.loc[:, "load in (MW)"]  / 1_000

    # add 2020 values with 0
    df_2020 = df_feed_final.query("year == 2030").copy()
    df_2020.loc[:, "year"] = 2020
    df_2020.loc[:, "load in (MW)"] = 0

    df_final = pd.concat([df_feed_final, df_2020])
    return df_final

def grab_demand_peaks():
    murcia = pd.read_csv(Path(__file__).parent / "Murcia_peak_demand_day.csv")
    leeuwarden = pd.read_csv(Path(__file__).parent / "Leeuwarden_peak_demand_day.csv")
    murcia.loc[:, "Region"] = "Murcia"
    leeuwarden.loc[:, "Region"] = "Leeuwarden"
    murcia.loc[:, "hours"] += 1
    leeuwarden.loc[:, "hours"] += 1
    df = pd.concat([murcia, leeuwarden]).drop(columns="variable")
    
    no_ev = calc_peak_values(df=df)
    no_ev.loc[:, "EV scenario"] = "no EV"

    # calculate profiles with added EV profiles:
    values_to_sort_by = ["hours", "policy scenario", "prosumager scenario", "Region", "year"]
    ev_demand = grab_ev_demand_at_demand_day()
    ev_demand.sort_values(by=values_to_sort_by, inplace=True)
    df.sort_values(by=values_to_sort_by, inplace=True)

    df.loc[:, "load in (MW)"] = df.loc[:, "load in (MW)"].values + ev_demand.loc[:, "load in (MW)"].values
    with_ev = calc_peak_values(df=df)
    with_ev.loc[:, "EV scenario"] = "with EV"
    return pd.concat([no_ev, with_ev], axis=0)

def grab_house_feed_day():
    murcia = pd.read_csv(Path(__file__).parent / "Murcia_peak_feed_day.csv")
    leeuwarden = pd.read_csv(Path(__file__).parent / "Leeuwarden_peak_feed_day.csv")
    murcia.loc[:, "hours"] -= 71
    leeuwarden.loc[:, "hours"] -= 71
    murcia.loc[:, "Region"] = "Murcia"
    leeuwarden.loc[:, "Region"] = "Leeuwarden"
    df = pd.concat([murcia, leeuwarden]).drop(columns="variable")
    return df

def calc_negative_peak_values(df):
     # now pick the feed peak
    new_df = pd.DataFrame(columns=df.columns)
    for i, (names, group) in enumerate(df.groupby(["policy scenario", "prosumager scenario", "Region", "year"])):
        new_df.loc[i, ["policy scenario", "prosumager scenario", "Region", "year"]] = list(names)
        new_df.loc[i, "load in (MW)"] = group.loc[:, "load in (MW)"].min()  # we need the highest feed in which is negative
    new_df.drop(columns="hours", inplace=True)
    return new_df

def calc_peak_values(df):
     # now pick the feed peak
    new_df = pd.DataFrame(columns=df.columns)
    for i, (names, group) in enumerate(df.groupby(["policy scenario", "prosumager scenario", "Region", "year"])):
        new_df.loc[i, ["policy scenario", "prosumager scenario", "Region", "year"]] = list(names)
        new_df.loc[i, "load in (MW)"] = group.loc[:, "load in (MW)"].max()  # we need the highest feed in which is negative
    new_df.drop(columns="hours", inplace=True)
    return new_df

def grab_feed_in_peaks():
    ev_demand_feed_day = grab_ev_demand_at_feed_day()
    values_to_sort_by = ["hours", "policy scenario", "prosumager scenario", "Region", "year"]
    ev_demand_feed_day.sort_values(by=values_to_sort_by, inplace=True)

    houses_demand_feed_day = grab_house_feed_day()
    houses_demand_feed_day.sort_values(by=values_to_sort_by, inplace=True)
    houses_demand_feed_no_ev = houses_demand_feed_day.copy()
    no_ev = calc_negative_peak_values(df=houses_demand_feed_no_ev)  # maximum feed in is negative
    no_ev.loc[:, "EV scenario"] = "no EV"

    houses_demand_feed_day.loc[:, "load in (MW)"] = houses_demand_feed_day.loc[:, "load in (MW)"].values + ev_demand_feed_day.loc[:, "load in (MW)"].values

    with_ev = calc_negative_peak_values(df=houses_demand_feed_day)
    with_ev.loc[:, "EV scenario"] = "with EV"
    return pd.concat([no_ev, with_ev], axis=0)

def plot_peak_demand_as_line_plot(df, demand_or_feed: str, zoom: bool=False):
    if demand_or_feed == "demand":
        legend_locs = ["upper left", "upper right"]
    else:
        legend_locs = ["lower left", "lower left"]
    matplotlib.rc("font", **{"size": 28})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 18), sharey=True)
    
    # Define a color palette
    palette = sns.color_palette("hls", len(df['prosumager scenario'].unique()))

    x_labels = ["2020", "2030", "2040", "2050"]
    # Define unique groups for policy and prosumager scenarios
    unique_groups = df[['policy scenario', 'prosumager scenario', "EV scenario"]].drop_duplicates()
    for r, region in enumerate(["Leeuwarden", "Murcia",]):
        for i, (policy, prosumager, ev) in enumerate(unique_groups.values):
            group_df = df[(df['policy scenario'] == policy) & (df['prosumager scenario'] == prosumager) & (df['EV scenario'] == ev) & (df["Region"] == region)]
            ax = axes[r]
            y_values = group_df["load in (MW)"].to_numpy()

            if "high" in group_df["prosumager scenario"].values[0].lower():
                color = palette[0]
            elif "medium" in group_df["prosumager scenario"].values[0].lower():
                color = palette[1]
            else:
                color = palette[2]
            if group_df["EV scenario"].values[0] == "with EV":
                linestyle = "--"
            else:
                linestyle = "-"
            if "weak" in group_df["policy scenario"].values[0].lower():
                marker = "o"
            else:
                marker = "X"

            ax.plot(x_labels, y_values, color=color, linestyle=linestyle, marker=marker, linewidth=1.2, markersize=10)

            # Zoom window for Leeuwarden:
            if r == 0 and i ==0 and zoom:
                # inset Axes....
                x1, x2, y1, y2 = 1.9, 3.1, 45, 60  # subregion of the original image
                axins = ax.inset_axes(
                    [0.07, 0.35, 0.4, 0.4],  # linker rand,  unterer rand, rechter rand, oberer rand
                    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[2040, 2050], yticklabels=[45, 50, 55, 60])
                rect = (x1, y1, x2 - x1, y2 - y1)
                box = ax.indicate_inset(rect, edgecolor="black", alpha=1, lw=0.7)

                cp1 = ConnectionPatch(xyA=(x1, y1), xyB=(0, 0), axesA=ax, axesB=axins,
                    coordsA="data", coordsB="axes fraction", lw=0.7, ls=":")
                cp2 = ConnectionPatch(xyA=(x2, y2), xyB=(1, 1), axesA=ax, axesB=axins,
                    coordsA="data", coordsB="axes fraction", lw=0.7, ls=":")
                ax.add_patch(cp1)
                ax.add_patch(cp2)
            if r == 0 and zoom:
                axins.plot(x_labels, y_values, color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=8)
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_yticks([45, 50, 55, 60])
                axins.set_xticks([2, 3])
                # ax.indicate_inset_zoom(axins, edgecolor="grey", linewidth=1, linestyle="-",)
                

                
                axins.tick_params(axis='x', labelsize=15)  # Adjust the label size as needed
                axins.tick_params(axis='y', labelsize=15) 

            # Zoom window for Murcia:
            if r == 1 and i == 0 and zoom:
                 # inset Axes....
                x1, x2, y1, y2 = 1.9, 3.1, 89, 100  # subregion of the original image
                axins2 = ax.inset_axes(
                    [0.07, 0.35, 0.4, 0.4],  # linker rand,  unterer rand, rechter rand, oberer rand
                    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[2040, 2050], yticklabels=[90, 95, 100])
                rect = (x1, y1, x2 - x1, y2 - y1)
                box = ax.indicate_inset(rect, edgecolor="black", alpha=1, lw=0.7)

                cp1 = ConnectionPatch(xyA=(x1, y1), xyB=(0, 0), axesA=ax, axesB=axins2,
                    coordsA="data", coordsB="axes fraction", lw=0.7, ls=":")
                cp2 = ConnectionPatch(xyA=(x2, y2), xyB=(1, 1), axesA=ax, axesB=axins2,
                    coordsA="data", coordsB="axes fraction", lw=0.7, ls=":")
                ax.add_patch(cp1)
                ax.add_patch(cp2)
            if r == 1 and zoom:
                axins2.plot(x_labels, y_values, color=color, linestyle=linestyle, marker=marker, linewidth=1.2, markersize=8)
                axins2.set_xlim(x1, x2)
                axins2.set_ylim(y1, y2)
                axins2.set_yticks([90, 95, 100])
                axins2.set_xticks([2, 3])
                # ax.indicate_inset_zoom(axins2, edgecolor="grey", linewidth=1, linestyle="-")
                axins2.tick_params(axis='x', labelsize=15)  # Adjust the label size as needed
                axins2.tick_params(axis='y', labelsize=15) 

    
        axes[r].set_title(region)
    legend_elements = [
                Line2D([0],[0], color=palette[0], label="High Prosumager share", linewidth=4),
                Line2D([0],[0], color=palette[1], label="Medium Prosumager share", linewidth=4),
                Line2D([0],[0], color=palette[2], label="Low Prosumager share", linewidth=4),
                Line2D([0],[0], color="black", linestyle="-", label="without EV", linewidth=4),
                Line2D([0],[0], color="black", linestyle="--", label="with EV", linewidth=4),
                Line2D([0],[0], color="black", linestyle="", marker="o",label="Weak policy", markersize=15),
                Line2D([0],[0], color="black", linestyle="", marker="X",label="Strong policy", markersize=15),
                ]



    legend_elements_1 = [
        Line2D([0], [0], color=palette[0], label="High Prosumager share", linewidth=4),
        Line2D([0], [0], color=palette[1], label="Medium Prosumager share", linewidth=4),
        Line2D([0], [0], color=palette[2], label="Low Prosumager share", linewidth=4),
    ]

    legend_elements_2 = [
        Line2D([0], [0], color="black", linestyle="-", label="without EV", linewidth=4),
        Line2D([0], [0], color="black", linestyle="--", label="with EV", linewidth=4),
    ]

    legend_elements_3 = [
        Line2D([0], [0], color="black", linestyle="", marker="o", label="Weak policy", markersize=15),
        Line2D([0], [0], color="black", linestyle="", marker="X", label="Strong policy", markersize=15),
    ]
        # if r==0:
        #     axes[r].legend(handles=legend_elements, loc=legend_locs[r], fontsize=24)
        # else:
        #     axes[r].legend(handles=legend_elements, loc=legend_locs[r], fontsize=24)

    axes[0].set_ylabel(f"total peak {demand_or_feed} (MW)")
    # Combine legends, each with its own ncol setting
    legend1 = axes[0].legend(handles=legend_elements_1, loc='upper left', bbox_to_anchor=(0, 1.3), ncol=3, fontsize=18)
    legend2 = axes[0].legend(handles=legend_elements_2, loc='upper left', bbox_to_anchor=(0, 1.2), ncol=2, fontsize=18)
    legend3 = axes[0].legend(handles=legend_elements_3, loc='upper left', bbox_to_anchor=(0.8, 1.2), ncol=2, fontsize=18)

    # Add the legends back to the axes (for overlapping legends)
    axes[0].add_artist(legend1)
    axes[0].add_artist(legend2)
    axes[0].add_artist(legend3)

    fig.subplots_adjust(top=0.8)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"peak_{demand_or_feed}.png".replace(" ","_"))
    plt.savefig(Path(__file__).parent / f"peak_{demand_or_feed}.svg".replace(" ","_"))
    plt.close()
   

def plot_peak_total_and_peak_feed_with_EV():
    demand_peaks = grab_demand_peaks()
    feed_in_peaks = grab_feed_in_peaks()

    barplot_peak_demand(df=demand_peaks.query("Region == 'Leeuwarden'"), region="Leeuwarden", data_path=Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken"), name="total peak demand (MW)")
    barplot_peak_demand(df=feed_in_peaks.query("Region == 'Leeuwarden'"), region="Leeuwarden", data_path=Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken"), name="total peak feed (MW)")

    barplot_peak_demand(df=demand_peaks.query("Region == 'Murcia'"), region="Murcia", data_path=Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken"), name="total peak demand (MW)")
    barplot_peak_demand(df=feed_in_peaks.query("Region == 'Murcia'"), region="Murcia", data_path=Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken"), name="total peak feed (MW)")

    # plot_peak_demand_as_line_plot(df=demand_peaks, demand_or_feed="demand", zoom=True)
    # plot_peak_demand_as_line_plot(df=feed_in_peaks, demand_or_feed="feed in", zoom=False)

   



# show_pv_and_ac_change_over_all_scenarios()
# plot_params_same_for_all()

# line_plot(add_no_heating_to_heating_dict(heating_murcia_high_eff), graphik_name="Murcia_H_heating_systems", stacked=True)
# line_plot(add_no_heating_to_heating_dict(heating_murcia_moderate_eff), graphik_name="Murcia_M_heating_systems", stacked=True)

# line_plot(devices_murcia_high_eff, graphik_name="Devices_murcia_high_eff", stacked=False)
# line_plot(devices_murcia_moderate_eff, graphik_name="Devices_murcia_low_eff", stacked=False)



# line_plot(add_no_heating_to_heating_dict(heating_leeuwarden_high_eff), graphik_name="Leeuwarden_strong_policy_heating_systems", stacked=True)
# line_plot(add_no_heating_to_heating_dict(heating_leeuwarden_moderate_eff), graphik_name="Leeuwarden_weak_policy_heating_systems", stacked=True)

# line_plot(devices_leeuwarden_high_eff, graphik_name="Devices_Leeuwarden_Strong_policy", stacked=False)
# line_plot(devices_leeuwarden_moderate_eff, graphik_name="Devices_Leeuwarden_Weak_policy", stacked=False)


# show_building_attributes()


# plot_leeuwarden_results()
# plot_murcia_results()

plot_peak_total_and_peak_feed_with_EV()


