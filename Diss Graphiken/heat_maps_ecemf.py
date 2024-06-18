import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import os
import pickle
import numpy as np

# Set the environment variable to increase the timeout to 5 seconds
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5'

Low = [0, 0.05, 0.1, 0.2]
Medium = [0, 0.1, 0.3, 0.5]
High = [0, 0.15, 0.4, 0.8]
scenario_dict_murcia = {}
for i, prosumager_shares in enumerate([Low, Medium, High]):
    if i == 0:
        pr = "Prosumager-low"
    elif i == 1:
        pr = "Prosumager-medium"
    elif i == 2:
        pr = "Prosumager-high"
    scenario_high_eff = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Murcia",
        "building_scenario": "H",
        "pv_installation_percentage": [0.015, 0.1, 0.4, 0.6],
        "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
        "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
        "heating_element_percentage": 0,
        "air_hp_percentage": [0.2, 0.35, 0.6, 0.8],
        "ground_hp_percentage": [0, 0.02, 0.04, 0.06],
        "direct_electric_heating_percentage": [0.39, 0.3, 0.2, 0.05],
        "gases_percentage": [0.19, 0.15, 0.07, 0.02],
        "ac_percentage": [0.5, 0.6, 0.8, 0.9],
        "battery_percentage": [0.1, 0.12, 0.2, 0.3],
        "prosumager_percentage": prosumager_shares,
    }
    scenario_dict_murcia[f"Strong policy {pr}"] = scenario_high_eff

    scenario_moderate_eff = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Murcia",
        "building_scenario": "M",
        "pv_installation_percentage": [0.015, 0.15, 0.3, 0.5],
        "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
        "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
        "heating_element_percentage": 0,
        "air_hp_percentage": [0.2, 0.3, 0.5, 0.7],
        "ground_hp_percentage": [0, 0.01, 0.02, 0.03],
        "direct_electric_heating_percentage": [0.39, 0.32, 0.2, 0.1],
        "gases_percentage": [0.19, 0.16, 0.1, 0.05],
        "ac_percentage": [0.5, 0.65, 0.8, 0.95],
        "battery_percentage": [0.1, 0.12, 0.16, 0.25],
        "prosumager_percentage": prosumager_shares,
    }
    scenario_dict_murcia[f"Weak policy {pr}"] = scenario_moderate_eff


scenario_dict_leeuwarden = {}
for i, prosumager_shares in enumerate([Low, Medium, High]):
    if i == 0:
        pr = "Prosumager-low"
    elif i == 1:
        pr = "Prosumager-medium"
    elif i == 2:
        pr = "Prosumager-high" 
    scenario_high_eff_leeuwarden = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Leeuwarden",
        "building_scenario": "H",
        "pv_installation_percentage": [0.02, 0.15, 0.4, 0.6],
        "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
        "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
        "heating_element_percentage": 0,
        "air_hp_percentage": [0.04, 0.18, 0.5, 0.7],
        "ground_hp_percentage": [0, 0.05, 0.1, 0.15],
        "direct_electric_heating_percentage": [0.02, 0.03, 0.02, 0.01],
        "gases_percentage": [0.9, 0.7, 0.34, 0.1],
        "ac_percentage": [0.2, 0.3, 0.5, 0.7],
        "battery_percentage": [0.1, 0.12, 0.2, 0.3],
        "prosumager_percentage": prosumager_shares,
    }
    scenario_dict_leeuwarden[f"Strong policy {pr}"] = scenario_high_eff_leeuwarden

    scenario_moderate_eff_leeuwarden = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Leeuwarden",
        "building_scenario": "M",
        "pv_installation_percentage": [0.02, 0.1, 0.3, 0.5],
        "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
        "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
        "heating_element_percentage": 0,
        "air_hp_percentage": [0.04, 0.18, 0.45, 0.6],
        "ground_hp_percentage": [0, 0.02, 0.06, 0.1],
        "direct_electric_heating_percentage": [0.02, 0.03, 0.02, 0.02],
        "gases_percentage": [0.9, 0.73, 0.43, 0.24],
        "ac_percentage": [0.2, 0.35, 0.6, 0.8],
        "battery_percentage": [0.1, 0.12, 0.16, 0.25],
        "prosumager_percentage": prosumager_shares,
    }
    scenario_dict_leeuwarden[f"Weak policy {pr}"] = scenario_moderate_eff_leeuwarden



def get_file_name(dictionary: dict):
    if dictionary["year"] == 2020:  # for 2020 the scenario is always the same
        building_scenario = "H"
    else:
        building_scenario = dictionary['building_scenario']

    return f"{dictionary['year']}_" \
           f"{dictionary['region']}_" \
           f"{building_scenario}_" \
           f"PV-{round(dictionary['pv_installation_percentage'] * 100)}%_" \
           f"DHW-{round(dictionary['dhw_storage_percentage'] * 100)}%_" \
           f"Buffer-{round(dictionary['buffer_storage_percentage'] * 100)}%_" \
           f"HE-{round(dictionary['heating_element_percentage'] * 100)}%_" \
           f"AirHP-{round(dictionary['air_hp_percentage'] * 100)}%_" \
           f"GroundHP-{round(dictionary['ground_hp_percentage'] * 100)}%_" \
           f"directE-{round(dictionary['direct_electric_heating_percentage'] * 100)}%_" \
           f"Conventional-{round(dictionary['gases_percentage'] * 100)}%_" \
           f"AC-{round(dictionary['ac_percentage'] * 100)}%_" \
           f"Battery-{round(dictionary['battery_percentage'] * 100)}%_" \
           f"Prosumager-{round(dictionary['prosumager_percentage'] * 100)}%"


def load_file_from_server(region: str, filename: str):
    path_to_results_on_server = Path(f"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/projects/ECEMF_T4.3_{region}/data_output")
    return pd.read_parquet(path_to_results_on_server / filename)

def load_normed_profiles(region: str, filename: str, variable_name: str):
    df = load_summed_profiles(region=region, filename=filename, variable_name=variable_name)
    # load building component to get Af
    path_to_building_component = Path(f"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/input_operation/ECEMF_T4.3_{region}_2020_H") / "OperationScenario_Component_Building.xlsx"
    building_df = pd.read_excel(path_to_building_component, engine="openpyxl")
    merged = df.merge(building_df.loc[:, ["ID_Building", "Af"]], on="ID_Building")
    merged[variable_name] = merged[variable_name] / merged["Af"]
    merged.drop(columns="Af", inplace=True)
    return merged

def load_summed_profiles(region: str, filename: str, variable_name: str):
    parquet_file = load_file_from_server(region, filename)
    df = pd.DataFrame(parquet_file.sum(axis=0)).reset_index().rename(columns={"index": "ID_Building", 0: variable_name})
    df[variable_name] = df[variable_name] / 1_000  # Wh -> kWh
    df["ID_Building"] = df["ID_Building"].astype(int)
    return df


def find_ax_row_column(scen_name: str, year: int):
    # based on the year the column is selected:
    if year == 2030:
        i_column = 0
    elif year == 2040:
        i_column = 1
    elif year == 2050:
        i_column = 2
    # based on the policy scenario and the prosumager share the row is selected:
    if "weak" in scen_name.lower():
        possible_rows = [0, 1, 2]  # low ,medium, high prosuamgers
    else:
        possible_rows = [3, 4, 5]
    if "low" in scen_name.lower():
        i_row = possible_rows[0]
    elif "medium" in scen_name.lower():
        i_row=possible_rows[1]
    elif "high" in scen_name.lower():
        i_row = possible_rows[2]
    
    return i_row, i_column

def select_cmap(metric: str):
    if "heating" in metric.lower():
        cmap = "Reds"
    elif "cooling" in metric.lower():
        cmap = "Blues"
    elif "demand" in metric.lower():
        cmap = "Greens"
    return cmap


def plot_baseyear(region: str, region_gdf: gpd.GeoDataFrame, scenario: dict, metric: str, variable_name: str, v_dict: dict):
    print(f"creating 2020 plot for {metric}")
    fontsize = 18
    file_name = get_file_name(scenario)
    parquet_file = f"{metric}_{file_name}.parquet.gzip"
    if normed:
        profiles = load_normed_profiles(region=region, 
                                        filename=parquet_file, 
                                        variable_name=variable_name)
    else:
        profiles = load_summed_profiles(region=region,
                                    filename=parquet_file,
                                    variable_name=variable_name)

    gdf = region_gdf.merge(profiles, on="ID_Building")

    vmin, vmax = v_dict[metric]
    c_map = plt.get_cmap(select_cmap(metric=metric))
    cmap_with_under_over = colors.ListedColormap(c_map(np.linspace(0, 1, c_map.N)))
    cmap_with_under_over.set_under('lightgray')
    cmap_with_under_over.set_over('gold')
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # Create a ScalarMappable and set under and over colors
    sm = plt.cm.ScalarMappable(cmap=cmap_with_under_over, norm=norm)
    sm.set_array([])  # Only needed for the colorbar
    
    fig = plt.figure(figsize=(20, 20))
    ax=plt.gca()
    gdf.plot(column=variable_name, 
                legend=False, 
                cmap=cmap_with_under_over,
                norm=norm, 
                ax=ax,
                legend_kwds={'extend':'both'},
                )
    cbar = plt.colorbar(sm, ax=ax, extend='both', fraction=0.035, pad=0.0)
    cbar.ax.set_ylabel(f'{variable_name}', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if normed:
        plt.savefig(Path("Diss Graphiken") / f"Density_map{metric}_{region}_2020_Normed.svg")
    else:
        plt.savefig(Path("Diss Graphiken") / f"Density_map{metric}_{region}_2020.svg")
    plt.close()
    print(f"created 2020 desity map for {metric} {region}")


def create_single_scenarios_from_year_scenarios(year_scens: dict):
    # create single scenarios for each subscenario:
    scenarios = []
    changing_parameters = [(key, value) for key, value in year_scens.items() if isinstance(value, list)]
    for i in range(len(changing_parameters[0][1])):
        new_scen = year_scens.copy()
        for param, values in changing_parameters:
            new_scen[param] = values[i]
        scenarios.append(new_scen)
    return scenarios


def create_big_subplot_with_density_maps(region: str, region_gdf: gpd.GeoDataFrame, scenario_dict: dict, v_dict: dict, metric_dict: dict):
    fontsize = 18
    for metric, variable_name in metric_dict.items():
        
        # create subplots for each metric:
        fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(40, 32))
        
        for j, (scenario_name, year_scenarios) in enumerate(scenario_dict.items()):
            print(f"creating plot for {scenario_name}")

            scenarios = create_single_scenarios_from_year_scenarios(year_scens=year_scenarios)
            for scen in scenarios:
                if scen["year"] == 2020:
                    # plot the baseyear in the first run only:
                    if j == 0:
                        plot_baseyear(region=region, region_gdf=region_gdf, scenario=scen, metric=metric, variable_name=variable_name, v_dict=v_dict)
                    continue
                i_row, i_column = find_ax_row_column(scen_name=scenario_name, year=scen["year"])
                file_name = get_file_name(scen)

                vmin, vmax = v_dict[metric]
                parquet_file = f"{metric}_{file_name}.parquet.gzip"
                if normed:
                    profiles = load_normed_profiles(region=region, 
                                                    filename=parquet_file, 
                                                    variable_name=variable_name)
                else:
                    profiles = load_summed_profiles(region=region,
                                                filename=parquet_file,
                                                variable_name=variable_name)
                gdf = region_gdf.merge(profiles, on="ID_Building")
                c_map = plt.get_cmap(select_cmap(metric=metric))
                cmap_with_under_over = colors.ListedColormap(c_map(np.linspace(0, 1, c_map.N)))
                cmap_with_under_over.set_under('lightgray')
                cmap_with_under_over.set_over('gold')
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                # Create a ScalarMappable and set under and over colors
                sm = plt.cm.ScalarMappable(cmap=cmap_with_under_over, norm=norm)
                sm.set_array([])  # Only needed for the colorbar
                
                gdf.plot(column=variable_name, 
                         cmap=cmap_with_under_over,
                         norm=norm, 
                         legend=False, 
                         ax=axes[i_row, i_column],
                         legend_kwds={'extend':'both'},
                         )
                axes[i_row, i_column].set_xticks([])
                axes[i_row, i_column].set_yticks([])
                cbar = plt.colorbar(sm, ax=axes[i_row, i_column], extend='both', fraction=0.035, pad=0.03)
                cbar.ax.set_ylabel(f'{variable_name}', fontsize=fontsize)
                cbar.ax.tick_params(labelsize=fontsize)

        plt.tight_layout()
        if normed:
            plt.savefig(Path("Diss Graphiken") / f"Density_map_{region}{metric}_Normed.svg")
        else:
            plt.savefig(Path("Diss Graphiken") / f"Density_map_{region}{metric}.svg")
        plt.close()
        print(f"saved plot for {metric} {region}")


def vamx_for_density_maps(scenario_dict, region: str, normed: bool, metric_dict: dict):
    if normed:
        file_path = Path(__file__).parent / f"vmax_dictionary_{region}_normed.pkl"
    else:
        file_path = Path(__file__).parent / f"vmax_dictionary_{region}.pkl"

    if not file_path.exists():
        print("creating dictionary containing vmin and vmax for plots")
        v_dict = {}
        # create dict for heating, cooling, electricity with max values
        for metric, variable_name in metric_dict.items():
            metric_list = []
            for j, (scenario_name, year_scenarios) in enumerate(scenario_dict.items()):
                scenarios = create_single_scenarios_from_year_scenarios(year_scens=year_scenarios)
                for scen in scenarios:
                    parquet_file = f"{metric}_{get_file_name(scen)}.parquet.gzip"
                    # to get a nice graph the outliers have to be ignored
                    # we take the max of the 90th percentile excluding the highest 10th percentile
                    if normed:
                        profiles = load_normed_profiles(region=region, filename=parquet_file, variable_name=variable_name)
                        metric_list.append(np.ceil(profiles[variable_name].quantile(0.9)))
                        metric_list.append(np.floor(profiles[variable_name].quantile(0.1)))
                    else:
                        profiles = load_summed_profiles(region=region, filename=parquet_file, variable_name=variable_name)
                        metric_list.append(np.ceil(profiles[variable_name].quantile(0.9) / 1_000) * 1_000)
                        metric_list.append(np.floor(profiles[variable_name].quantile(0.1) / 1_000) * 1_000)
            v_dict[metric] = (min(metric_list), max(metric_list))

        # save to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(v_dict, f)
        print(f"saved dictionary for vmin and vmax for {region}")
    with open(file_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

# murcia shapefile
path_to_gdf_files = Path(__file__).parent / "input_data"
file_murica = "H_geometry_df_Murcia.shp"
murcia_gdf = gpd.read_file(path_to_gdf_files / file_murica).rename(columns={"ID_Buildin": "ID_Building"})

file_leeuwarden = "H_geometry_df_Leeuwarden.shp"
leeuwarden_gdf = gpd.read_file(path_to_gdf_files / file_leeuwarden).rename(columns={"ID_Buildin": "ID_Building"})


normed = True
if normed:
    metric_dict = {"Heating_q": "yearly heat demand (kWh/m$^2$)", 
                "Demand": "yearly electricity demand (kWh/m$^2$)", 
                "Cooling_e": "yearly cooling demand (kWh/m$^2$)"}
else:
    metric_dict = {"Heating_q": "yearly heat demand (kWh)", 
                    "Demand": "yearly electricity demand (kWh)", 
                    "Cooling_e": "yearly cooling demand (kWh)"}
    
v_dict = vamx_for_density_maps(scenario_dict=scenario_dict_leeuwarden, region="Leeuwarden", normed=normed, metric_dict=metric_dict)
create_big_subplot_with_density_maps(region="Leeuwarden", region_gdf=leeuwarden_gdf, scenario_dict=scenario_dict_leeuwarden, v_dict=v_dict, metric_dict=metric_dict)

v_dict = vamx_for_density_maps(scenario_dict=scenario_dict_murcia, region="Murcia", normed=normed, metric_dict=metric_dict)
create_big_subplot_with_density_maps(region="Murcia", region_gdf=murcia_gdf, scenario_dict=scenario_dict_murcia, v_dict=v_dict, metric_dict=metric_dict)


