import pandas as pd
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
import sqlalchemy
from pathlib import Path
from typing import List
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


BUILDING_ID_2_NAME = {
    4: "EZFH 1 B" ,
    5: "EZFH 1 S" ,
    1: "EZFH 5 B" ,
    2: "EZFH 5 S" ,
    3: "EZFH 9 B" ,
    8: "MFH 1 B",
    9: "MFH 1 S",
    6: "MFH 5 B",
    7: "MFH 5 S",
}

def read_parquet(
    table_name: str, scenario_ID: int, folder: Path, column_names: List[str] = None
) -> pd.DataFrame:
    """
    Returns: dataframe containing the results for the table name and specific scenario ID
    """
    file_name = f"{table_name}_S{scenario_ID}.parquet.gzip"
    path_to_file = folder / file_name
    if column_names:
        df = pd.read_parquet(path=path_to_file, engine="auto", columns=column_names)
    else:
        df = pd.read_parquet(path=path_to_file, engine="auto")
    return df


def import_FLEX_hourly_reference_results():
    ref_dfs = []
    for i in range(10, 19):
        ref_dfs.append(
            read_parquet(
                table_name=f"OperationResult_RefHour",
                scenario_ID=i,
                folder=Path(
                    r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects\Test_bed\output"
                ),
            )
        )

    df = pd.concat(ref_dfs)
    return df


# load the hourly reference results once:
DF_HOURLY = import_FLEX_hourly_reference_results()


def import_heat_demand_from_Flex_testing():
    heat_demand = DF_HOURLY.loc[:, ["ID_Scenario", "Q_RoomHeating"]]
    return heat_demand.pivot(columns="ID_Scenario")


def import_electricity_demand_from_Flex_testing():
    elec_demand = DF_HOURLY.loc[:, ["ID_Scenario", "BaseLoadProfile"]]
    return elec_demand.pivot(columns="ID_Scenario")


def import_heat_pump_COP_from_Flex_testing():
    cop = DF_HOURLY.loc[DF_HOURLY.loc[:, "ID_Scenario"] == 10, "SpaceHeatingHourlyCOP"]
    return np.array(cop)


def import_price_profile():
    elec_price = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects\Test_bed\input") / "OperationScenario_EnergyPrice.xlsx", engine="openpyxl")[["electricity_1", "electricity_2", "electricity_3"]] + 0.020  # 20 cent grid tarif
    return elec_price


@dataclass
class ThermalStorage:
    size: float
    min_temperature: float
    max_temperature: float
    heat_loss: float = 0.2
    initial_SOC: float = 0
    sorounding_temperature: float = 20
    CPWater: float = 4200 / 3600


@dataclass
class BatteryStorage:
    max_charging_rate: float
    max_discharging_rate: float
    capacity: float
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95


@dataclass
class FlexInputs:
    electricity_price: pd.DataFrame
    heat_demand: pd.DataFrame
    cop: np.array
    electricity_demand: pd.DataFrame


def create_battery_optimisation(battery: BatteryStorage, flex_data: FlexInputs):
    nr_buildings = 9
    nr_prices = len(flex_data.electricity_price.columns)
    results = []
    for price_profile in flex_data.electricity_price.columns:
        if "1" in price_profile:
            year = 2019
        elif "2" in price_profile:
            year = 2021
        elif "3" in price_profile:
            year = 2022
        # parameters:
        heat_demand = cp.Parameter(
            shape=(8760, nr_buildings),
            value=flex_data.heat_demand.values,
            name="heat_demand",
        )
        electricity_demand = cp.Parameter(
            shape=(8760, nr_buildings),
            value=flex_data.electricity_demand.values,
            name="electricity_demand",
        )

        heat_pump_COP = cp.Parameter(
            shape=(8760,), value=flex_data.cop, name="heat_pump_COP"
        )
        electricity_price = cp.Parameter(
            shape=(8760,), value=flex_data.electricity_price[price_profile].values, name="electricity_price"
        )

        # variables:
        battery_soc = cp.Variable(
            shape=(8760, nr_buildings), name="battery_soc", nonneg=True
        )
        battery_charge_power = cp.Variable(shape=(8760, nr_buildings), name="battery_charge_power", nonneg=True)
        battery_discharge_power = cp.Variable(shape=(8760, nr_buildings), name="battery_discharge_power", nonneg=True)

        grid_power = cp.Variable(shape=(8760, nr_buildings), name="grid_power", nonneg=True)
        heat_pump_electricity = cp.Variable(
            shape=(8760, nr_buildings), name="heat_pump_electricity", nonneg=True
        )

        constraints = []
        # constraining battery SOC
        constraints.append(battery_soc[:, :] <= battery.capacity)  # max capacity
        constraints.append(battery_soc[0, :] == 0)
        constraints.append(
            battery_soc[1:, :] == battery_soc[:-1, :] + battery_charge_power[1:, :] * battery.charge_efficiency - battery_discharge_power[1:, :] / battery.discharge_efficiency
        )

        # contraining battery power
        constraints.append(battery_charge_power[:, :] <= battery.max_charging_rate)
        constraints.append(battery_discharge_power[:, :] <= battery.max_discharging_rate)

        # electricity demand has to be satisfied:
        constraints.append(
            electricity_demand + heat_pump_electricity + battery_charge_power == grid_power + battery_discharge_power
        )

        # heat demand has to be satisfied:
        constraints.append(
            heat_demand == cp.multiply(cp.reshape(heat_pump_COP, (8760, 1)), heat_pump_electricity)
        )

        objective = cp.Minimize(
            cp.sum(cp.multiply(cp.reshape(electricity_price, (8760, 1)), grid_power))
        )

        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()
        if problem.status == "optimal":
            print("solved")
            plot_dual_variables(constraints=constraints)
            result = create_result_df(
                battery_soc=battery_soc.value,
                battery_power=battery_charge_power.value - battery_discharge_power.value,
                grid_power=grid_power.value,
                heat_pump_electricity=heat_pump_electricity.value,
                price=flex_data.electricity_price[price_profile],
            )
            result["year"] = year
            results.append(result)

        else:
            print("not solved!")

    return pd.concat(results)


def plot_dual_variables(constraints: list):
    df = pd.concat([pd.Series(l) for l in constraints[-2].dual_value], axis=1).T
    df.columns = [f"{BUILDING_ID_2_NAME[i]}" for i in range (1, 10)]
    
    fig = px.line(
        data_frame=df
    )
    fig.show()
    fig = plt.figure(figsize=(18, 15))
    ax = plt.gca()
    ax.plot(constraints[-2].dual_value)

    plt.show()


def create_result_df(
    battery_soc,
    battery_power,
    grid_power,
    heat_pump_electricity,
    price,
):
    dfs = [
        pd.DataFrame(battery_soc / 1000, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9]).melt(
            var_name="Building ID", value_name="SOC Battery (kWh)"
        ),
        pd.DataFrame(battery_power / 1000, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9]).melt(
            var_name="Building ID", value_name="Battery power (kW)"
        ),
        pd.DataFrame(grid_power / 1000, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9]).melt(
            var_name="Building ID", value_name="Grid power (kW)"
        ),
        pd.DataFrame(
            heat_pump_electricity / 1000, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9]
        ).melt(var_name="Building ID", value_name="heat pump power (kW)"),
    ]

    df = pd.concat(dfs, axis=1)
    df.loc[:, "hour"] = np.tile(np.arange(8760), 9)
    df.loc[:, "electricity price (cent/kWh)"] = np.tile(price * 1_000, 9)
    df = df.loc[:, ~df.columns.duplicated()]  # .melt(id_vars=["Building ID", "hour"])
    return df


def plot_optimization_results(df):
    fig = px.line(
        data_frame=df.loc[:, ["hour", "Battery power (kW)", "Building ID", "year"]],
        x="hour",
        y="Battery power (kW)",
        color="year",
        facet_row="Building ID",
    )

    fig.add_traces(
        list(
            px.bar(
                data_frame=df.loc[:, ["hour", "SOC Battery (kWh)", "Building ID", "year"]],
                x="hour",
                y="SOC Battery (kWh)",
                facet_row="Building ID",
                color="year",
            ).select_traces()
        )
    )

    fig.show()


def find_charging_price_quantiles(df):
    quantiles_weighted = {}
    quantiles_unweighted = {}
    for (scen_id, year), group in df.groupby(["Building ID", "year"]):
        pre_heat_indices = np.where(group["Battery power (kW)"] > 0)[0]
        preheating_energy_cut = (
            group["Battery power (kW)"].iloc[pre_heat_indices].copy()
        )
        electricity_pre_heat_price = (
            group["electricity price (cent/kWh)"].iloc[pre_heat_indices]
            * preheating_energy_cut
        ).sum() / preheating_energy_cut.sum()
        electricity_pre_heat_price_unweighted = (
            group["electricity price (cent/kWh)"].iloc[pre_heat_indices].mean()
        )
        sorted_price = sorted(group["electricity price (cent/kWh)"])

        # Step 2: Calculate the quantile
        # Use numpy to find the quantile
        # This function finds the relative position of the value in the sorted data
        def find_quantile(value, sorted_data):
            pos = np.searchsorted(sorted_data, value, side="right")
            quantile = pos / len(sorted_data)
            return quantile

        # Calculate the quantile for the given value
        quantile = find_quantile(electricity_pre_heat_price_unweighted, sorted_price)
        quantile_weighted = find_quantile(electricity_pre_heat_price, sorted_price)

        quantiles_weighted[(scen_id, year)] = quantile_weighted
        quantiles_unweighted[(scen_id, year)] = quantile

    return quantiles_unweighted, quantiles_weighted

def plot_price_quantiles_for_charging(weighted_q, unweighted_q, storage_type: str):
    df = pd.DataFrame(unweighted_q, index=["percentile"]).T.reset_index().rename(columns={"level_0": "ID_Building", "level_1":"year"})  #.melt(id_vars=["ID_Building", "year"], value_name="price quantile for charging", var_name="method")
    # df[["ID_Building","year"]] = pd.DataFrame(df["ID_Building"].to_list(), columns=["ID_Building","year"])
    df["ID_Building"] = df["ID_Building"].map(BUILDING_ID_2_NAME)
    df.sort_values(by="ID_Building", inplace=True)

    matplotlib.rc("font", **{"size": 22})
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(
        data=df,
        x="ID_Building",
        y="percentile",
        hue="year",
        
    )
    plt.ylabel(r"price quantile ($\alpha$) " + f"at which {storage_type} is charged")
    plt.xticks(rotation=0)
    ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Battery_charging_quantiles.svg")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Battery_charging_quantiles.png")


    plt.show()


if __name__ == "__main__":
    battery = BatteryStorage(
        max_charging_rate=4500,
        max_discharging_rate=4500,
        capacity=7000,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )

    results = create_battery_optimisation(
        battery=battery,
        flex_data=FlexInputs(
            electricity_price=import_price_profile(),
            heat_demand=import_heat_demand_from_Flex_testing(),
            cop=import_heat_pump_COP_from_Flex_testing(),
            electricity_demand=import_electricity_demand_from_Flex_testing(),
        ),
    )

    unweighted, weighted = find_charging_price_quantiles(results)
    plot_price_quantiles_for_charging(weighted_q=weighted, unweighted_q=unweighted, storage_type="Battery")


