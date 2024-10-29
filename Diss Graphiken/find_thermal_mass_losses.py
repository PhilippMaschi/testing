import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from matplotlib import cm
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import PercentFormatter
import tqdm

matplotlib.rc("font", **{"size": 22})


building_id_to_name = {
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

def ref_HeatingCooling(T_outside, Q_solar, Buildings, initial_thermal_mass_temp=20, T_air_min=20, T_air_max=27,
                       ):
    """
    This function calculates the heating and cooling demand as well as the indoor temperature for every building
    category based in the 5R1C model. The results are hourls vectors for one year. Q_solar is imported from a CSV
    at the time! Inputs for T_outside and Q_solar have to be numpy arrays, otherwise the temperature and radiation
    is taken from the database.
    """

    InternalGains = Buildings['spec_int_gains_cool_watt'].to_numpy()
    Hop = Buildings['Hop'].to_numpy()
    Htr_w = Buildings['Htr_w'].to_numpy()
    Hve = Buildings['Hve'].to_numpy()
    CM_factor = Buildings['CM_factor'].to_numpy()
    Am_factor = Buildings['Am_factor'].to_numpy()
    Af = Buildings["Af"].to_numpy()

    T_air_min = np.full((len(T_outside),), T_air_min)
    T_air_max = np.full((len(T_outside),), T_air_max)

    # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
    Atot = 4.5 * Af
    # Speicherkapazität J/K
    Cm = CM_factor * Af
    # wirksame Massenbezogene Fläche [m^2]
    Am = Am_factor * Af
    # internal gains
    Q_InternalGains = InternalGains * Af
    timesteps = np.arange(len(T_outside))

    # Kopplung Temp Luft mit Temp Surface Knoten s
    his = np.float_(3.45)  # 7.2.2.2
    # kopplung zwischen Masse und  zentralen Knoten s (surface)
    hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
    Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
    Htr_em = 1 / (1 / Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
    # thermischer Kopplungswerte W/K
    Htr_is = his * Atot
    Htr_1 = np.float_(1) / (np.float_(1) / Hve + np.float_(1) / Htr_is)  # Equ. C.6
    Htr_2 = Htr_1 + Htr_w  # Equ. C.7
    Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8

    # Equ. C.1
    PHI_ia = 0.5 * Q_InternalGains

    Tm_t = np.zeros(shape=(len(timesteps), len(Hve)))
    T_sup = np.zeros(shape=(len(timesteps),))
    Q_Heating_noDR = np.zeros(shape=(len(timesteps), len(Hve)))
    Q_Cooling_noDR = np.zeros(shape=(len(timesteps), len(Hve)))
    T_Room_noDR = np.zeros(shape=(len(timesteps), len(Hve)))
    operative_temperature = np.zeros(shape=(len(timesteps), len(Hve)))
    heating_power_10 = Af * 10

    for t in timesteps:  # t is the index for each timestep
        # Equ. C.2
        PHI_m = Am / Atot * (0.5 * Q_InternalGains + Q_solar[t, :])
        # Equ. C.3
        PHI_st = (1 - Am / Atot - Htr_w / 9.1 / Atot) * \
                 (0.5 * Q_InternalGains + Q_solar[t, :])

        # (T_sup = T_outside weil die Zuluft nicht vorgewärmt oder vorgekühlt wird)
        T_sup[t] = T_outside[t]

        # Equ. C.5
        PHI_mtot_0 = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                PHI_st + Htr_w * T_outside[t] + Htr_1 * (((PHI_ia + 0) / Hve) + T_sup[t])) / \
                     Htr_2

        # Equ. C.5 with 10 W/m^2 heating power
        PHI_mtot_10 = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                PHI_st + Htr_w * T_outside[t] + Htr_1 * (
                ((PHI_ia + heating_power_10) / Hve) + T_sup[t])) / Htr_2

        # Equ. C.5 with 10 W/m^2 cooling power
        PHI_mtot_10_c = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                PHI_st + Htr_w * T_outside[t] + Htr_1 * (
                ((PHI_ia - heating_power_10) / Hve) + T_sup[t])) / Htr_2

        if t == 0:
            if type(initial_thermal_mass_temp) == int or type(initial_thermal_mass_temp) == float:
                Tm_t_prev = np.array([initial_thermal_mass_temp] * len(Hve))
            else:  # initial temperature is already a vector
                Tm_t_prev = initial_thermal_mass_temp
        else:
            Tm_t_prev = Tm_t[t - 1, :]

        # Equ. C.4
        Tm_t_0 = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_0) / \
                 (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

        # Equ. C.4 for 10 W/m^2 heating
        Tm_t_10 = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_10) / \
                  (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

        # Equ. C.4 for 10 W/m^2 cooling
        Tm_t_10_c = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_10_c) / \
                    (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

        # Equ. C.9
        T_m_0 = (Tm_t_0 + Tm_t_prev) / 2

        # Equ. C.9 for 10 W/m^2 heating
        T_m_10 = (Tm_t_10 + Tm_t_prev) / 2

        # Equ. C.9 for 10 W/m^2 cooling
        T_m_10_c = (Tm_t_10_c + Tm_t_prev) / 2

        # Euq. C.10
        T_s_0 = (Htr_ms * T_m_0 + PHI_st + Htr_w * T_outside[t] + Htr_1 *
                 (T_sup[t] + (PHI_ia + 0) / Hve)) / (Htr_ms + Htr_w + Htr_1)

        # Euq. C.10 for 10 W/m^2 heating
        T_s_10 = (Htr_ms * T_m_10 + PHI_st + Htr_w * T_outside[t] + Htr_1 *
                  (T_sup[t] + (PHI_ia + heating_power_10) / Hve)) / (Htr_ms + Htr_w + Htr_1)

        # Euq. C.10 for 10 W/m^2 cooling
        T_s_10_c = (Htr_ms * T_m_10_c + PHI_st + Htr_w * T_outside[t] + Htr_1 *
                    (T_sup[t] + (PHI_ia - heating_power_10) / Hve)) / (Htr_ms + Htr_w + Htr_1)

        # Equ. C.11
        T_air_0 = (Htr_is * T_s_0 + Hve * T_sup[t] + PHI_ia + 0) / \
                  (Htr_is + Hve)

        # Equ. C.11 for 10 W/m^2 heating
        T_air_10 = (Htr_is * T_s_10 + Hve * T_sup[t] + PHI_ia + heating_power_10) / \
                   (Htr_is + Hve)

        # Equ. C.11 for 10 W/m^2 cooling
        T_air_10_c = (Htr_is * T_s_10_c + Hve * T_sup[t] + PHI_ia - heating_power_10) / \
                     (Htr_is + Hve)
        operative_temperature_0 = 0.3 * T_air_0 + 0.7 * T_s_0
        operative_temperature_10 = 0.3 * T_air_10 + 0.7 * T_s_10
        operative_temperature_10_c = 0.3 * T_air_10_c + 0.7 * T_s_10_c
        op_temp_min = 0.3 * T_air_min[t] + 0.7 * T_s_0
        op_temp_max = 0.3 * T_air_max[t] + 0.7 * T_s_0

        for i in range(len(Hve)):
            # Check if air temperature without heating is in between boundaries and calculate actual HC power:
            if operative_temperature_0[i] >= op_temp_min[i] and operative_temperature_0[i] <= op_temp_max[i]:
                Q_Heating_noDR[t, i] = 0
            elif operative_temperature_0[i] < op_temp_min[i]:  # heating is required
                Q_Heating_noDR[t, i] = heating_power_10[i] * (T_air_min[t] - operative_temperature_0[i]) / (operative_temperature_10[i] - operative_temperature_0[i])
            elif operative_temperature_0[i] > op_temp_max:  # cooling is required
                Q_Cooling_noDR[t, i] = heating_power_10[i] * (T_air_max[t] - operative_temperature_0[i]) / (operative_temperature_10_c[i] - operative_temperature_0[i])

        # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
        # Equ. C.5 with actual heating power
        PHI_mtot_real = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                PHI_st + Htr_w * T_outside[t] + Htr_1 * (
                ((PHI_ia + Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / Hve) + T_sup[t])) / Htr_2
        # Equ. C.4
        Tm_t[t, :] = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_real) / \
                     (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

        # Equ. C.9
        T_m_real = (Tm_t[t, :] + Tm_t_prev) / 2

        # Euq. C.10
        T_s_real = (Htr_ms * T_m_real + PHI_st + Htr_w * T_outside[t] + Htr_1 *
                    (T_sup[t] + (PHI_ia + Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / Hve)) / \
                   (Htr_ms + Htr_w + Htr_1)

        # Equ. C.11 for 10 W/m^2 heating
        T_Room_noDR[t, :] = (Htr_is * T_s_real + Hve * T_sup[t] + PHI_ia +
                             Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / (Htr_is + Hve)

        operative_temperature[t, :] = 0.3*T_Room_noDR[t, :] + 0.7*T_s_real

    # fill nan
    Q_Cooling_noDR = np.nan_to_num(Q_Cooling_noDR, nan=0)
    Q_Heating_noDR = np.nan_to_num(Q_Heating_noDR, nan=0)
    T_Room_noDR = np.nan_to_num(T_Room_noDR, nan=0)
    Tm_t = np.nan_to_num(Tm_t, nan=0)
    operative_temperature = np.nan_to_num(operative_temperature, nan=0)

    return Q_Heating_noDR, Q_Cooling_noDR, operative_temperature, Tm_t


def get_constant_Q_Tm_t(Buildings, T_outside, T_min_indoor, T_max_indoor, initial_thermal_mass_temp, runtime: int):
    # start time steps to calculate constant values
    # starte mit 100 stunde, dann checken ob sich konstantes Tm_t eingestellt hat
    Temperature_outside = np.array([T_outside] * runtime)
    Q_sol = np.zeros((runtime, Buildings.shape[0]))
    # initial thermal mass temperature should be between T_min_indoor and T_max_indoor (22°C should always be ok)
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = ref_HeatingCooling(T_outside=Temperature_outside,
                                                                                         Q_solar=Q_sol,
                                                                                         initial_thermal_mass_temp=initial_thermal_mass_temp,
                                                                                         T_air_min=T_min_indoor,
                                                                                         T_air_max=T_max_indoor,
                                                                                         Buildings=Buildings)
    # check if stationary condition has set in
    prev_difference = 0
    while (-1e-4 >= T_thermalMass_noDR[-1, :] - T_thermalMass_noDR[-2, :]).any() or \
            (1e-4 <= T_thermalMass_noDR[-1, :] - T_thermalMass_noDR[-2, :]).any():
        new_difference = np.round(T_thermalMass_noDR[-1, :] - T_thermalMass_noDR[-2, :], 6)
        if (prev_difference == new_difference).any():
            # oscilating -> break the loop
            break
        else:
            Temperature_outside = np.array([T_outside] * runtime)
            Q_sol = np.zeros((runtime, Buildings.shape[0]))
            Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = ref_HeatingCooling(
                T_outside=Temperature_outside,
                Q_solar=Q_sol,
                initial_thermal_mass_temp=T_thermalMass_noDR[-1, :],
                T_air_min=T_min_indoor,
                T_air_max=T_max_indoor,
                Buildings=Buildings)
            prev_difference = new_difference

    return Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR


def calculate_LoadShiftPotential(Buildings: pd.DataFrame, 
                                 hours_of_preheating: int, 
                                 T_outside: float,
                                 T_min_indoor: float, 
                                 T_max_indoor: float, 
                                #  thermal_mass_starting_temp: float,
                                 T_offset_indoor: float=2, 
                                 ):
    # calculate the thermal mass temperature when there is thermal equilibrium:
    Q_heating_constant, Q_cooling_constant, T_room_constant, T_thermalMass_constant = \
        get_constant_Q_Tm_t(Buildings, T_outside, T_min_indoor, T_max_indoor, initial_thermal_mass_temp=18, runtime=500)
    # last value is constant value:
    Q_heating_constant = Q_heating_constant[-1, :]
    Q_cooling_constant = Q_cooling_constant[-1, :]
    T_room_constant = T_room_constant[-1, :]
    T_thermalMass_constant = T_thermalMass_constant[-1, :]

    # create temperature and solar gains array
    Temperature_outside = np.array([T_outside] * hours_of_preheating)
    # no solar gains are considered
    Q_sol = np.zeros((hours_of_preheating, Buildings.shape[0]))
    # calculate the thermal mass temperature after the time of preheating as well as heating/cooling power
    # this is done by raising the minimum indoor temperature for heating and lowering it for cooling by 2°C
    Q_preheating, Q_PreCooling, T_room_preheating, T_thermal_mass_preheating = ref_HeatingCooling(
        T_outside=Temperature_outside,
        Q_solar=Q_sol,
        initial_thermal_mass_temp=T_thermalMass_constant,
        T_air_min=T_min_indoor + T_offset_indoor,
        T_air_max=T_max_indoor - T_offset_indoor,
        Buildings=Buildings,)
    Q_PreHeating_total = Q_preheating.sum(axis=0)

    # calculate the energy until static temperature is reached again:
    Q_heating_shifting, Q_cooling_shifting, T_room_shifting, T_thermal_ass_shifting = \
        get_constant_Q_Tm_t(Buildings, T_outside, T_min_indoor, T_max_indoor, T_thermal_mass_preheating[-1, :], runtime=500)
    # total heating until steady state is reached after shifting:
    Q_Heating_afterShift_sum = Q_heating_shifting.sum(axis=0)
    # Energy still stored in thermal mass after unloading
    RemainingEnergy = (np.tile(Q_heating_constant, (Q_heating_shifting.shape[0], 1)) - Q_heating_shifting).sum(axis=0)
    
    # reference energy into building mass while pre-heating:
    Q_Heating_constant_total_preheating = Q_heating_constant * hours_of_preheating
    ExcessHeatPreheat = Q_PreHeating_total - Q_Heating_constant_total_preheating
    # total losses:
    TotalLoss = ExcessHeatPreheat - RemainingEnergy
    percentage_loss = TotalLoss/ExcessHeatPreheat
    T_delta_thermal_mass = T_thermal_mass_preheating[-1, :] - T_thermalMass_constant
    # print(f"loss in %: {percentage_loss}")

    return percentage_loss, T_delta_thermal_mass, RemainingEnergy




def plot_heat_demand_and_shifted_bars(
        ExcessHeatPreheat,
        RemainingEnergy,
        TotalLoss,
        Q_PreHeating,
        Q_shifting,
        Q_constant,
        T_thermal_mass_pre_heating,
        T_thermal_mass_shifting,
        T_thermal_mass_constant,
        T_room_pre_heating,
        T_room_shifting,
        T_room_constant,
        house_nr,
        preheating_hours,
        shifting_hours,
):
    # plots:
    fig0 = plt.figure(figsize=(5, 6))
    bar_width = 0.3
    ax0 = plt.gca()
    # Bar positions
    bar_positions = [0.5, 1]
    plt.bar(bar_positions[0], ExcessHeatPreheat[house_nr-1], color="red", label="Additional Energy", width=bar_width, alpha=0.7)
    plt.bar(bar_positions[1], SaveHeatShifting[house_nr-1], color="green", label="Reduced Energy", width=bar_width, alpha=0.7)
    plt.bar(bar_positions[1], RemainingEnergy[house_nr-1], color="orange", bottom=SaveHeatShifting[house_nr-1],
            label="Energy remaining in thermal mass", width=bar_width)
    plt.bar(bar_positions[1], TotalLoss[house_nr-1], color="grey",
            bottom=SaveHeatShifting[house_nr-1] + RemainingEnergy[house_nr-1], label="Thermal losses", width=bar_width)

    plt.title("Energy shifting at " + str(T_outside) + "°C, house Nr " + str(house_nr))
    ax0.set_xticks(bar_positions)
    ax0.set_xticklabels(['preheating', 'discharging'])
    plt.ylabel("Energy (kWh)")

    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.resolve() / f"Energy_shifted_{house_nr}.png")
    plt.savefig(Path(__file__).parent.resolve() / f"Energy_shifted_{house_nr}.svg")

    # plt.show()

    # plot results for one building:
    x_achse = np.arange(preheating_hours + shifting_hours)
    Q_Heating_plot = np.append(Q_PreHeating[:, house_nr - 1], Q_ReducedHeating_noDR[:, house_nr - 1])
    T_thermalMass_plot = np.append(T_thermal_mass_pre_heating[:, house_nr - 1], T_ReducedthermalMass_noDR[:, house_nr - 1])
    T_Room_plot = np.append(T_room_pre_heating[:, house_nr - 1], T_ReducedRoom_noDR[:, house_nr - 1])

    fig1, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x_achse, Q_Heating_plot, label="heating power", color="red")
    ax1.axhline(Q_constant[house_nr - 1], xmin=0, xmax=1, label="constant heating power", color="black")
    # ax1.vlines(x=0, ymin=Q_Heating_noDR_constant[0], ymax=Q_Heating_plot[0], color="red")
    ax1.axvline(x=preheating_hours - 1, color="black", linestyle="--", linewidth=0.7)

    ax1.fill_between(x_achse, Q_constant[house_nr - 1], Q_Heating_plot,
                     where=(Q_Heating_plot > Q_constant[house_nr - 1]), color='red', alpha=0.7,
                     label='Additional Energy', interpolate=True)
    ax1.fill_between(x_achse, Q_constant[house_nr - 1], Q_Heating_plot,
                     where=(Q_Heating_plot < Q_constant[house_nr - 1]), color='green', alpha=0.7,
                     label='Reduced Energy', interpolate=True)

    mass_temp = ax2.plot(x_achse, T_thermalMass_plot, label="Thermal mass temperature", color="purple")
    const_mass_temp = ax2.hlines(T_thermalMass_noDR_constant[house_nr - 1],
                                 xmin=0,
                                 xmax=preheating_hours + shifting_hours - 1,
                                 color="purple",
                                 label="Constant thermal mass temperature",
                                 linestyles="--")
    ax2.vlines(x=0, ymin=T_thermalMass_noDR_constant[0], ymax=T_thermalMass_plot[0], color="purple")
    ax2.axvline(x=preheating_hours - 1, color="black", linestyle="--", linewidth=0.5)

    room_temp = ax2.plot(x_achse, T_Room_plot, label="Room temperature", color="skyblue")
    const_room_temp = ax2.hlines(T_Room_noDR_constant[0],
                                 xmin=0,
                                 xmax=preheating_hours + shifting_hours - 1,
                                 color="skyblue",
                                 label="Constant room temperature",
                                 linestyles="--")
    ax2.vlines(x=0, ymin=T_Room_noDR_constant[house_nr - 1], ymax=T_Room_plot[0], color="skyblue")
    ax2.axvline(x=preheating_hours - 1, color="black", linestyle="--", linewidth=0.5)

    ax1.legend(loc="upper right")
    labels = ["Room temperature", "Thermal mass temperature", 'Variable', 'Constant']

    ax2.legend([
        Line2D([0], [0], color='skyblue', linestyle='-'),
        Line2D([0], [0], color='purple', linestyle='-'),

        Line2D([0], [0], color='black', linestyle='-'),
        Line2D([0], [0], color='black', linestyle='--')],
        labels,
        loc='upper right'
    )

    ax1.set_ylabel("heating power in kW")
    ax2.set_ylabel("temperature in °C")
    ax1.set_title("Load shift at " + str(T_outside) + " °C, House Nr " + str(house_nr))
    ax2.set_xlabel("time (h)")
    plt.tight_layout()
    fig1.savefig(Path(__file__).parent.resolve() / f"Subplot_energy_shifted_{house_nr}.png")
    fig1.savefig(Path(__file__).parent.resolve() / f"Subplot_energy_shifted_{house_nr}.svg")
    # plt.show()
    return fig0, fig1


def pearson_correlation_loss(df):
    # correlation between loss and other factors for each building:
    corr_df = df.copy().drop(columns=["shifted energy"])
    corr_df = corr_df.groupby(by="Building ID").corr()["loss"].reset_index().rename(columns={"level_1": "setting", "loss": "pearson correlation with loss"})
    corr_df["setting"] = corr_df["setting"].map({
        "offset_temp": "preheating \n temperature",
        "t_delta_thermal_mass": r"$\Delta$T thermal mass",
        "outside temperature": "outside \n temperature",
        "loss" : r"$\xi_{building}$",
        "shifted energy": "shifted \n energy",
        "preheating time": "preheating \n time"
    })
    corr_df = corr_df.loc[corr_df["setting"]!=r"$\xi_{building}$",:]
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(
        corr_df,
        x="setting",
        y="pearson correlation with loss",
        hue="Building ID",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="lower right", fontsize=17)
    plt.xticks(rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel(r"pearson correlation with $\xi_{building}$")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"pearson_correlation_building_loss.png")
    plt.savefig(Path(__file__).parent / f"pearson_correlation_building_loss.svg")

def remove_no_heating_loss_values(df):
    # when a building has a strong increase in loss with rising outside temperature it means that the building does not need to be preheated as it almost needs no heating anymore.
    groups = []
    for building_id, group in df.groupby("Building ID"):
        profile = group.loc[:, "loss"].copy()
        Q1 = np.percentile(profile, 25)
        Q3 = np.percentile(profile, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.2 * IQR
        upper_bound = Q3 + 1.2 * IQR
        reduced_df = group.loc[group.loc[:, "loss"] < upper_bound, :].copy()
        groups.append(reduced_df)
    new_df = pd.concat(groups, axis=0)

def create_file_with_xi_values_over_temperature(df):
    df.groupby(["Building ID", "outside temperature"]).max().reset_index().loc[:, ["Building ID", "outside temperature", "loss"]].to_csv(
        Path(__file__).parent / "thermal_mass_loss_based_on_outside_temperature.csv", sep=";", index=False
    )



def plot_losses_against_delta_t(results_df: pd.DataFrame):
    # due to flaoting point errors, losses with lower than 0.1 delta T are excluded:
    results_df["Building ID"] = results_df["Building ID"].map(building_id_to_name)
    
    create_file_with_xi_values_over_temperature(results_df)
    line_plot_loss_vs_outside_temp(results_df)
    pearson_correlation_loss(results_df)
    plot_mean_loss_per_building(results_df)
    line_plot_loss_vs_t_delta_thermal_mass(results_df)

    # remove_no_heating_loss_values(results_df)  # no need because i need the higher loss values
    
    matplotlib.rc("font", **{"size": 22})
    palette = sns.color_palette("tab10", len(results_df['outside temperature'].unique()))
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.scatterplot(
        data=results_df,
        x="outside temperature",
        y="loss",
        hue="t_delta_thermal_mass",
        style="Building ID",
        # palette=palette
    )
    ax.legend(loc="upper right", fontsize=18)

    # for i, b_id in enumerate(results_df["Building ID"].unique()):
    #     sns.regplot(
    #         data=results_df.loc[(results_df["Building ID"] == b_id), :],
    #         x="outside temperature",
    #         y="loss",
    #         ci=100,
    #         scatter=False,
    #         line_kws=dict(color="grey"),
    #         order=1
    #     )
    ax.set_xlabel(r"$\Delta T$ thermal mass (°C)")
    ax.set_ylabel(r"thermal loss $\xi_{building}$ (%)")
    ax.set_ylim(0.02, 0.15)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"Regression_of_the_thermal_mass_losses.png")
    plt.savefig(Path(__file__).parent / f"Regression_of_the_thermal_mass_losses.svg")
    plt.show()



    # Pearson correlation:
    losses_per_meter = pd.read_csv(Path(__file__).parent / "heat_demand_per_square_meter.csv", sep=";").rename(columns={"ID_Building": "Building ID"})
    losses_per_meter["Building ID"] = losses_per_meter["Building ID"].map(building_id_to_name)
    merged = pd.merge(left=mean_loss, right=losses_per_meter, on="Building ID")
    merged.loc[: , ["heat demand (kWh/m2)", "loss", "Af", "Hop", "Htr_w", "Hve", "CM_factor"]].corr(method="pearson")["loss"].plot(kind="bar")
    plt.show()


def plot_mean_loss_per_building(df):
    # plot the average loss value of every building:
    mean_loss = df.groupby(by=["Building ID"])["loss"].mean()
    fig, ax = plt.subplots(figsize=(20, 12))
    mean_loss.plot(kind="bar", color="grey")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel(r"mean shifting loss factor $\xi_{building}$ (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"mean_shifting_loss_factor_per_building.png")
    plt.savefig(Path(__file__).parent / f"mean_shifting_loss_factor_per_building.svg")
    plt.show()


def line_plot_loss_vs_outside_temp(df):
    palette = sns.color_palette("tab10", len(df['Building ID'].unique()))
    df.sort_values("Building ID", inplace=True)
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.lineplot(
        data=df,
        x="outside temperature",
        y="loss",
        hue="Building ID",
        palette=palette,
    )
    plt.xticks(rotation=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel(r"thermal loss $\xi_{building}$ (%)")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"line_plot_xi_loss_vs_outside_temperature.png")
    plt.savefig(Path(__file__).parent / f"line_plot_xi_loss_vs_outside_temperature.svg")
    plt.show()

def line_plot_loss_vs_t_delta_thermal_mass(df):
    palette = sns.color_palette("tab10", len(df['Building ID'].unique()))
    df.sort_values("Building ID", inplace=True)
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.lineplot(
        data=df,
        x="t_delta_thermal_mass",
        y="loss",
        hue="Building ID",
        palette=palette,
    )
    plt.xticks(rotation=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel(r"thermal loss $\xi_{building}$ (%)")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"line_plot_xi_loss_vs_t_delta_thermal_mass.png")
    plt.savefig(Path(__file__).parent / f"line_plot_xi_loss_vs_t_delta_thermal_mass.svg")
    plt.show()

def process_combinations(time_pre_heating, offset_temp, outside_temp, Buildings):
    T_min_indoor = 20
    T_max_indoor = 27
    loss, t_delta_thermal_mass, shifted_energy = calculate_LoadShiftPotential(
        Buildings=Buildings,
        hours_of_preheating=time_pre_heating,
        T_outside=outside_temp,
        T_min_indoor=T_min_indoor, 
        T_max_indoor=T_max_indoor, 
        T_offset_indoor=offset_temp,
    )

    result = []
    for building_id in range(len(loss)):
        result.append({
            'preheating time': time_pre_heating,
            'offset_temp': offset_temp,
            'outside temperature': outside_temp,
            'Building ID': building_id + 1,
            'loss': loss[building_id],
            't_delta_thermal_mass': t_delta_thermal_mass[building_id],
            'shifted energy': shifted_energy[building_id],
        })
    return result

if __name__ == "__main__":
    # path:
    project_directory_path = Path(__file__).parent.resolve()
    # define building data
    Buildings = pd.read_excel(project_directory_path / "thermal_mass_losses_buildings.xlsx", engine="openpyxl")

    combinations = [
        (time_pre_heating, offset_temp, outside_temp)
        for time_pre_heating in np.arange(1, 5)
        for offset_temp in np.arange(1, 3.25, 0.25)
        for outside_temp in range(-17, 15, 1)
    ]

    parallel = Parallel(n_jobs=-1)  # Use all available cores
    results = parallel(delayed(process_combinations)(time_pre_heating, offset_temp, outside_temp, Buildings) for time_pre_heating, offset_temp, outside_temp in tqdm.tqdm(combinations, desc="Processing combinations"))
    flattened_results = [item for sublist in results for item in sublist]

    df_results = pd.DataFrame(flattened_results)
    
    plot_losses_against_delta_t(df_results)





