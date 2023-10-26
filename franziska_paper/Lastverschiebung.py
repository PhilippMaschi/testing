import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

        for i in range(len(Hve)):
            # Check if air temperature without heating is in between boundaries and calculate actual HC power:
            if T_air_0[i] >= T_air_min[t] and T_air_0[i] <= T_air_max[t]:
                Q_Heating_noDR[t, i] = 0
            elif T_air_0[i] < T_air_min[t]:  # heating is required
                Q_Heating_noDR[t, i] = heating_power_10[i] * (T_air_min[t] - T_air_0[i]) / (T_air_10[i] - T_air_0[i])
            elif T_air_0[i] > T_air_max[t]:  # cooling is required
                Q_Cooling_noDR[t, i] = heating_power_10[i] * (T_air_max[t] - T_air_0[i]) / (T_air_10_c[i] - T_air_0[i])

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

    # fill nan
    Q_Cooling_noDR = np.nan_to_num(Q_Cooling_noDR, nan=0)
    Q_Heating_noDR = np.nan_to_num(Q_Heating_noDR, nan=0)
    T_Room_noDR = np.nan_to_num(T_Room_noDR, nan=0)
    Tm_t = np.nan_to_num(Tm_t, nan=0)

    return Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, Tm_t


def get_constant_Q_Tm_t(Buildings, T_outside, T_min_indoor, T_max_indoor, initial_thermal_mass_temp):
    # start time steps to calculate constant values
    runtime = 100
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
    for i in range(T_thermalMass_noDR.shape[1]):
        while -1e-6 >= T_thermalMass_noDR[-1, i] - T_thermalMass_noDR[-2, i] or \
                1e-6 <= T_thermalMass_noDR[-1, i] - T_thermalMass_noDR[-2, i]:
            runtime += 100
            Temperature_outside = np.array([T_outside] * runtime)
            Q_sol = np.zeros((runtime, Buildings.shape[0]))
            Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = ref_HeatingCooling(
                T_outside=Temperature_outside,
                Q_solar=Q_sol,
                initial_thermal_mass_temp=initial_thermal_mass_temp,
                T_air_min=T_min_indoor,
                T_air_max=T_max_indoor,
                Buildings=Buildings)

    return Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR


def calculate_LoadShiftPotential(Buildings, hours_of_preheating, hours_of_shifting, T_outside,
                                 T_min_indoor, T_max_indoor, HouseNr, T_offset_indoor=2, plot_on=False):
    # calculate the thermal mass temperature when there is thermal equilibrium:
    Q_Heating_noDR_constant, Q_Cooling_noDR_constant, T_Room_noDR_constant, T_thermalMass_noDR_constant = \
        get_constant_Q_Tm_t(Buildings, T_outside, T_min_indoor, T_max_indoor, initial_thermal_mass_temp=22)
    # last value is constant value:
    Q_Heating_noDR_constant = Q_Heating_noDR_constant[-1, :]
    Q_Cooling_noDR_constant = Q_Cooling_noDR_constant[-1, :]
    T_Room_noDR_constant = T_Room_noDR_constant[-1, :]
    T_thermalMass_noDR_constant = T_thermalMass_noDR_constant[-1, :]

    # create temperature and solar gains array
    Temperature_outside = np.array([T_outside] * hours_of_preheating)
    # no solar gains are considered
    Q_sol = np.zeros((hours_of_preheating, Buildings.shape[0]))
    # calculate the thermal mass temperature after the time of preheating as well as heating/cooling power
    # this is done by raising the minimum indoor temperature for heating and lowering it for cooling by 2°C
    Q_PreHeating_noDR, Q_PreCooling_noDR, T_PreRoom_noDR, T_PrethermalMass_noDR = ref_HeatingCooling(
        Temperature_outside,
        Q_solar=Q_sol,
        initial_thermal_mass_temp=T_thermalMass_noDR_constant,
        T_air_min=T_min_indoor + T_offset_indoor,
        T_air_max=T_max_indoor - T_offset_indoor,
        Buildings=Buildings)

    # now calculate the heating/cooling power with the old indoor temperature settings starting from the values
    # calculated in the preheating:
    # create temperature and solar gains array
    Temperature_outside = np.array([T_outside] * hours_of_shifting)
    # no solar gains are considered
    Q_sol = np.zeros((hours_of_shifting, Buildings.shape[0]))
    Q_ReducedHeating_noDR, Q_ReducedCooling_noDR, T_ReducedRoom_noDR, T_ReducedthermalMass_noDR = ref_HeatingCooling(
        Temperature_outside,
        Q_solar=Q_sol,
        initial_thermal_mass_temp=T_PrethermalMass_noDR[-1, :],  # take the last one
        T_air_min=T_min_indoor,
        T_air_max=T_max_indoor,
        Buildings=Buildings)

    # calculate and plot the difference between steady state and shifting demand:
    # total thermal power during preheating when load is constant:
    Q_Heating_constant_total_preheating = Q_Heating_noDR_constant * hours_of_preheating
    Q_Heating_constant_total_shifting = Q_Heating_noDR_constant * hours_of_shifting
    Q_PreHeating_noDR_total = Q_PreHeating_noDR.sum(axis=0)
    Q_ReducedHeating_noDR_total = Q_ReducedHeating_noDR.sum(axis=0)

    ExcessHeatPreheat = Q_PreHeating_noDR_total - Q_Heating_constant_total_preheating
    SaveHeatShifting = Q_Heating_constant_total_shifting - Q_ReducedHeating_noDR_total

    # calculate the energy until static temperature is reached again:
    Q_Heating_afterShift, Q_Cooling_afterShift, T_Room_afterShift, T_thermalMass_afterShift = \
        get_constant_Q_Tm_t(Buildings, T_outside, T_min_indoor, T_max_indoor, T_ReducedthermalMass_noDR[-1, :])
    # total heating until steady state is reached after shifting:
    Q_Heating_afterShift_sum = Q_Heating_afterShift.sum(axis=0)
    # Energy still stored in thermal mass after unloading
    RemainingEnergy = (np.tile(Q_Heating_noDR_constant, (Q_Heating_afterShift.shape[0], 1)) - Q_Heating_afterShift) \
        .sum(axis=0)

    # total losses:
    TotalLoss = ExcessHeatPreheat - SaveHeatShifting - RemainingEnergy
    if plot_on:
        plot1, plot2 = plot_heat_demand_and_shifted_bars(
            ExcessHeatPreheat,
            SaveHeatShifting,
            RemainingEnergy,
            TotalLoss,
            Q_PreHeating_noDR,
            Q_ReducedHeating_noDR,
            T_PrethermalMass_noDR,
            T_ReducedRoom_noDR,
            T_ReducedthermalMass_noDR,
            T_PreRoom_noDR,
            Q_Heating_noDR_constant,
            T_thermalMass_noDR_constant,
            T_Room_noDR_constant,
            HouseNr,
            hours_of_preheating,
            hours_of_shifting,
        )
        return plot1, plot2

    return Q_PreHeating_noDR


def plot_heat_demand_and_shifted_bars(
        ExcessHeatPreheat,
        SaveHeatShifting,
        RemainingEnergy,
        TotalLoss,
        Q_PreHeating_noDR,
        Q_ReducedHeating_noDR,
        T_PrethermalMass_noDR,
        T_ReducedRoom_noDR,
        T_ReducedthermalMass_noDR,
        T_PreRoom_noDR,
        Q_Heating_noDR_constant,
        T_thermalMass_noDR_constant,
        T_Room_noDR_constant,
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
    plt.ylabel("Energy (Wh)")

    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.resolve() / f"Energy_shifted_{house_nr}.png")
    plt.savefig(Path(__file__).parent.resolve() / f"Energy_shifted_{house_nr}.svg")

    # plt.show()

    # plot results for one building:
    x_achse = np.arange(preheating_hours + shifting_hours)
    Q_Heating_plot = np.append(Q_PreHeating_noDR[:, house_nr - 1], Q_ReducedHeating_noDR[:, house_nr - 1])
    T_thermalMass_plot = np.append(T_PrethermalMass_noDR[:, house_nr - 1], T_ReducedthermalMass_noDR[:, house_nr - 1])
    T_Room_plot = np.append(T_PreRoom_noDR[:, house_nr - 1], T_ReducedRoom_noDR[:, house_nr - 1])

    fig1, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x_achse, Q_Heating_plot, label="heating power", color="red")
    ax1.axhline(Q_Heating_noDR_constant[house_nr - 1], xmin=0, xmax=1, label="constant heating power", color="black")
    # ax1.vlines(x=0, ymin=Q_Heating_noDR_constant[0], ymax=Q_Heating_plot[0], color="red")
    ax1.axvline(x=preheating_hours - 1, color="black", linestyle="--", linewidth=0.7)

    ax1.fill_between(x_achse, Q_Heating_noDR_constant[house_nr - 1], Q_Heating_plot,
                     where=(Q_Heating_plot > Q_Heating_noDR_constant[house_nr - 1]), color='red', alpha=0.7,
                     label='Additional Energy', interpolate=True)
    ax1.fill_between(x_achse, Q_Heating_noDR_constant[house_nr - 1], Q_Heating_plot,
                     where=(Q_Heating_plot < Q_Heating_noDR_constant[house_nr - 1]), color='green', alpha=0.7,
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

    ax1.set_ylabel("heating power in W")
    ax2.set_ylabel("temperature in °C")
    ax1.set_title("Load shift at " + str(T_outside) + " °C, House Nr " + str(house_nr))
    ax2.set_xlabel("time (h)")
    plt.tight_layout()
    fig1.savefig(Path(__file__).parent.resolve() / f"Subplot_energy_shifted_{house_nr}.png")
    fig1.savefig(Path(__file__).parent.resolve() / f"Subplot_energy_shifted_{house_nr}.svg")
    # plt.show()
    return fig0, fig1


if __name__ == "__main__":
    # path:
    project_directory_path = Path(__file__).parent.resolve()
    # define building data
    Buildings = pd.read_excel(project_directory_path / "Sprungantwort_tests.xlsx", engine="openpyxl")

    # Sprungantwort()
    # compare_solar_radation()

    hours_of_preheating = 3
    hours_of_shifting = 20
    T_outside = -5
    T_min_indoor = 20
    T_max_indoor = 23
    T_offset_indoor = 2
    HouseNr = 3  # startet bei 1! nicht bei 0
    plotON = True
    # installierte Leistung bei -5°C
    for houses in [1, 2, 3, 4]:
        line_plot, barplot = calculate_LoadShiftPotential(Buildings,
                                                          hours_of_preheating,
                                                          hours_of_shifting,
                                                          T_outside,
                                                          T_min_indoor,
                                                          T_max_indoor,
                                                          houses,
                                                          T_offset_indoor=T_offset_indoor,
                                                          plot_on=True)



    installierte_Leistung = calculate_LoadShiftPotential(Buildings,
                                                         hours_of_preheating,
                                                         hours_of_shifting,
                                                         -15,
                                                         T_min_indoor,
                                                         T_max_indoor,
                                                         HouseNr,
                                                         T_offset_indoor=T_offset_indoor,
                                                         plot_on=False)

    installierte_Leistung = installierte_Leistung[0, :]  # installierte leistung für jedes haus

    eingespeicherte_energie = []
    eingespeicherte_energie2d = []
    # temperaturen von -15 bis 18°C
    for temp in range(-15, 18):
        # für temperatur 2d plot:
        Heizleistung_Vorheizen = calculate_LoadShiftPotential(Buildings, hours_of_preheating, hours_of_shifting, temp,
                                                              T_min_indoor, T_max_indoor, HouseNr,
                                                              T_offset_indoor=T_offset_indoor)
        eingespeicherte_energie2d.append(Heizleistung_Vorheizen.sum(axis=0))

        # verschiedene einspeicherzeiten: 1 bis 5 stunden:
        for preheatingHours in range(1, 5):
            Heizleistung_Vorheizen = calculate_LoadShiftPotential(Buildings, preheatingHours, hours_of_shifting, temp,
                                                                  T_min_indoor, T_max_indoor, HouseNr,
                                                                  T_offset_indoor=T_offset_indoor)

            eingespeicherte_energie.append([temp, preheatingHours,
                                            Heizleistung_Vorheizen.sum(axis=0)[0],
                                            Heizleistung_Vorheizen.sum(axis=0)[1],
                                            Heizleistung_Vorheizen.sum(axis=0)[2]])

    eingespeicherte_energie2d = np.vstack(eingespeicherte_energie2d)  # list to matrix
    Daten = np.vstack(eingespeicherte_energie)  # list to matrix
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x_data = Daten[:, 0]  # Temperature
    y_data = Daten[:, 1]  # preheating duration
    z_data1 = Daten[:, 2] / installierte_Leistung[0]  # stored energy house 1
    z_data2 = Daten[:, 3] / installierte_Leistung[1]  # stored energy house 2
    z_data3 = Daten[:, 4] / installierte_Leistung[2]  # stored energy house 3

    ax.scatter3D(x_data, y_data, z_data1, cmap="Blues", label="House 1")
    ax.scatter3D(x_data, y_data, z_data2, cmap="greens", label="House 2")
    ax.scatter3D(x_data, y_data, z_data3, cmap="reds", label="House 3")

    plt.legend()
    ax.set_xlabel("temperature in °C")
    ax.set_ylabel("preheating hours")
    ax.set_zlabel(r'$\frac{stored \; energy}{installed \; HP \; power}$')
    plt.savefig(project_directory_path / "Stored_energy_3D.png")
    plt.show()

    # 2d plot über temperatur
    x_achse = np.arange(-15, 18)
    for i in range(3):
        plt.plot(x_achse, eingespeicherte_energie2d[:, i] / installierte_Leistung[i], label="House " + str(i + 1))

    plt.legend()
    plt.grid()
    plt.xlabel("Temperature in °C")
    plt.ylabel(r'$\frac{stored \; energy}{installed \; HP \; power}$')
    plt.title("Stored energy to installed power, " + str(hours_of_preheating) + " hours preheating")
    plt.savefig(project_directory_path / "Stored_energy_2D_temp.png")
    plt.show()
