import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt



class HeatingCooling_noDR:

    def ref_HeatingCooling(self, T_outside=None, Q_solar=None, initial_thermal_mass_temp=20, T_air_min=20, T_air_max=27,
                           **kwargs):
        """
        This function calculates the heating and cooling demand as well as the indoor temperature for every building
        category based in the 5R1C model. The results are hourls vectors for one year. Q_solar is imported from a CSV
        at the time! Inputs for T_outside and Q_solar have to be numpy arrays, otherwise the temperature and radiation
        is taken from the database.
        """
        # ------------------------------------------------------------------------------------------
        # this is for the purpose of using the function outside of the database with specific inputs
        if "Buildings" in kwargs:
            self.InternalGains = kwargs["Buildings"]['spec_int_gains_cool_watt'].to_numpy()
            self.Hop = kwargs["Buildings"]['Hop'].to_numpy()
            self.Htr_w = kwargs["Buildings"]['Htr_w'].to_numpy()
            self.Hve = kwargs["Buildings"]['Hve'].to_numpy()
            self.CM_factor = kwargs["Buildings"]['CM_factor'].to_numpy()
            self.Am_factor = kwargs["Buildings"]['Am_factor'].to_numpy()

        if isinstance(T_outside, np.ndarray):
            pass
        else:
            pass
            # get outside temperature from database
            # T_outside = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn).Temperature.to_numpy()
        if isinstance(T_air_min, int):
            T_air_min = np.full((len(T_outside),), T_air_min)
            T_air_max = np.full((len(T_outside),), T_air_max)
        else:
            pass
        if isinstance(Q_solar, np.ndarray):
            pass
        else:
            # calculate solar gains from database
            # solar gains from different celestial directions
            # radiation = DB().read_DataFrame(REG_Table().Gen_Sce_Weather_Radiation_SkyDirections, self.Conn)
            # Q_sol_north = np.outer(radiation.north.to_numpy(), self.AreaWindowSouth)
            # Q_sol_east = np.outer(radiation.east.to_numpy(), self.AreaWindowEastWest / 2)
            # Q_sol_south = np.outer(radiation.south.to_numpy(), self.AreaWindowSouth)
            # Q_sol_west = np.outer(radiation.west.to_numpy(), self.AreaWindowEastWest / 2)
            #
            # Q_solar = ((Q_sol_north + Q_sol_south + Q_sol_east + Q_sol_west).squeeze())
            pass
        # ------------------------------------------------------------------------------------------

        # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
        Atot = 4.5 * self.Af
        # Speicherkapazität J/K
        Cm = self.CM_factor * self.Af
        # wirksame Massenbezogene Fläche [m^2]
        Am = self.Am_factor * self.Af
        # internal gains
        Q_InternalGains = self.InternalGains * self.Af
        timesteps = np.arange(len(T_outside))

        # Kopplung Temp Luft mit Temp Surface Knoten s
        his = np.float_(3.45)  # 7.2.2.2
        # kopplung zwischen Masse und  zentralen Knoten s (surface)
        hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
        Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
        Htr_em = 1 / (1 / self.Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
        # thermischer Kopplungswerte W/K
        Htr_is = his * Atot
        Htr_1 = np.float_(1) / (np.float_(1) / self.Hve + np.float_(1) / Htr_is)  # Equ. C.6
        Htr_2 = Htr_1 + self.Htr_w  # Equ. C.7
        Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8

        # Equ. C.1
        PHI_ia = 0.5 * Q_InternalGains

        Tm_t = np.zeros(shape=(len(timesteps), len(self.Hve)))
        T_sup = np.zeros(shape=(len(timesteps),))
        Q_Heating_noDR = np.zeros(shape=(len(timesteps), len(self.Hve)))
        Q_Cooling_noDR = np.zeros(shape=(len(timesteps), len(self.Hve)))
        T_Room_noDR = np.zeros(shape=(len(timesteps), len(self.Hve)))
        heating_power_10 = self.Af * 10

        for t in timesteps:  # t is the index for each timestep
            # Equ. C.2
            PHI_m = Am / Atot * (0.5 * Q_InternalGains + Q_solar[t, :])
            # Equ. C.3
            PHI_st = (1 - Am / Atot - self.Htr_w / 9.1 / Atot) * \
                     (0.5 * Q_InternalGains + Q_solar[t, :])

            # (T_sup = T_outside weil die Zuluft nicht vorgewärmt oder vorgekühlt wird)
            T_sup[t] = T_outside[t]

            # Equ. C.5
            PHI_mtot_0 = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                    PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (((PHI_ia + 0) / self.Hve) + T_sup[t])) / \
                         Htr_2

            # Equ. C.5 with 10 W/m^2 heating power
            PHI_mtot_10 = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                    PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                    ((PHI_ia + heating_power_10) / self.Hve) + T_sup[t])) / Htr_2

            # Equ. C.5 with 10 W/m^2 cooling power
            PHI_mtot_10_c = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                    PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                    ((PHI_ia - heating_power_10) / self.Hve) + T_sup[t])) / Htr_2

            if t == 0:
                if type(initial_thermal_mass_temp) == int or type(initial_thermal_mass_temp) == float:
                    Tm_t_prev = np.array([initial_thermal_mass_temp] * len(self.Hve))
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
            T_s_0 = (Htr_ms * T_m_0 + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                     (T_sup[t] + (PHI_ia + 0) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

            # Euq. C.10 for 10 W/m^2 heating
            T_s_10 = (Htr_ms * T_m_10 + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                      (T_sup[t] + (PHI_ia + heating_power_10) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

            # Euq. C.10 for 10 W/m^2 cooling
            T_s_10_c = (Htr_ms * T_m_10_c + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                        (T_sup[t] + (PHI_ia - heating_power_10) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

            # Equ. C.11
            T_air_0 = (Htr_is * T_s_0 + self.Hve * T_sup[t] + PHI_ia + 0) / \
                      (Htr_is + self.Hve)

            # Equ. C.11 for 10 W/m^2 heating
            T_air_10 = (Htr_is * T_s_10 + self.Hve * T_sup[t] + PHI_ia + heating_power_10) / \
                       (Htr_is + self.Hve)

            # Equ. C.11 for 10 W/m^2 cooling
            T_air_10_c = (Htr_is * T_s_10_c + self.Hve * T_sup[t] + PHI_ia - heating_power_10) / \
                         (Htr_is + self.Hve)

            for i in range(len(self.Hve)):
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
                    PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                    ((PHI_ia + Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / self.Hve) + T_sup[t])) / Htr_2
            # Equ. C.4
            Tm_t[t, :] = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_real) / \
                         (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.9
            T_m_real = (Tm_t[t, :] + Tm_t_prev) / 2

            # Euq. C.10
            T_s_real = (Htr_ms * T_m_real + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                        (T_sup[t] + (PHI_ia + Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / self.Hve)) / \
                       (Htr_ms + self.Htr_w + Htr_1)

            # Equ. C.11 for 10 W/m^2 heating
            T_Room_noDR[t, :] = (Htr_is * T_s_real + self.Hve * T_sup[t] + PHI_ia +
                                 Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / (Htr_is + self.Hve)

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
    Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = HeatingCooling_noDR().ref_HeatingCooling(Temperature_outside,
                                                                                                   Q_solar=Q_sol,
                                                                                                   initial_thermal_mass_temp=initial_thermal_mass_temp,
                                                                                                   T_air_min=T_min_indoor,
                                                                                                   T_air_max=T_max_indoor,
                                                                                                               **Buildings)
    # check if stationary condition has set in
    for i in range(T_thermalMass_noDR.shape[1]):
        while -1e-6 >= T_thermalMass_noDR[-1, i] - T_thermalMass_noDR[-2, i] or \
                1e-6 <= T_thermalMass_noDR[-1, i] - T_thermalMass_noDR[-2, i]:
            runtime += 100
            Temperature_outside = np.array([T_outside] * runtime)
            Q_sol = np.zeros((runtime, Buildings.shape[0]))
            Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR = HeatingCooling_noDR().ref_HeatingCooling(
                Temperature_outside,
                Q_solar=Q_sol,
                initial_thermal_mass_temp=initial_thermal_mass_temp,
                T_air_min=T_min_indoor,
                T_air_max=T_max_indoor,
                **Buildings)

    return Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, T_thermalMass_noDR


def calculate_LoadShiftPotential(Buildings, hours_of_preheating, hours_of_shifting, T_outside,
                                 T_min_indoor, T_max_indoor, HouseNr, T_offset_indoor=2):
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
    Q_PreHeating_noDR, Q_PreCooling_noDR, T_PreRoom_noDR, T_PrethermalMass_noDR = HeatingCooling_noDR().ref_HeatingCooling(
        Temperature_outside,
        Q_solar=Q_sol,
        initial_thermal_mass_temp=T_thermalMass_noDR_constant,
        T_air_min=T_min_indoor + T_offset_indoor,
        T_air_max=T_max_indoor - T_offset_indoor,
        **Buildings)

    # now calculate the heating/cooling power with the old indoor temperature settings starting from the values
    # calculated in the preheating:
    # create temperature and solar gains array
    Temperature_outside = np.array([T_outside] * hours_of_shifting)
    # no solar gains are considered
    Q_sol = np.zeros((hours_of_shifting, Buildings.shape[0]))
    Q_ReducedHeating_noDR, Q_ReducedCooling_noDR, T_ReducedRoom_noDR, T_ReducedthermalMass_noDR = HeatingCooling_noDR().ref_HeatingCooling(
        Temperature_outside,
        Q_solar=Q_sol,
        initial_thermal_mass_temp=T_PrethermalMass_noDR[-1, :],  # take the last one
        T_air_min=T_min_indoor,
        T_air_max=T_max_indoor,
        **Buildings)

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
    RemainingEnergy = (np.tile(Q_Heating_noDR_constant, (Q_Heating_afterShift.shape[0], 1)) - Q_Heating_afterShift)\
        .sum(axis=0)

    # total losses:
    TotalLoss = ExcessHeatPreheat - SaveHeatShifting - RemainingEnergy


    if plotON:
        # plots:
        fig0 = plt.figure()
        plt.bar(["preheating", "discharging"], [ExcessHeatPreheat[0], SaveHeatShifting[0]], color=["red", "green"])
        plt.bar(["preheating", "discharging"], [0, RemainingEnergy[0]], color=["red", "orange"], bottom=SaveHeatShifting[0])
        plt.bar(["preheating", "discharging"], [0, TotalLoss[0]], color=["red", "grey"], bottom=SaveHeatShifting[0] + RemainingEnergy[0])

        plt.text("preheating", ExcessHeatPreheat[0]/2, "additional \n energy", ha="center")
        plt.text("discharging", SaveHeatShifting[0]/2, "reduced energy", ha="center")
        plt.text("discharging", SaveHeatShifting[0] + RemainingEnergy[0] / 2, "energy remaining \n in thermal mass", ha="center")
        plt.text("discharging", SaveHeatShifting[0] + RemainingEnergy[0] + TotalLoss[0] / 2, "thermal losses", ha="center")

        plt.title("Energy shifting at " + str(T_outside) + "°C, house Nr " + str(HouseNr))
        plt.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Barplot_energy_shifted_" +
                    str(HouseNr) + ".png")
        plt.show()


        # plot results for one building:
        x_achse = np.arange(hours_of_preheating + hours_of_shifting)
        Q_Heating_plot = np.append(Q_PreHeating_noDR[:, HouseNr - 1], Q_ReducedHeating_noDR[:, HouseNr - 1])
        T_thermalMass_plot = np.append(T_PrethermalMass_noDR[:, HouseNr - 1], T_ReducedthermalMass_noDR[:, HouseNr - 1])
        T_Room_plot = np.append(T_PreRoom_noDR[:, HouseNr - 1], T_ReducedRoom_noDR[:, HouseNr - 1])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.bar(x_achse, Q_Heating_plot, label="heating power", color="red")
        ax1.axhline(Q_Heating_noDR_constant[HouseNr - 1], xmin=0, xmax=1, label="constant heating power", color="black")
        # ax1.vlines(x=0, ymin=Q_Heating_noDR_constant[0], ymax=Q_Heating_plot[0], color="red")
        ax1.axvline(x=hours_of_preheating - 1, color="black", linestyle="--", linewidth=0.5)

        ax2.plot(x_achse, T_thermalMass_plot, label="thermal mass temp", color="blue")
        ax2.hlines(T_thermalMass_noDR_constant[HouseNr - 1], xmin=0, xmax=hours_of_preheating + hours_of_shifting - 1, color="purple",
                   label="constant thermal mass temp")
        ax2.vlines(x=0, ymin=T_thermalMass_noDR_constant[0], ymax=T_thermalMass_plot[0], color="blue")
        ax2.axvline(x=hours_of_preheating - 1, color="black", linestyle="--", linewidth=0.5)

        ax3.plot(x_achse, T_Room_plot, label="Room temp", color="green")
        ax3.hlines(T_Room_noDR_constant[0], xmin=0, xmax=hours_of_preheating + hours_of_shifting - 1, color="skyblue",
                   label="constant room temp")
        ax3.vlines(x=0, ymin=T_Room_noDR_constant[HouseNr - 1], ymax=T_Room_plot[0], color="green")
        ax3.axvline(x=hours_of_preheating - 1, color="black", linestyle="--", linewidth=0.5)

        ax1.legend(loc="lower left")
        ax2.legend()
        ax3.legend()
        ax1.set_ylabel("heating power in W")
        ax2.set_ylabel("temperature in °C")
        ax3.set_ylabel("temperature in °C")
        ax3.set_xlabel("hours")
        ax1.set_title("Load shift at " + str(T_outside) + " °C, House Nr " + str(HouseNr))
        plt.tight_layout()
        fig.savefig(
            "C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Subplot_energy_shifted_" + str(
                HouseNr) + ".png")
        plt.show()

    return Q_PreHeating_noDR



if __name__ == "__main__":
    # path:
    project_directory_path = Path(__file__).parent.resolve()
    base_input_path = project_directory_path / "inputdata"
    # define building data
    Buildings = pd.read_excel(base_input_path / "Sprungantwort_tests.xlsx", engine="openpyxl")


    # Sprungantwort()
    # compare_solar_radation()

    hours_of_preheating = 3
    hours_of_shifting = 30
    T_outside = -15
    T_min_indoor = 20
    T_max_indoor = 26
    T_offset_indoor = 2
    HouseNr = 3  # startet bei 1! nicht bei 0
    plotON = False
    # installierte Leistung bei -15°C
    installierte_Leistung = calculate_LoadShiftPotential(Buildings, hours_of_preheating, hours_of_shifting, -15,
                                 T_min_indoor, T_max_indoor, HouseNr, T_offset_indoor=T_offset_indoor)
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
                                     T_min_indoor, T_max_indoor, HouseNr, T_offset_indoor=T_offset_indoor)

            eingespeicherte_energie.append([temp, preheatingHours, Heizleistung_Vorheizen.sum(axis=0)[0],
                                                                   Heizleistung_Vorheizen.sum(axis=0)[1],
                                                                   Heizleistung_Vorheizen.sum(axis=0)[2]])


    eingespeicherte_energie2d = np.vstack(eingespeicherte_energie2d)  # list to matrix
    Daten = np.vstack(eingespeicherte_energie)  # list to matrix
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x_data = Daten[:, 0]  # Temperature
    y_data = Daten[:, 1]   # preheating duration
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
    plt.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Stored_energy_3D.png")
    plt.show()

    # 2d plot über temperatur
    x_achse = np.arange(-15, 18)
    for i in range(3):
        plt.plot(x_achse, eingespeicherte_energie2d[:, i]/installierte_Leistung[i], label="House "+str(i+1))

    plt.legend()
    plt.grid()
    plt.xlabel("Temperature in °C")
    plt.ylabel(r'$\frac{stored \; energy}{installed \; HP \; power}$')
    plt.title("Stored energy to installed power, " + str(hours_of_preheating) + " hours preheating")
    plt.savefig("C:\\Users\\mascherbauer\\PycharmProjects\\NewTrends\\Myfigs\\Paper_figs\\Stored_energy_2D_temp.png")
    plt.show()

