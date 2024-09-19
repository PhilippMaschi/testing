import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Building:
    building_data: pd.DataFrame 
    weather_data: pd.DataFrame
    minimum_indoor_temperature: float

    def __post_init__(self):
        self.names = self.building_data["my_name"]
        self.Am_factor = self.building_data["Am_factor"]
        self.Af = self.building_data["Af"]
        self.Cm_factor = self.building_data["CM_factor"]
        self.internal_gains = self.building_data["internal_gains"]
        self.Htr_w = self.building_data["Htr_w"]
        self.Hop = self.building_data["Hop"]
        self.Hve = self.building_data["Hve"]

        # Performing calculations
        self.Am = self.Am_factor * self.Af  # Effective mass related area [m^2]
        self.Cm = self.Cm_factor * self.Af
        self.Atot = 4.5 * self.Af  # Area of all surfaces facing the building zone
        self.Qi = self.internal_gains * self.Af
        self.Htr_ms = 9.1 * self.Am  # From 12.2.2 Equ. (64)
        self.Htr_is = 3.45 * self.Atot
        self.Htr_em = 1 / (1 / self.Hop - 1 / self.Htr_ms)  # From 12.2.2 Equ. (63)
        self.Htr_1 = 1 / (1 / self.Hve + 1 / self.Htr_is)  # Equ. C.6
        self.Htr_2 = self.Htr_1 + self.Htr_w  # Equ. C.7
        self.Htr_3 = 1 / (1 / self.Htr_2 + 1 / self.Htr_ms)  # Equ.C.8
        self.PHI_ia = 0.5 * self.Qi  # Equ. C.1

        # Weather data
        self.T_outside = self.weather_data["temperature"].to_numpy()

        # Solar radiation calculations
        self.Q_solar_north = np.outer(self.weather_data["radiation_north"], self.building_data["effective_window_area_north"])
        self.Q_solar_east = np.outer(self.weather_data["radiation_east"], self.building_data["effective_window_area_west_east"] / 2)
        self.Q_solar_south = np.outer(self.weather_data["radiation_south"], self.building_data["effective_window_area_sout"])
        self.Q_solar_west = np.outer(self.weather_data["radiation_west"], self.building_data["effective_window_area_west_east"] / 2)

        # Combine all solar radiation contributions
        self.Q_solar = (self.Q_solar_north + self.Q_solar_south + self.Q_solar_east + self.Q_solar_west).squeeze()

        self.cooling_power = 0
        self.target_temperature_array_min = np.full(shape=8760, fill_value=self.minimum_indoor_temperature)





def calculate_heating_and_cooling_demand(
            building, thermal_start_temperature: float = 15, static=False
    ) -> (np.array, np.array, np.array, np.array):
        """
        if "static" is True, then the RC model will calculate a static heat demand calculation for the first hour of
        the year by using this hour 100 times. This way a good approximation of the thermal mass temperature of the
        building in the beginning of the calculation is achieved. Solar gains are set to 0.
        Returns: heating demand, cooling demand, indoor air temperature, temperature of the thermal mass
        """
        nr_buildings = len(building.Af)
        heating_power_10 = building.Af * 10

        if building.cooling_power  == 0:
            T_air_max = np.full((8760,), 100)
            # if no cooling is adopted --> raise max air temperature to 100 so it will never cool:
        else:
            T_air_max = building.target_temperature_array_max

        if static:
            Q_solar = np.array([0] * 100)
            T_outside = np.array([building.T_outside[0]] * 100)
            T_air_min = np.array( [building.minimum_indoor_temperature] * 100)
            time = np.arange(100)

            Tm_t = np.zeros(shape=(100, nr_buildings))  # thermal mass temperature
            T_sup = np.zeros(shape=(100, nr_buildings))
            heating_demand = np.zeros(shape=(100, nr_buildings))
            cooling_demand = np.zeros(shape=(100, nr_buildings))
            room_temperature = np.zeros(shape=(100, nr_buildings))

        else:
            Q_solar = building.Q_solar
            T_outside = building.T_outside
            T_air_min = building.target_temperature_array_min
            time = np.arange(8760)

            Tm_t = np.zeros(shape=(8760, nr_buildings))  # thermal mass temperature
            T_sup = np.zeros(shape=(8760, nr_buildings))
            heating_demand = np.zeros(shape=(8760, nr_buildings))
            cooling_demand = np.zeros(shape=(8760, nr_buildings))
            room_temperature = np.zeros(shape=(8760, nr_buildings))

        # RC-Model
        for t in time:  # t is the index for each time step
            # Equ. C.2
            PHI_m = building.Am / building.Atot * (0.5 * building.Qi + Q_solar[t])
            # Equ. C.3
            PHI_st = (1 - building.Am / building.Atot - building.Htr_w / 9.1 / building.Atot) * (
                    0.5 * building.Qi + Q_solar[t]
            )

            # (T_sup = T_outside because incoming air is not preheated)
            T_sup[t] = T_outside[t]

            # Equ. C.5
            PHI_mtot_0 = (
                    PHI_m
                    + building.Htr_em * T_outside[t]
                    + building.Htr_3
                    * (
                            PHI_st
                            + building.Htr_w * T_outside[t]
                            + building.Htr_1 * (((building.PHI_ia + 0) / building.Hve) + T_sup[t])
                    )
                    / building.Htr_2
            )

            # Equ. C.5 with 10 W/m^2 heating power
            PHI_mtot_10 = (
                    PHI_m
                    + building.Htr_em * T_outside[t]
                    + building.Htr_3
                    * (
                            PHI_st
                            + building.Htr_w * T_outside[t]
                            + building.Htr_1
                            * (((building.PHI_ia + heating_power_10) / building.Hve) + T_sup[t])
                    )
                    / building.Htr_2
            )

            # Equ. C.5 with 10 W/m^2 cooling power
            PHI_mtot_10_c = (
                    PHI_m
                    + building.Htr_em * T_outside[t]
                    + building.Htr_3
                    * (
                            PHI_st
                            + building.Htr_w * T_outside[t]
                            + building.Htr_1
                            * (((building.PHI_ia - heating_power_10) / building.Hve) + T_sup[t])
                    )
                    / building.Htr_2
            )

            if t == 0:
                Tm_t_prev = thermal_start_temperature
            else:
                Tm_t_prev = Tm_t[t - 1]

            # Equ. C.4
            Tm_t_0 = (
                             Tm_t_prev * (building.Cm / 3600 - 0.5 * (building.Htr_3 + building.Htr_em))
                             + PHI_mtot_0
                     ) / (building.Cm / 3600 + 0.5 * (building.Htr_3 + building.Htr_em))

            # Equ. C.4 for 10 W/m^2 heating
            Tm_t_10 = (
                              Tm_t_prev * (building.Cm / 3600 - 0.5 * (building.Htr_3 + building.Htr_em))
                              + PHI_mtot_10
                      ) / (building.Cm / 3600 + 0.5 * (building.Htr_3 + building.Htr_em))

            # Equ. C.4 for 10 W/m^2 cooling
            Tm_t_10_c = (
                                Tm_t_prev * (building.Cm / 3600 - 0.5 * (building.Htr_3 + building.Htr_em))
                                + PHI_mtot_10_c
                        ) / (building.Cm / 3600 + 0.5 * (building.Htr_3 + building.Htr_em))

            # Equ. C.9
            T_m_0 = (Tm_t_0 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 heating
            T_m_10 = (Tm_t_10 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 cooling
            T_m_10_c = (Tm_t_10_c + Tm_t_prev) / 2

            # Euq. C.10
            T_s_0 = (
                            building.Htr_ms * T_m_0
                            + PHI_st
                            + building.Htr_w * T_outside[t]
                            + building.Htr_1 * (T_sup[t] + (building.PHI_ia + 0) / building.Hve)
                    ) / (building.Htr_ms + building.Htr_w + building.Htr_1)

            # Euq. C.10 for 10 W/m^2 heating
            T_s_10 = (
                             building.Htr_ms * T_m_10
                             + PHI_st
                             + building.Htr_w * T_outside[t]
                             + building.Htr_1 * (T_sup[t] + (building.PHI_ia + heating_power_10) / building.Hve)
                     ) / (building.Htr_ms + building.Htr_w + building.Htr_1)

            # Euq. C.10 for 10 W/m^2 cooling
            T_s_10_c = (
                               building.Htr_ms * T_m_10_c
                               + PHI_st
                               + building.Htr_w * T_outside[t]
                               + building.Htr_1 * (T_sup[t] + (building.PHI_ia - heating_power_10) / building.Hve)
                       ) / (building.Htr_ms + building.Htr_w + building.Htr_1)

            # Equ. C.11
            T_air_0 = (building.Htr_is * T_s_0 + building.Hve * T_sup[t] + building.PHI_ia + 0) / (
                    building.Htr_is + building.Hve
            )

            # Equ. C.11 for 10 W/m^2 heating
            T_air_10 = (
                               building.Htr_is * T_s_10
                               + building.Hve * T_sup[t]
                               + building.PHI_ia
                               + heating_power_10
                       ) / (building.Htr_is + building.Hve)

            # Equ. C.11 for 10 W/m^2 cooling
            T_air_10_c = (
                                 building.Htr_is * T_s_10_c
                                 + building.Hve * T_sup[t]
                                 + building.PHI_ia
                                 - heating_power_10
                         ) / (building.Htr_is + building.Hve)

            # Check if air temperature without heating is in between boundaries and calculate actual HC power:
            # mask_no_demand = (T_air_0 >= T_air_min[t]) & (T_air_0 <= T_air_max[t])  # vector is already filled with zeros

            # Condition where heating is required
            mask_heating = T_air_0 < T_air_min[t]

            # Condition where cooling is required
            mask_cooling = T_air_0 > T_air_max[t]

            if mask_heating.all():
                heating_demand[t, mask_heating] = (
                    heating_power_10 * (T_air_min[t] - T_air_0[mask_heating]) / (T_air_10 - T_air_0[mask_heating])
                )

            if mask_cooling.all():  # if there is one building that needs it, calculate it
                cooling_demand[t, mask_cooling] = (
                    heating_power_10 * (T_air_max[t] - T_air_0[mask_cooling]) / (T_air_10_c - T_air_0[mask_cooling])
                )

            # if T_air_0 >= T_air_min[t] and T_air_0 <= T_air_max[t]:
            #     heating_demand[t] = 0
            # elif T_air_0 < T_air_min[t]:  # heating is required
            #     heating_demand[t] = (
            #             heating_power_10 * (T_air_min[t] - T_air_0) / (T_air_10 - T_air_0)
            #     )
            # elif T_air_0 > T_air_max[t]:  # cooling is required
            #     cooling_demand[t] = (
            #             heating_power_10 * (T_air_max[t] - T_air_0) / (T_air_10_c - T_air_0)
            #     )

            # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
            # Equ. C.5 with actual heating power
            PHI_mtot_real = (
                    PHI_m
                    + building.Htr_em * T_outside[t]
                    + building.Htr_3
                    * (
                            PHI_st
                            + building.Htr_w * T_outside[t]
                            + building.Htr_1
                            * (
                                    (
                                            (building.PHI_ia + heating_demand[t] - cooling_demand[t])
                                            / building.Hve
                                    )
                                    + T_sup[t]
                            )
                    )
                    / building.Htr_2
            )
            # Equ. C.4
            Tm_t[t] = (
                              Tm_t_prev * (building.Cm / 3600 - 0.5 * (building.Htr_3 + building.Htr_em))
                              + PHI_mtot_real
                      ) / (building.Cm / 3600 + 0.5 * (building.Htr_3 + building.Htr_em))

            # Equ. C.9
            T_m_real = (Tm_t[t] + Tm_t_prev) / 2

            # Euq. C.10
            T_s_real = (
                               building.Htr_ms * T_m_real
                               + PHI_st
                               + building.Htr_w * T_outside[t]
                               + building.Htr_1
                               * (
                                       T_sup[t]
                                       + (building.PHI_ia + heating_demand[t] - cooling_demand[t]) / building.Hve
                               )
                       ) / (building.Htr_ms + building.Htr_w + building.Htr_1)

            # Equ. C.11 for 10 W/m^2 heating
            room_temperature[t] = (
                                          building.Htr_is * T_s_real
                                          + building.Hve * T_sup[t]
                                          + building.PHI_ia
                                          + heating_demand[t]
                                          - cooling_demand[t]
                                  ) / (building.Htr_is + building.Hve)

        return heating_demand, cooling_demand, room_temperature, Tm_t


def calculate_heat_demand_profiles(building):
    _, _, _, mass_temperature = calculate_heating_and_cooling_demand(building=building, static=True)
    BuildingMassTemperatureStartValue = mass_temperature[-1]
    Q_RoomHeating, Q_RoomCooling, T_Room, T_BuildingMass = calculate_heating_and_cooling_demand(building=building, thermal_start_temperature=BuildingMassTemperatureStartValue, static=False)
    return Q_RoomHeating


def import_weather_data(path):
    return pd.read_csv(path, sep=";")



if __name__ == "__main__":
    data = pd.read_csv(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\Sonstiges") / "invert_data_2020_aut_einzelne.csv", sep=";")
    weather_file = import_weather_data(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\Sonstiges") / "pv_gis_weather_vienna.csv")
    building = Building(building_data=data, weather_data=weather_file, minimum_indoor_temperature=20)
    heat_demand = calculate_heat_demand_profiles(building)

    df = pd.DataFrame(heat_demand, columns=building.names)
    df.to_csv(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\Sonstiges") / "heat_demand.csv", sep=";", index=False)


