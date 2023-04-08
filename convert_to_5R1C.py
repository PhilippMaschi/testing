from dataclasses import dataclass

import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class Building:
    Cm: float  # thermal capacity
    HD: float  # heat transfer to external environment
    Hg: float  # heat transmission to the ground
    HU: float  # heat transfer to unconditioned rooms
    HA: float  # heat transfer to adjacent buildings
    Hv: float  # ventilation coefficient
    Hop: float  # heat transfer coefficient through opaque building elements

@dataclass
class Create5R1CParameters:
    def __init__(self, df):
        self.building_df = pd.DataFrame(columns=["ID Building",
                                                 "Af", "Hop", "Htr_w", "Hve", "CM_factor", "Am_factor",
                                                 "internal_gains",
                                                 "effective_window_area_west_east",
                                                 "effective_window_area_south",
                                                 "effective_window_area_north",
                                                 "grid_power_max"])
        self.df = df
        self.length = None
        self.width = None
        self.height = None
        self.specific_heat_capacity = None
        self.foot_print = None
        self.floors = None
        self.roof_area = None
        self.wall_outside_area = None
        self.volume = None
        self.u_wall = None
        self.u_window = None
        self.u_floor = None
        self.u_roof = None
        self.n_air = None
        self.internal_gains = None
        self.Af = None

        self.wall_east = None
        self.wall_west = None
        self.wall_north = None
        self.wall_south = None

        self.window_area_east = None
        self.window_area_west = None
        self.window_area_north = None
        self.window_area_south = None

        self.Cm_total = None
        self.Cm_outer_walls = None
        self.Cm_roof = None
        self.Cm_ground = None

    def fill_params_invert(self):
        self.length = self.df.loc[:, "length_of_building"]
        self.width = self.df.loc[:, "width_of_building"]

        self.specific_heat_capacity = self.df.loc[:, "specific"]
        self.foot_print = self.length * self.width
        self.floors = self.df.loc[:, "number_of_floors"]
        self.height = (self.df.loc[:, "room_height"] + 0.3) * self.floors
        self.roof_area = self.length * self.width
        self.wall_outside_area = self.df.loc[:, "total_vertical_surface_area"]
        self.volume = self.df.loc[:, "heatedvolume"]

        self.u_wall = self.df.loc[:, "u_value_exterior_walls"]
        self.u_window = self.df.loc[:, "u_value_windows1"]
        self.u_floor = self.df.loc[:, "u_value_floor"]
        self.n_air = self.df.loc[:, "n_50"]
        self.internal_gains = self.df.loc[:, "spec_int_gains_cool_watt"]
        self.Af = self.df.loc[:, "heated_area"]

        self.wall_east = self.df.loc[:, "wallarea_east_m2"]
        self.wall_west = self.df.loc[:, "wallarea_west_m2"]
        self.wall_north = self.df.loc[:, "wallarea_north_m2"]
        self.wall_south = self.df.loc[:, "wallarea_south_m2"]

        window_area = self.df.loc[:, "areawindows"]
        share_window_west_east = (1 - self.df.loc[:, "share_of_windows_oriented_to_north"] - self.df.loc[:, "share_of_windows_oriented_to_south"]) / 2
        self.window_area_east = share_window_west_east * window_area
        self.window_area_west = share_window_west_east * window_area
        self.window_area_north = self.df.loc[:, "share_of_windows_oriented_to_north"] * window_area
        self.window_area_south = self.df.loc[:, "share_of_windows_oriented_to_south"] * window_area

        self.percentage_of_building_surface_attached_length = self.df.loc[:, "percentage_of_building_surface_attached_length"]
        self.percentage_of_building_surface_attached_width = self.df.loc[:, "percentage_of_building_surface_attached_width"]

        self.share_of_window_area_on_gross_surface_area = self.df.loc[:, "share_of_window_area_on_gross_surface_area"]

        # Whole area ext walls detached
        self.waewd = 2 * (self.length * (1 - self.percentage_of_building_surface_attached_length) +
                                        self.width * (1 - self.percentage_of_building_surface_attached_width)) * \
                                        self.height * 1  # f_gliederung

        self.areawindows = self.waewd * self.share_of_window_area_on_gross_surface_area

        # Area ext walls detached
        self._bc.aewd = self.waewd - self._bc.areadoors - self._bc.areawindows
        # Area ceiling
        self._bc.areaceiling = self.lengthbywidth
        # Area floor
        self._bc.areafloor = self.lengthbywidth

    def calculate_Atot(self, floor_area):
        """
        Λat is estimated to be 4.5 after DIN ISO 13790 (2008)
        Args:
            floor_area:

        Returns:

        """
        return floor_area * 4.5

    def total_area_of_building_elements(self):
        return self.foot_print + self.roof_area + self.wall_outside_area + self.area_inner_walls

    def specific_capacity_per_sqm(self):
        cm_wall_spec = self.Cm_outer_walls / self.wall_outside_area
        cm_ground_spec = self.Cm_ground / self.foot_print
        cm_roof_spec = self.Cm_roof / self.roof_area
        cm_inner_wall_spec = self.Cm_inner_walls / self.area_inner_walls
        return cm_wall_spec + cm_ground_spec + cm_roof_spec + cm_inner_wall_spec

    def calculate_Am(self):
        """Am is Cm^2 divided by the area of building elements * specific capacity^2"""
        Cm2 = self.Cm_total * self.Cm_total
        km_wall = (self.Cm_outer_walls / self.wall_outside_area) * (self.Cm_outer_walls / self.wall_outside_area)
        km_ceiling = (self.Cm_roof / self.roof_area) * (self.Cm_roof / self.roof_area)
        km_ground = (self.Cm_ground / self.foot_print) * (self.Cm_ground / self.foot_print)
        km_inner_wall = (self.Cm_inner_walls / self.area_inner_walls) * (self.Cm_inner_walls / self.area_inner_walls)

        return Cm2 / (km_wall * self.wall_outside_area + km_ground * self.foot_print + km_ceiling * self.roof_area + km_inner_wall * self.area_inner_walls)

    def calculate_HD(self):
        """heat transfer to external environment"""
        # Area of the walls * U value of the walls
        return self.wall_outside_area * self.u_wall + self.roof_area * self.u_roof

    def calculate_Hg(self):
        """heat transmission to the ground after DIN ISO 13370 ignoring Ψg*P"""
        # Area of Ground * U value of the floor
        return self.foot_print * self.u_floor

    def calculate_HU(self,
                     wall_area_unconditioned_to_outside,
                     u_value_unconditioned_to_outside,
                     wall_area_inside_to_unconditioned,
                     u_value_inside_to_unconditioned):
        """heat transfer to unconditioned rooms after DIN ISO 13789 (2007)"""
        Hiu = wall_area_inside_to_unconditioned * u_value_inside_to_unconditioned
        Hue = wall_area_unconditioned_to_outside * u_value_unconditioned_to_outside
        if Hiu == 0 and Hue == 0:
            return np.full((len(self.Af),), 0)  # return 0 if no unconditioned rooms are there
        else:
            return Hiu * Hue / (Hiu + Hue)

    def calculate_HA(self,
                     area_to_adjacent_building,
                     u_value_to_adjacent_building,
                     inside_temperature,
                     inside_temperature_adjacent_building,
                     outside_temperature):
        """heat transfer to adjacent buildings after DIN ISO 13789 (2007)"""
        if inside_temperature != 0 and outside_temperature != 0:
            b = (inside_temperature - inside_temperature_adjacent_building) / (inside_temperature - outside_temperature)
        else:
            b = 0
        return area_to_adjacent_building * u_value_to_adjacent_building * b

    def calculate_Hop(self):
        """Hop is the heat transfer coefficient through opaque building elements and is the sum of
        Hg, HU, HA and HD"""
        Hg = self.calculate_Hg()
        HU = self.calculate_HU(0, 1, 0, 1)  # buildings do not have unconditioned rooms
        HA = self.calculate_HA(0, 1, 0, 0, 0)  # buildings are stand-alone (no adjacent area)
        HD = self.calculate_HD()
        return Hg + HU + HA + HD

    def calculate_Htr_w(self):
        """
        heat transfer through windows is area of windows * u value window (DIN ISO 13790, 2007) 8.3 equ. 18
        Returns:

        """
        # ignoring the 0.7 factor which would represent the glazing
        window_area = self.window_area_north + self.window_area_south + self.window_area_east + self.window_area_west
        return self.u_window * window_area

    def calculate_Hve(self):
        """
        ventilation coefficient calculated after DIN ISO 13789 (2007) chapter 9.3, equ 21
        air volume flow has to be provided in m^3/h
        """
        roh_ca = 1_200 / 3_600  # Wh/m3K
        air_volume_flow = self.volume * self.n_air
        return roh_ca * air_volume_flow

    def fill_building_df(self):
        self.building_df.loc[:, "ID Building"] = np.arange(1, len(self.Af)+1)
        self.building_df.loc[:, "Af"] = self.Af
        self.building_df.loc[:, "Hop"] = self.calculate_Hop()
        self.building_df.loc[:, "Htr_w"] = self.calculate_Htr_w()
        self.building_df.loc[:, "Hve"] = self.calculate_Hve()
        self.building_df.loc[:, "CM_factor"] = self.Cm_total / self.Af
        self.building_df.loc[:, "Am_factor"] = self.calculate_Am() / self.Af
        self.building_df.loc[:, "internal_gains"] = self.internal_gains
        self.building_df.loc[:, "effective_window_area_west_east"] = (self.window_area_east * self.df.loc[:, "median_gWert_weighted_mean_AF"] +
                                                                      self.window_area_west * self.df.loc[:, "median_gWert_weighted_mean_AF"])
        self.building_df.loc[:, "effective_window_area_south"] = self.window_area_south * self.df.loc[:, "median_gWert_weighted_mean_AF"]
        self.building_df.loc[:, "effective_window_area_north"] = self.window_area_north * self.df.loc[:, "median_gWert_weighted_mean_AF"]
        self.building_df.loc[:, "grid_power_max"] = np.full((len(self.Af,)), 21_000)



    def main(self):
        self.fill_params()

        self.fill_building_df()
        output_dir = Path(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\5R1C_validation\output")
        self.building_df.to_excel(output_dir / Path("5R1C_buildings.xlsx"), index=False)


if __name__ == "__main__":
    import_directory = Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\building_data\AUT")

    filename = "2030.parquet.gzip"
    frame = pd.read_parquet(import_directory / Path(filename))
    Create5R1CParameters(frame).main()
