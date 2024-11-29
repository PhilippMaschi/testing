import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from pyproj import CRS, Transformer
import urllib.error
import os

TEMP_DATA = Path(__file__).parent.parent / "PVGIS" / "temp"
TEMP_DATA.mkdir(exist_ok=True, parents=True)


class PVGIS:

    def __init__(self):
        self.id_hour = np.arange(1, 8761)

    def get_nuts_center(
            self,
            region_id,
            url_nuts0_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_0.geojson",
            url_nuts1_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_1.geojson",
            url_nuts2_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_2.geojson",
            url_nuts3_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_3.geojson"
    ):
        """
        This function returns the latitude and longitude of the center of a NUTS region.
        """
        region_id = region_id.strip()
        if len(region_id) == 2:
            url = url_nuts0_poly
        elif len(region_id) == 3:
            url = url_nuts1_poly
        elif len(region_id) == 4:
            url = url_nuts2_poly
        elif len(region_id) > 4:
            url = url_nuts3_poly
        else:
            assert False, f"Error could not identify nuts level for {region_id}"
        nuts = gpd.read_file(url)
        transformer = Transformer.from_crs(CRS("EPSG:3035"), CRS("EPSG:4326"))
        point = nuts[nuts.NUTS_ID == region_id].centroid.values[0]
        return transformer.transform(point.y, point.x)  # returns lat, lon

    def get_PV_generation(self, lat, lon, startyear, endyear, nuts_id) -> np.array:
        # % JRC data
        # possible years are 2005 to 2017
        pvCalculation = 1  # 0 for no and 1 for yes
        peakPower = 1  # kWp
        pvLoss = 14  # system losses in %
        pvTechChoice = "crystSi"  # Choices are: "crystSi", "CIS", "CdTe" and "Unknown".
        trackingtype = 0  # Type of suntracking used, 0=fixed, 1=single horizontal axis aligned north-south,
        # 2=two-axis tracking, 3=vertical axis tracking, 4=single horizontal axis aligned east-west,
        # 5=single inclined axis aligned north-south.
        optimalInclination = 1  # Calculate the optimum inclination angle. Value of 1 for "yes".
        # All other values (or no value) mean "no". Not relevant for 2-axis tracking.
        optimalAngles = 1  # Calculate the optimum inclination AND orientation angles. Value of 1 for "yes".
        # All other values (or no value) mean "no". Not relevant for tracking planes.

        req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&" \
              f"lon={lon}&" \
              f"startyear={startyear}&" \
              f"endyear={endyear}&" \
              f"pvcalculation={pvCalculation}&" \
              f"peakpower={peakPower}&" \
              f"loss={pvLoss}&" \
              f"pvtechchoice={pvTechChoice}&" \
              f"components={1}&" \
              f"trackingtype={trackingtype}&" \
              f"optimalinclination={optimalInclination}&" \
              f"optimalangles={optimalAngles}"

        try:
            # read the csv from api and set column names to list of 20 because depending on input parameters the number
            # of rows will vary. This way all parameters are included for sure, empty rows are dropped afterwards:
            df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
            # drop rows with nan:
            df = df.dropna().reset_index(drop=True)
            # set header to first column
            header = df.iloc[0]
            df = df.iloc[1:, :]
            df.columns = header
            df = df.reset_index(drop=True)
            PV_Profile = pd.to_numeric(df["P"]).to_numpy()  # W
            return PV_Profile
        except urllib.error.HTTPError:  # Error when nuts center is somewhere where there is no PVGIS data
            PV_Profile = None
            print("PV Data is not available for this location {}".format(nuts_id))
            return PV_Profile

    def get_temperature_and_solar_radiation(self, lat, lon, aspect, startyear, endyear, nuts_id) -> pd.DataFrame:
        # % JRC data
        # possible years are 2005 to 2017
        pvCalculation = 0  # 0 for no and 1 for yes
        peakPower = 1  # kWp
        pvLoss = 14  # system losses in %
        pvTechChoice = "crystSi"  # Choices are: "crystSi", "CIS", "CdTe" and "Unknown".
        trackingtype = 0  # Type of suntracking used, 0=fixed, 1=single horizontal axis aligned north-south,
        # 2=two-axis tracking, 3=vertical axis tracking, 4=single horizontal axis aligned east-west,
        # 5=single inclined axis aligned north-south.
        # angle is set to 90° because we are looking at a vertical plane
        angle = 90  # Inclination angle from horizontal plane
        optimalInclination = 0  # Calculate the optimum inclination angle. Value of 1 for "yes".
        # All other values (or no value) mean "no". Not relevant for 2-axis tracking.
        optimalAngles = 0  # Calculate the optimum inclination AND orientation angles. Value of 1 for "yes".
        # All other values (or no value) mean "no". Not relevant for tracking planes.

        req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&" \
              f"lon={lon}&" \
              f"startyear={startyear}&" \
              f"endyear={endyear}&" \
              f"pvcalculation={pvCalculation}&" \
              f"peakpower={peakPower}&" \
              f"loss={pvLoss}&" \
              f"pvtechchoice={pvTechChoice}&" \
              f"components={1}&" \
              f"trackingtype={trackingtype}&" \
              f"optimalinclination={optimalInclination}&" \
              f"optimalangles={optimalAngles}&" \
              f"angle={angle}&" \
              f"aspect={aspect}"

        # read the csv from api and set column names to list of 20 because depending on input parameters the number
        # of rows will vary. This way all parameters are included for sure, empty rows are dropped afterwards:
        nuts_not_working = []
        try:
            df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
            # drop rows with nan:
            df = df.dropna().reset_index(drop=True)
            # set header to first column
            header = df.iloc[0]
            df = df.iloc[1:, :]
            df.columns = header
            return df
        except urllib.error.HTTPError:  # Error when nuts center is somewhere where there is no PVGIS data
            nuts_not_working.append(nuts_id)
            print("{} does not have a valid location for PV-GIS".format(nuts_id))
            print(nuts_not_working)
            return None


    def get_PVGIS_data(self,
                       country: str,
                       nuts_id_list: list,
                       start_year: int,
                       end_year: int,
                    ) -> None:
        """
        This function gets the solar radiation for every cilestial direction (North, East, West, South) on a vertical
        plane. This radiation will later be multiplied with the respective window areas to calculate the radiation
        gains. The function takes the latitude and longitude as input and the start and endyear. Possible years are
        2005 until 2017.

                    Explanation of header of returned dataframe:
                #  P: PV system power (W)
                #  Gb(i): Beam (direct) irradiance on the inclined plane (plane of the array) (W/m2)
                #  Gd(i): Diffuse irradiance on the inclined plane (plane of the array) (W/m2)
                #  Gr(i): Reflected irradiance on the inclined plane (plane of the array) (W/m2)
                #  H_sun: Sun height (degree)
                #  T2m: 2-m air temperature (degree Celsius)
                #  WS10m: 10-m total wind speed (m/s)
                #  Int: 1 means solar radiation values are reconstructed

        In addition the temperatur for different nuts_ids is saved as well as the produced PV profile for a certain PV
        peak power.
        The output is NUTS2_PV_Data, NUTS2_Radiation_Data and NUTS2_Temperature_Data in the SQLite Database.
        """
        # clear temp Data
        for file in TEMP_DATA.iterdir():
            file.unlink()
        # determine if NUTS1,2 or 3:
        if len(nuts_id_list[0]) == 3:  # NUTS1
            nuts_level = "1"
        elif len(nuts_id_list[0]) == 4:
            nuts_level = "2"
        elif len(nuts_id_list[0]) == 5:
            nuts_level = "3"

        for nuts in nuts_id_list:
            table_name_temperature = f"Temperature_NUTS{nuts}_{country}.parquet.gzip"
            table_name_radiation = f"Radiation_NUTS{nuts}_{country}.parquet.gzip"
            table_name_PV = f"PV_generation_NUTS{nuts}_{country}.parquet.gzip"
            lat, lon = self.get_nuts_center(nuts)
            # CelestialDirections are "north", "south", "east", "west":
            # Orientation (azimuth) angle of the (fixed) plane, 0=south, 90=west, -90=east. Not relevant for tracking planes
            valid_nuts_id = True
            for CelestialDirection in ["south", "east", "west", "north"]:
                if CelestialDirection == "south":
                    aspect = 0
                    df_south = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year, nuts)
                    if df_south is None:
                        valid_nuts_id = False
                        break
                    south_radiation = pd.to_numeric(df_south["Gb(i)"]) + pd.to_numeric(df_south["Gd(i)"])
                elif CelestialDirection == "east":
                    aspect = -90
                    df_east = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year, nuts)
                    east_radiation = pd.to_numeric(df_east["Gb(i)"]) + pd.to_numeric(df_east["Gd(i)"])
                elif CelestialDirection == "west":
                    aspect = 90
                    df_west = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year, nuts)
                    west_radiation = pd.to_numeric(df_west["Gb(i)"]) + pd.to_numeric(df_west["Gd(i)"])
                elif CelestialDirection == "north":
                    aspect = -180
                    df_north = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year, nuts)
                    north_radiation = pd.to_numeric(df_north["Gb(i)"]) + pd.to_numeric(df_north["Gd(i)"])
            if not valid_nuts_id:
                continue

            # save solar radiation
            radiation_table = {"south": south_radiation.reset_index(drop=True).to_numpy(),
                               "east": east_radiation.reset_index(drop=True).to_numpy(),
                               "west": west_radiation.reset_index(drop=True).to_numpy(),
                               "north": north_radiation.reset_index(drop=True).to_numpy()
                            }
            solar_radiation_table = pd.DataFrame(radiation_table)
            solar_radiation_table.to_parquet(path=TEMP_DATA / table_name_radiation)

            # save temperature
            outsideTemperature = pd.to_numeric(df_south["T2m"].reset_index(drop=True).rename("temperature"))
            temperature_dict = {"temperature": outsideTemperature.to_numpy()}
            # write table for outside temperature:
            temperature_table = pd.DataFrame(temperature_dict)
            temperature_table.to_parquet(TEMP_DATA / table_name_temperature)

            # PV profiles:
            unit = np.full((len(self.id_hour),), "W")

            PVProfile = self.get_PV_generation(lat, lon, start_year, end_year, nuts)

            pv_type = np.full((len(PVProfile),), 1)
            columns_pv = {"power": PVProfile}
            pv_table = pd.DataFrame(columns_pv)
            pv_table.to_parquet(TEMP_DATA / table_name_PV)

            print(f"{nuts} saved to TEMP DATA")

    def calculate_single_profile_for_country(self, country: str, nuts_level: int, year: int) -> None:
        """nuts level 1, 2 or 3

        Args:
            country: european country code 2 digits (eg. DE, AT, ES...)
            nuts_level: int [0, 1, 2, 3]
        """
        # weighted average over ground floor area from hotmaps
        absolut_path = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"residential_floor_area.json")
        floor_area_total = pd.read_json(absolut_path, orient="table")
        # select the floor area on the respective nuts level
        if nuts_level == 0:
            mask = floor_area_total["nuts_id"].str.len() == 2
            floor_area = floor_area_total.loc[mask]
        elif nuts_level == 1:
            mask = floor_area_total["nuts_id"].str.len() == 3
            floor_area = floor_area_total.loc[mask]
        elif nuts_level == 2:
            mask = floor_area_total["nuts_id"].str.len() == 4
            floor_area = floor_area_total.loc[mask]
        elif nuts_level == 3:
            mask = floor_area_total["nuts_id"].str.len() == 5
            floor_area = floor_area_total.loc[mask]
        else:
            assert "nuts level has to be integer of [0, 1, 2, 3]"
        # select the floor area for the respective country
        floor_area = floor_area[floor_area["nuts_id"].str.startswith(country)].reset_index(drop=True)
        # total sum of floor area
        floor_area_sum = floor_area.loc[:, "sum"].sum()

        # create numpy arrays that will be filled inside the loop:
        temperature_weighted_sum = np.zeros((8760, 1))
        radiation_weighted_sum = np.zeros((8760, 4))
        PV_weighted_sum = np.zeros((8760, 1))

        # dictionary for different PV types
        for index, row in floor_area.iterrows():
            nuts_id = row["nuts_id"]
            floor_area_of_specific_region = row["sum"]
            # Temperature
            temperature_profile = pd.read_parquet(TEMP_DATA / f"Temperature_NUTS{nuts_id}_{country}.parquet.gzip").iloc[:, 0].to_numpy()
            temperature_weighted_sum += temperature_profile * floor_area_of_specific_region / floor_area_sum
            
            # Solar radiation
            radiation_profile = pd.read_parquet(TEMP_DATA / f"Radiation_NUTS{nuts_id}_{country}.parquet.gzip")
            radiation_weighted_sum += radiation_profile * floor_area_of_specific_region / floor_area_sum

            # PV
            # iterate through different PV types and add each weighted profiles to corresponding dictionary index
            PV_profile = pd.read_parquet(TEMP_DATA / f"PV_generation_NUTS{nuts_id}_{country}.parquet.gzip").iloc[:, 0].to_numpy()
            PV_weighted_sum += PV_profile * floor_area_of_specific_region / floor_area_sum

        # create table for saving to DB: Temperature and Radiation will be saved in the region table
        # Temperature + Radiation
        temperature_radiation_columns = {"ID_Region": np.full((8760,), country),
                                         "id_hour": self.id_hour,
                                         "south": radiation_weighted_sum[:, 0],
                                         "east": radiation_weighted_sum[:, 1],
                                         "west": radiation_weighted_sum[:, 2],
                                         "north": radiation_weighted_sum[:, 3],
                                         "temperature": temperature_weighted_sum.flatten(),
                                         "pv_generation_optimal": PV_weighted_sum,
                                         }

        temperature_radiation_table = pd.DataFrame(temperature_radiation_columns)
        temperature_radiation_table.to_csv(Path(__file__).parent / f"Weather_data_{country}_{year}.csv")
        # TODO save
        assert sorted(list(temperature_radiation_table.columns)) == sorted(list(input_data_structure.RegionData().__dict__.keys()))
        # save weighted temperature average profile to database:
        DB().write_dataframe(table_name=Table().region,
                             data_frame=temperature_radiation_table,
                             data_types=input_data_structure.RegionData().__dict__,
                             if_exists="replace"
                             )

        # PV
        pv_columns = input_data_structure.PVData().__dict__.keys()
        # create single row on which tables are stacked up
        pv_table_numpy = np.zeros((1, len(pv_columns)))
        for (id, peak_power), values in PV_weighted_sum.items():
            # create the table for each pv type
            single_pv_table = np.column_stack([np.full((8760,), country),  # nuts_id
                                               np.full((8760,), id),  # ID_PV
                                               self.id_hour,  # id_hour
                                               values,  # power
                                               np.full((8760,), "W"),  # unit
                                               np.full((8760,), peak_power),  # peak power
                                               np.full((8760,), "kWp")]  # peak power unit
                                              )
            # stack the tables from dictionary to one large table
            pv_table_numpy = np.vstack([pv_table_numpy, single_pv_table])

        # remove the first row (zeros) from the pv_table
        pv_table_numpy = pv_table_numpy[1:]
        pv_table = pd.DataFrame(pv_table_numpy, columns=pv_columns)
        # save to sql database:
        DB().write_dataframe(table_name=Table().pv_generation,
                             data_frame=pv_table,
                             data_types=input_data_structure.PVData().__dict__,
                             if_exists="replace"
                             )

        print(f"mean tables for country {country} have been saved")

    def get_nuts_id_list(self, nuts_level: int, country: str) -> list:
        absolut_path = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"NUTS{nuts_level}.json")
        all_nuts_ids = pd.read_json(absolut_path, orient="table")
        all_nuts_ids = all_nuts_ids[all_nuts_ids["country"] == country]["nuts_id"]
        return list(all_nuts_ids)

    def run(self,
            nuts_level: int,
            country_code: str,
            start_year: int,
            end_year: int,
        ):

        self.get_PVGIS_data(nuts_id_list=self.get_nuts_id_list(nuts_level, country_code),
                            country=country_code,
                            start_year=start_year,
                            end_year=end_year,
                        )
        # creating a single profile as mean profile for a country weighted by the floor area of each nuts region:
        self.calculate_single_profile_for_country(country=country_code, nuts_level=nuts_level, year=start_year)


if __name__ == "__main__":
    PVGIS().run(nuts_level=3, country_code="AT", start_year=2023, end_year=2023)
