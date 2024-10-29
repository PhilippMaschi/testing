import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

path_to_folder = Path(r"X:\projects4\workspace_philippm\FLEX\projects\ECEMF_T4.3_Leeuwarden\data_output")

file_2020 = "2020_Leeuwarden_H_PV-5%_DHW-50%_Buffer-0%_HE-0%_AirHP-10%_GroundHP-0%_directE-5%_Conventional-80%_AC-5%_Battery-10%_Prosumager-0%.csv"
file_2030 = "2020_Leeuwarden_H_PV-10%_DHW-50%_Buffer-0%_HE-0%_AirHP-10%_GroundHP-0%_directE-5%_Conventional-80%_AC-5%_Battery-10%_Prosumager-0%.csv"

moderate_df = pd.read_csv(path_to_folder / file_2030, sep=";")
high_df = pd.read_csv(path_to_folder / file_2020, sep=";")

assert moderate_df.columns.all() == high_df.columns.all(), "columns are not the same"

ordered_columns = sorted(list(moderate_df.iloc[:, 3:].columns))

diff = moderate_df.iloc[:24, 3:][ordered_columns] - high_df.iloc[:24, 3:][ordered_columns]


not_zero = diff.loc[:, (diff != 0).any(axis=0)].copy()
# Buildings IDs that are not zero meaning that the buildings changed:
z_columns = list(not_zero.columns.astype(int))

# load the start

high_eff_building_coordinates = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") / "high_eff_Building_coordinates_Leeuwarden.csv")
buildings = high_eff_building_coordinates.loc[high_eff_building_coordinates.loc[:, "ID_Building"].isin(z_columns), :]

moderate_df.iloc[0:96, 3:].reset_index(drop=True) - moderate_df.iloc[96:, 3:].reset_index(drop=True)

moderate_df.iloc[0:96, 3:].reset_index(drop=True) - high_df.iloc[0:96, 3:].reset_index(drop=True)

changed_buildings = gpd.GeoDataFrame(buildings, geometry='location', crs="epsg:3035")
changed_buildings.to_file(f"different_buildings.shp", driver="ESRI Shapefile")


not_zero.shape
diff.plot()
plt.show()

not_zero.plot()
plt.show()


