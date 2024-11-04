import cartopy.crs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
from pathlib import Path
import hashlib
import base64
import rasterio
from shapely.geometry import box
import geopandas as gpd
from geopy.geocoders import Nominatim
import json
import cartopy
import shapely
import seaborn as sns
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.lines import Line2D
import tqdm
import time


PATH_2_INPUT_TIFS = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/input_tifs/")
CLASSIFIER_RESULTS = pd.read_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/") / "Classifier_Results.csv", sep=";")
with open(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis") / "OSM_IDs_lat_lon.json", "r") as f:
    BUILDING_LOCS = json.load(f)

VALENCIA_DATA = pd.read_csv(Path(__file__).parent / "datos-de-autoconsumo-de-energias-renovables.csv", sep=";", encoding="ISO-8859-1")
VALENCIA_DATA.loc[VALENCIA_DATA["CODIGO POSTAL"].isna(), "CODIGO POSTAL"] = VALENCIA_DATA.loc[VALENCIA_DATA["CODIGO POSTAL"].isna(), "CODIGO MUNICIPIO"]
VALENCIA_DATA.drop_duplicates(keep="first", inplace=True)

# PATH_2_INPUT_TIFS = Path(r"X:\projects4\workspace_philippm\building-stock-analysis\T3.1-dynamic-analysis\Case-study-II-III-PV-analysis\solar-panel-classifier\new_data\input_tifs")
# BUILDING_LOCS = {"1000248182": "39.345798,-0.571910"}



# make sure that this function is identical with the one used in the project!
def generate_hash(input_string, length=8):
    # Create a raw binary SHA256 hash
    hash_binary = hashlib.sha256(input_string.encode()).digest()
    # Base64 encode the hash and make it URL-safe
    hash_base64 = base64.urlsafe_b64encode(hash_binary).decode('utf-8')
    # Shorten the string before returning
    return hash_base64[:length]

def split_id_name(name: str):
    n = name.replace(".npy", "")
    osmid, hash = n.split("_")[1], n.split("_")[2]
    return osmid, hash


def get_input_tif_from_hash(hash: str) -> Path:
    input_tifs = [f for f in PATH_2_INPUT_TIFS.iterdir() if f.suffix == ".tif"]
    hashes = [generate_hash(f.name) for f in input_tifs]
    index = hashes.index(hash)
    return input_tifs[index]


def select_all_buildings_with_specific_hash(hash: str):
    return CLASSIFIER_RESULTS.loc[CLASSIFIER_RESULTS["OSM_ID"].str.contains(hash), :]


def find_adress_and_post_codes():
    identified = [name.split("_")[1] for name in CLASSIFIER_RESULTS.loc[CLASSIFIER_RESULTS["prediction"]==1, "OSM_ID"]]
    post_codes = {}
    villages = {}
    geolocator = Nominatim(user_agent="my_geocoder")

    for osmid in tqdm.tqdm(identified):
        lat_lon_str = BUILDING_LOCS[osmid]
        lat, lon = float(lat_lon_str.split(",")[0]), float(lat_lon_str.split(",")[1])

        location = geolocator.reverse((lat, lon), addressdetails=True)
        try:
            post_codes[osmid] = location.raw["address"]["postcode"]
        except KeyError as e:
            post_codes[osmid] = "NOCODE"
        try:
            villages[osmid] = location.raw["address"]["village"]
        except KeyError as e:
            villages[osmid] = "NOCODE"
        time.sleep(1)
    
    df = pd.DataFrame(villages, post_codes)


def number_of_PVs_per_region(gdf_3035) -> pd.DataFrame:
    path_rg = Path(r"nuts_json") / "NUTS_RG_01M_2021_3035_LEVL_2.json"
    gdf_rg = gpd.read_file(path_rg)
    valencia = gdf_rg[gdf_rg["NAME_LATN"]=="Comunitat Valenciana"].copy()

    path_rg3 = Path(r"nuts_json") / "NUTS_RG_01M_2021_3035_LEVL_3.json"
    gdf_rg3 = gpd.read_file(path_rg3)
    spain = gdf_rg3[gdf_rg3["CNTR_CODE"]=="ES"].copy()
    valencia3 = spain[spain.within(valencia.geometry.iloc[0])]

    number_per_region = {}
    for i, row in valencia3.iterrows():
        region = row["NAME_LATN"].split("/")[0]
        pv_number = 0
        if type(row.geometry) == shapely.geometry.multipolygon.MultiPolygon:
            for poly in row.geometry.geoms:
                PVs = gdf_3035[gdf_3035.within(poly)]
                pv_number += len(PVs)
        else:
            PVs = gdf_3035[gdf_3035.within(row.geometry.iloc[0])]
            pv_number += len(PVs)

        number_per_region[region] = pv_number

    
    valencia_numbers = VALENCIA_DATA.groupby("PROVINCIA").size().reset_index().rename(columns={0: "static data"})
    valencia_numbers["PROVINCIA"] = valencia_numbers["PROVINCIA"].apply(lambda x: x.lower().capitalize())
    for key, value in number_per_region.items():
        valencia_numbers.loc[valencia_numbers["PROVINCIA"]==key.replace("ó", "o"), "dynamic data"] = value
    

    valencia_numbers["identified number of PVs in (%)"] = valencia_numbers["dynamic data"] / valencia_numbers["static data"] * 100

    return valencia_numbers

def plot_identified_numbers(gdf_3035):
    numbers_df = number_of_PVs_per_region(gdf_3035)
    
    plot_df = numbers_df.melt(id_vars=["PROVINCIA", "identified number of PVs in (%)"], value_name="number", var_name="source")
    fig, ax = plt.subplots(figsize=(20, 15))#, subplot_kw={'projection': cartopy.crs.epsg(3035)})
    sns.barplot(
        data=plot_df,
        x="PROVINCIA",
        y="number",
        hue="source",
    )
    ax.set_ylabel("number of PV systems", fontsize=18)
    ax.set_xlabel("Provinces in Valencia", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "MODERATE_T3.4" / "Identified_Numbers.png", dpi=600)
    plt.close()


def plot_pvs_on_map(gdf_3035, gdf_3035_all):
    # load valencia boundaries:
    path_rg = Path(r"nuts_json") / "NUTS_RG_01M_2021_3035_LEVL_2.json"
    gdf_rg = gpd.read_file(path_rg)
    valencia = gdf_rg[gdf_rg["NAME_LATN"]=="Comunitat Valenciana"].copy()
    # plot the identified PVs on the map 
    fig, ax = plt.subplots(figsize=(20, 15))#, subplot_kw={'projection': cartopy.crs.epsg(3035)})
    ax = valencia.plot(color="lightgrey")
    gdf_3035_all.plot(ax=ax, color="blue", markersize=0.5, alpha=0.5)
    gdf_3035.plot(ax=ax, color="red", markersize=0.5, alpha=0.5)
    legend_elements = [
        Line2D([0], [0], color="white", marker="o", label="Buildings", markerfacecolor='blue'),
        Line2D([0], [0], marker="o", color="white", label="identified PV", markerfacecolor='red')
    ]
    ax.legend(handles=legend_elements, loc='lower right')#, bbox_to_anchor=(1.05, 1))
    plt.savefig(Path(__file__).parent / "MODERATE_T3.4" / "Valencia.png", dpi=1200)


def main():

    input_tifs = [f for f in PATH_2_INPUT_TIFS.iterdir() if f.suffix == ".tif"]
    hashes = [generate_hash(f.name) for f in input_tifs]
    regions = {}
    # for i, hash in enumerate(hashes):
    #     tif = input_tifs[i]
    #     df = select_all_buildings_with_specific_hash(hash)
        
    #     source = rasterio.open(tif)
    #     bounds = source.bounds
    #     polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    #     polygon_wgs84 = ox.projection.project_geometry(polygon, crs=source.crs, to_crs='EPSG:4326')[0]
    #     regions = ox.features_from_polygon(polygon=polygon_wgs84, tags={'boundary': 'administrative', 'admin_level': '6'})
    #     buildings = ox.features_from_polygon(polygon=polygon_wgs84, tags={'building': True})



    lats_pv = []
    lons_pv = []
    lats = []
    lons = []
    identified = [name.split("_")[1] for name in CLASSIFIER_RESULTS.loc[CLASSIFIER_RESULTS["prediction"]==1, "OSM_ID"]]
    for osmid, lat_lon_str in BUILDING_LOCS.items():
        lat, lon = float(lat_lon_str.split(",")[0]), float(lat_lon_str.split(",")[1])
        lats.append(lat)
        lons.append(lon)
        if osmid in identified:
            lats_pv.append(lat)
            lons_pv.append(lon)

    gdf_4326 = gpd.GeoDataFrame(
        {'latitude': lats_pv, 'longitude': lons_pv},
        geometry=[shapely.geometry.Point(lon, lat) for lon, lat in zip(lons_pv, lats_pv)],
        crs="EPSG:4326"  # Original coordinate reference system
    )
    gdf_3035 = gdf_4326.to_crs("EPSG:3035")

    # show how many PVs were identified in each NUTS3 region:
    plot_identified_numbers(gdf_3035=gdf_3035)


    gdf_4326_all = gpd.GeoDataFrame(
        {'latitude': lats, 'longitude': lons},
        geometry=[shapely.geometry.Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326"  # Original coordinate reference system
    )
    gdf_3035_all = gdf_4326_all.to_crs("EPSG:3035")

    plot_pvs_on_map(gdf_3035, gdf_3035_all)





if __name__ == "__main__":
    main()

