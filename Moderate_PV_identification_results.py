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

# PATH_2_INPUT_TIFS = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/input_tifs/")
# CLASSIFIER_RESULTS = pd.read_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/") / "Classifier_Results.csv", sep=";")
# BUILDING_LOCS = pd.read_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis") / "OSM_IDs_lat_lon.json", sep=";")

VALENCIA_DATA = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\building-stock-analysis\T3.1-dynamic-analysis\Case-study-II-III-PV-analysis\data") / "datos-de-autoconsumo-de-energias-renovables.csv", sep=";", encoding="ISO-8859-1")
VALENCIA_DATA.loc[VALENCIA_DATA["CODIGO POSTAL"].isna(), "CODIGO POSTAL"] = VALENCIA_DATA.loc[VALENCIA_DATA["CODIGO POSTAL"].isna(), "CODIGO MUNICIPIO"]

PATH_2_INPUT_TIFS = Path(r"X:\projects4\workspace_philippm\building-stock-analysis\T3.1-dynamic-analysis\Case-study-II-III-PV-analysis\solar-panel-classifier\new_data\input_tifs")
BUILDING_LOCS = {"1000248182": "39.345798,-0.571910"}



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



def main():
    input_tifs = [f for f in PATH_2_INPUT_TIFS.iterdir() if f.suffix == ".tif"]
    hashes = [generate_hash(f.name) for f in input_tifs]
    regions = {}
    for i, hash in enumerate(hashes):
        tif = input_tifs[i]
        df = select_all_buildings_with_specific_hash(hash)
        
        source = rasterio.open(tif)
        bounds = source.bounds
        polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        polygon_wgs84 = ox.projection.project_geometry(polygon, crs=source.crs, to_crs='EPSG:4326')[0]
        regions = ox.features_from_polygon(polygon=polygon_wgs84, tags={'boundary': 'administrative', 'admin_level': '6'})
        buildings = ox.features_from_polygon(polygon=polygon_wgs84, tags={'building': True})




    post_codes = {}
    villages = {}
    geolocator = Nominatim(user_agent="my_geocoder")
    for osmid, lat_lon_str in BUILDING_LOCS.items():
        lat, lon = float(lat_lon_str.split(",")[0]), float(lat_lon_str.split(",")[1])

        location = geolocator.reverse((lat, lon), addressdetails=True)
        post_codes[osmid] = location.raw["address"]["postcode"]
        villages[osmid] = location.raw["address"]["village"]
            

        



if __name__ == "__main__":
    main()

