# import cartopy.crs
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
# import cartopy
import shapely
import seaborn as sns
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.lines import Line2D
import tqdm
import time
from PIL import Image
import shutil

# labels = pd.read_csv(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\data\VITO") / "labels_P6269_elec.csv", sep=";")
# data = pd.read_csv(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\data\VITO") / "P6269_1_50_DMK_Sample_Elek_Volume_Afname_kWh_resampled.csv", sep=",")

# df_heatpumps = pd.DataFrame()
# df_no = pd.DataFrame()

# for i, row in labels.iterrows():
#     if row["Heatpump"] == 0:
#         df_no.loc[:, str(row["ID"])] = data.loc[:, str(row["ID"])].copy()
#     else:
#         df_heatpumps.loc[:, str(row["ID"])] = data.loc[:, str(row["ID"])].copy()
      
# df_no.to_csv(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\data\VITO") / "NO_heatpumps.csv", sep=";", index=False)
# df_heatpumps.to_csv(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\data\VITO") / "Heatpumps.csv", sep=";", index=False)


# BOZEN
# BOZEN_CLASSIFIER_RESULTS = pd.read_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Bozen/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/") / "Classifier_Results.csv", sep=";")
BOZEN_CLASSIFIER_RESULTS = pd.read_csv(Path(r"X:\projects4\workspace_philippm\Bozen\building-stock-analysis\T3.4-PV-identification\results") / "Classifier_Results.csv", sep=";")

# with open(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Bozen/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis") / "OSM_IDs_lat_lon.json", "r") as f:
#     BOZEN_BUILDING_LOCS = json.load(f)
    
with open(Path(r"X:\projects4\workspace_philippm\Bozen\building-stock-analysis\T3.4-PV-identification\results") / "OSM_IDs_lat_lon.json", "r") as f:
    BOZEN_BUILDING_LOCS = json.load(f)

# BOZEN_MANUALLY_IDENTIFIED = pd.read_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Bozen/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/") / "OSM_IDs_labeled.csv", sep=";")
BOZEN_MANUALLY_IDENTIFIED = pd.read_csv(Path(r"X:\projects4\workspace_philippm\Bozen\building-stock-analysis\T3.4-PV-identification\results") / "OSM_IDs_labeled.csv", sep=";")


# Valencia
# PATH_2_INPUT_TIFS = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/input_tifs/")
PATH_2_INPUT_TIFS = Path(r"X:\projects4\workspace_philippm\building-stock-analysis\T3.1-dynamic-analysis\Case-study-II-III-PV-analysis\solar-panel-classifier\new_data\input_tifs")

# CLASSIFIER_RESULTS = pd.read_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/") / "Classifier_Results.csv", sep=";")
CLASSIFIER_RESULTS = pd.read_csv(Path(r"X:\projects4\workspace_philippm\building-stock-analysis\T3.1-dynamic-analysis\Case-study-II-III-PV-analysis\solar-panel-classifier\new_data") / "Classifier_Results.csv", sep=";")

# with open(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis") / "OSM_IDs_lat_lon.json", "r") as f:
#     BUILDING_LOCS = json.load(f)

# VALENCIA_DATA = pd.read_csv(Path(__file__).parent / "datos-de-autoconsumo-de-energias-renovables.csv", sep=";", encoding="ISO-8859-1")
# VALENCIA_DATA.loc[VALENCIA_DATA["CODIGO POSTAL"].isna(), "CODIGO POSTAL"] = VALENCIA_DATA.loc[VALENCIA_DATA["CODIGO POSTAL"].isna(), "CODIGO MUNICIPIO"]
# VALENCIA_DATA.drop_duplicates(keep="first", inplace=True)
# VALENCIA_DATA = VALENCIA_DATA.loc[VALENCIA_DATA["COMBUSTIBLE"].isna(), :].copy()


# MANUALLY_IDENTIFIED = pd.read_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/") / "OSM_IDs_labeled.csv", sep=";")


df = pd.read_csv(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\data\PV_classifier_results") / "Valencia_OSM_ID_with_PV_classification.csv", sep=";")
df["OSM_ID"] = df["OSM_ID"].apply(lambda x: f'{x.split("_")[0]}_{x.split("_")[1]}')

df.to_csv(Path(r"C:\Users\mascherbauer\OneDrive\EEG_Projekte\MODERATE\data\PV_classifier_results") / "Valencia_OSM_ID_with_PV_classification.csv", sep=";", index=False)


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


def get_input_tif_from_hash(hash: str, ) -> Path:
    input_tifs = [f for f in PATH_2_INPUT_TIFS.iterdir() if f.suffix == ".tif"]
    hashes = [generate_hash(f.name) for f in input_tifs]
    index = hashes.index(hash)
    return input_tifs[index]


def select_all_buildings_with_specific_hash(classifier_result: str):
    return classifier_result.loc[classifier_result["OSM_ID"].str.contains(hash), :]


def find_adress_and_post_codes(classifier_result, building_locs):
    identified = [name.split("_")[1] for name in classifier_result.loc[classifier_result["prediction"]==1, "OSM_ID"]]
    post_codes = {}
    villages = {}
    geolocator = Nominatim(user_agent="my_geocoder")

    for osmid in tqdm.tqdm(identified):
        lat_lon_str = building_locs[osmid]
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


def number_of_PVs_per_region_valencia(gdf_3035) -> pd.DataFrame:
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
    numbers_df = number_of_PVs_per_region_valencia(gdf_3035)
    
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

def load_LAU_data():
    path2data = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/testing/MODERATE_T3.4/") / "LAU_RG_01M_2021_3035.shp"
    gdf = gpd.read_file(path2data)
    spain_lau = gdf[gdf["CNTR_CODE"]=="ES"]


    VALENCIA_DATA.drop(columns=["FECHA RECHAZO", "OBSERVACIONES RECHAZO", "FECHA INSCRIPCIÃN", "MODALIDAD", "EMPRESA DISTRIBUIDORA",
                                "TENSIÃN PUNTO CONEXIÃN", "INDIVIDUAL/COLECTIVO", "NÃMERO CONSUMIDORES", "EMPRESA DISTRIBUIDORA INST", "ESQUEMA MEDIDA", 
                                "FECHA COMUNICACIÃN DISTRIBUIDORA", "CÃDIGO RADNE", "SUBTIPO TECNOLOGÃA", "TIPO INSTALACIÃN", "TENSIÃN GENERACIÃN",
                                "SSAA", "CÃDIGO PRETOR", "CÃDIGO RAIPRE", "CÃDIGO ALMACENAMIENTO RADNE", "ENERGÃA MÃX ALMACENABLE",
                                "CÃDIGO REGISTRO AUTONÃMICO", "COMPENSACIÃN", "POBLACIÃN INST", "ESTADO", "USO", "SUELO URBANO", "POTENCIA SALIDA", "POBLACIÃN"], inplace=True)

    pass


def plot_pvs_on_map(gdf_3035, gdf_3035_all, region: str):
    # load valencia boundaries:
    path_rg = Path(r"nuts_json") / "NUTS_RG_01M_2021_3035_LEVL_2.json"
    gdf_rg = gpd.read_file(path_rg)
    if region == "valencia":
        nuts3_region = gdf_rg[gdf_rg["NAME_LATN"]=="Comunitat Valenciana"].copy()

    else:
        nuts3_region = gdf_rg[gdf_rg["NAME_LATN"]=="Provincia Autonoma di Bolzano/Bozen"].copy()


    # plot the identified PVs on the map 
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 15))#, subplot_kw={'projection': cartopy.crs.epsg(3035)})
    nuts3_region.plot(ax=axes[0], color="lightgrey")
    nuts3_region.plot(ax=axes[1], color="lightgrey")
    gdf_3035_all.plot(ax=axes[1], color="blue", markersize=0.5, alpha=0.5)
    gdf_3035.plot(ax=axes[0], color="red", markersize=0.5, alpha=0.5)
    legend_elements = [
        Line2D([0], [0], color="white", marker="o", label="Buildings", markerfacecolor='blue'),
        Line2D([0], [0], marker="o", color="white", label="identified PV", markerfacecolor='red')
    ]
    fig.legend(handles=legend_elements, loc='lower right',  prop={'size': 20})
    for ax in axes:
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "MODERATE_T3.4" / f"{region}.png", dpi=600)
    plt.close()


def main():

    # input_tifs = [f for f in PATH_2_INPUT_TIFS.iterdir() if f.suffix == ".tif"]
    # hashes = [generate_hash(f.name) for f in input_tifs]
    # regions = {}
    # for i, hash in enumerate(hashes):
    #     tif = input_tifs[i]
    #     df = select_all_buildings_with_specific_hash(hash)
        
    #     source = rasterio.open(tif)
    #     bounds = source.bounds
    #     polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    #     polygon_wgs84 = ox.projection.project_geometry(polygon, crs=source.crs, to_crs='EPSG:4326')[0]
    #     regions = ox.features_from_polygon(polygon=polygon_wgs84, tags={'boundary': 'administrative', 'admin_level': '6'})
    #     buildings = ox.features_from_polygon(polygon=polygon_wgs84, tags={'building': True})

    # hashs = [generate_hash(file_name) for file_name in [
    #         "020201_2023CVAL0025_25830_8bits_RGBI_0893_1-4.tif"
    #         "020201_2023CVAL0025_25830_8bits_RGBI_0893_1-5.tif",
    #         "020201_2023CVAL0025_25830_8bits_RGBI_0893_1-6.tif",
    #         "020201_2023CVAL0025_25830_8bits_RGBI_0893_2-4.tif",
    #         "020201_2023CVAL0025_25830_8bits_RGBI_0893_2-5.tif",
    #         "020201_2023CVAL0025_25830_8bits_RGBI_0893_2-6.tif",
    #     ]
    # ]


    # find the manually identified npy file and save it as png to check if its the same and correct:
    # for m in manually_identified:
    #     name = f"building_{m}.npy"
    #     file = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/processed/")
    #     file_name = file / name
    #     image_array = np.load(file_name)
    #     img = np.moveaxis(image_array, 0, -1) 
    #     img = Image.fromarray(img.astype('uint8'))
    #     img.save(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/testing/checkimg/") / f"{name.replace('.npy', '')}.png")



    identified = [name.replace("building_", "") for name in CLASSIFIER_RESULTS.loc[CLASSIFIER_RESULTS["prediction"]==1, "OSM_ID"]]
    identified = [i.split("_")[0] for i in identified]

    all_oms = [name.replace("building_", "") for name in CLASSIFIER_RESULTS.loc[:, "OSM_ID"]]
    # all_hashs = ["".join(name.replace("building_", "").split("_")[1:]) for name in CLASSIFIER_RESULTS.loc[:, "OSM_ID"]]

    # wrong = []
    # for i in MANUALLY_IDENTIFIED.loc[MANUALLY_IDENTIFIED["has_pv"]=="yes", "osmid"]:
    #     if str(i) not in identified:
    #         wrong.append(i)
    #     if str(i) not in all_oms:
    #         print("FUCK")
    


    lats_pv = []
    lons_pv = []
    lats = []
    lons = []
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

    plot_pvs_on_map(gdf_3035, gdf_3035_all, region="valencia")

def main_bozen():


    identified = [name.replace("building_", "") for name in BOZEN_CLASSIFIER_RESULTS.loc[BOZEN_CLASSIFIER_RESULTS["prediction"]==1, "OSM_ID"]]
    all_oms = [name.replace("building_", "") for name in BOZEN_CLASSIFIER_RESULTS.loc[:, "OSM_ID"]]
    # all_hashs = ["".join(name.replace("building_", "").split("_")[1:]) for name in CLASSIFIER_RESULTS.loc[:, "OSM_ID"]]

    # wrong = []
    # for i in MANUALLY_IDENTIFIED.loc[MANUALLY_IDENTIFIED["has_pv"]=="yes", "osmid"]:
    #     if str(i) not in identified:
    #         wrong.append(i)
    #     if str(i) not in all_oms:
    #         print("FUCK")
    


    lats_pv = []
    lons_pv = []
    lats = []
    lons = []
    for osmid, lat_lon_str in BOZEN_BUILDING_LOCS.items():
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
    # plot_identified_numbers(gdf_3035=gdf_3035)


    gdf_4326_all = gpd.GeoDataFrame(
        {'latitude': lats, 'longitude': lons},
        geometry=[shapely.geometry.Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326"  # Original coordinate reference system
    )
    gdf_3035_all = gdf_4326_all.to_crs("EPSG:3035")

    plot_pvs_on_map(gdf_3035, gdf_3035_all, region="bozen")



if __name__ == "__main__":
    # main()
    main_bozen()

