import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import plotly.express as px


def invert_hwb():
    path_to_projects = Path(r"X:\projects4\workspace_philippm\FLEX\projects")

    projects = [p for p in path_to_projects.glob("*") if p.is_dir() and len(p.name) == 8 and not "analysis" in p.name]
    files = [f / "input" / f"INVERT_{f.name}.csv" for f in projects]

    dfs = []
    for file in tqdm(files):
        df = pd.read_csv(file)
        avg_hwb = ((df["hwb"] * df["number_buildings_heat_pump_air"] + df["hwb"] * df["number_buildings_heat_pump_ground"]) / (df["number_buildings_heat_pump_air"].sum() + df["number_buildings_heat_pump_ground"].sum())).sum()

        df = pd.DataFrame(data={"country": [file.name.split("_")[1]], "year": [file.name.split("_")[-1].replace(".csv", "")], "avg_hwb": [avg_hwb]})
        dfs.append(df)

    df = pd.concat(dfs)

    order = df.groupby("country")["avg_hwb"].mean().sort_values(ascending=True).index
    sns.barplot(
        data=df, 
        x="country", 
        y="avg_hwb", 
        hue="year",
        palette=sns.color_palette(),

        order=order,
    )
    plt.xticks(rotation=90)
    plt.ylabel("average heating demand for HP heated buildings (kWh/m²a)")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "hwb_Invert_HP_heated_buildings.svg")
    plt.show()


    df.reset_index(drop=True).to_csv(Path(r"X:\projects4\workspace_philippm\testing\Country_level_prosumaging") / "hwb_Invert_HP_heated_buildings.csv", index=False)


def compare_invert_outside_temp_with_flex_outside_temp():
    path_to_flex_outside_temp = Path(r"X:\projects4\workspace_philippm\FLEX\projects")

    projects = [p for p in path_to_flex_outside_temp.glob("*") if p.is_dir() and len(p.name) == 8 and not "analysis" in p.name]
    files = [f / "input" / f"OperationScenario_RegionWeather.csv" for f in projects]
    flex_dfs = []
    for file in files:
        df = pd.read_csv(file)
        outside_temp = df[["id_hour", "temperature"]]
        outside_temp["id_hour"] = pd.date_range(start="2020-01-01", periods=8760, freq="H")
        outside_temp.loc[:, "month"] = outside_temp["id_hour"].dt.month
        monthly_mean = outside_temp.groupby("month")["temperature"].mean().reset_index().T
        monthly_mean.columns = monthly_mean.iloc[0]
        monthly_mean = monthly_mean.iloc[1:]
        monthly_mean.loc[:, "country"] = file.parent.parent.name.split("_")[0]
        monthly_mean.loc[:, "year"] = file.parent.parent.name.split("_")[-1].replace(".csv", "")
        flex_dfs.append(monthly_mean)

    flex_df = pd.concat(flex_dfs, axis=0)
    flex_df["model"] = "flex"
    flex_df.reset_index(drop=True, inplace=True)


    path_to_invert_outside_temp = Path(r"X:\projects3\2021_ECEMF\invert\input\input_ecemf_invert_eelab_secondround_231115")
    projects = [p for p in path_to_invert_outside_temp.glob("*") if p.is_dir() and len(p.name) == 3]
    invert_dfs = []

    for p in projects:
        file = [f for f in (p / r"_BASE_DATA_ALL_SCENARIOS\_sub_scenarios\_BASE_").glob("*csv") if "climate_region" in f.name][0]
        df = pd.read_csv(file)

        columns = [c for c in df.columns if "mean_temp" in c]
        df = df[columns].dropna().reset_index(drop=True)
        outside_temp = df.iloc[0, :].reset_index().T
        outside_temp.columns = outside_temp.iloc[0]
        outside_temp = outside_temp.iloc[1:]
        outside_temp.columns = [x for x in np.arange(1, 13)]

        outside_temp.loc[:, "country"] = file.parent.parent.parent.parent.name
        outside_temp.loc[:, "year"] = 2020
        invert_dfs.append(outside_temp)


    invert_df = pd.concat(invert_dfs, axis=0)
    invert_df["model"] = "invert"

    c_df = pd.concat([flex_df.loc[flex_df["year"] == "2020"], invert_df], axis=0)
    plot_df = c_df.melt(id_vars=["country", "year", "model"], var_name="month", value_name="temperature")

    fig = px.line(plot_df, 
                  x="month", 
                  y="temperature", 
                  color="country", 
                  line_dash="model",
                  )
    fig.show()













compare_invert_outside_temp_with_flex_outside_temp()



# def format_references():
#     import bibtexparser
#     from bibtexparser.bwriter import BibTexWriter
#     from bibtexparser.bibdatabase import BibDatabase

#     # Load your .bib file
#     with open(Path(r"C:\Users\mascherbauer\Downloads") / "References.bib") as bibtex_file:
#         bib_database = bibtexparser.load(bibtex_file)


#     # Function to format a single entry
#     def format_entry(entry):
#         authors = entry.get('author', '').replace(' and ', ', ')
#         title = entry.get('title', '')
#         journal = entry.get('journal', '')
#         volume = entry.get('volume', '')
#         pages = entry.get('pages', '')
#         year = entry.get('year', '')
#         doi = entry.get('doi', '')
#         url = entry.get('url', '')

#         formatted_entry = f"\\bibitem{{{entry['ID']}}} {authors}, ``{title},'' {journal}, vol. {volume}, pp. {pages}, {year}."
#         if doi:
#             formatted_entry += f" DOI: {doi}"
#         if url:
#             formatted_entry += f" URL: {url}"
#         return formatted_entry

#     # Generate the thebibliography environment
#     bibliography = "\\begin{thebibliography}{00}\n"
#     for entry in bib_database.entries:
#         bibliography += format_entry(entry) + "\n"
#     bibliography += "\\end{thebibliography}"

#     # Save to a .tex file
#     with open(Path(r"C:\Users\mascherbauer\Downloads") / "formatted_references.tex", 'w') as f:
#         f.write(bibliography)


#     print("Formatted references saved to formatted_references.tex")

# format_references()
