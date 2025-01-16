import pandas as pd
from pathlib import Path


source_path = Path(r"C:\Users\mascherbauer\OneDrive\PycharmProjects\FLEX\projects\AURESII_DATA\AURES2 curtailment+demand\leviathan")

files = []
for file_path in source_path.iterdir():
    files.append(file_path.name.replace(".csv", "").split("_")[-1])
countries = list(set([f for f in files if len(f)>1]))

dfs = []
for country in countries:
    for year in [2030, 2040, 2050]:
        file_name = f"leviathan_dem_{year}_{country}.csv"
        df = pd.read_csv(source_path / file_name)
        if "ENDOGENOUS" in df.columns:
            df["demand"] = df["EXOGENOUS"] + df["ENDOGENOUS"]
        else:
            df["demand"] = df["EXOGENOUS"]
        df.rename(columns={"RRR":"country"}, inplace=True)
        df["year"] = year
        dfs.append(df[["country", "demand", "UNITS", "year"]])

big_df = pd.concat(dfs)
big_df.to_csv(Path(r"C:\Users\mascherbauer\OneDrive\PycharmProjects\FLEX\projects\AURESII_DATA") / "AURES_leviathan_demand.csv", sep=",", index=False)


