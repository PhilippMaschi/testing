import shutil
from pathlib import Path
import pandas as pd



def find_total_energy_demand_result(directory: Path):
    for file in directory.rglob('001_ec_fed_total_energy_demand.csv'):
        return file
    return None


def copy_file_to_disc_and_rename(source: Path, destination: Path, country_name: str):
    """Copy a file from src to dst, creating any missing directories in the destination path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    new_filename = country_name + '_' + source.name
    full_destination = destination / new_filename
    shutil.copy(source, full_destination)
    print(f"copied {source.name} to local disk as {destination}")


def main(path_dict: dict, output_folder: Path):
    country_list = [
        'AUT',
        'BEL',
        'BGR',
        'HRV',
        'CYP',
        'CZE',
        'DNK',
        'EST',
        'FIN',
        'FRA',
        'DEU',
        'GRC',
        'HUN',
        'IRL',
        'ITA',
        'LVA',
        'LTU',
        'LUX',
        'MLT',
        'NLD',
        'POL',
        'PRT',
        'ROU',
        'SVK',
        'SVN',
        'ESP',
        'SWE'
    ]

    for country in country_list:
        source_directory = path_dict["SOURCE_PATH"] / path_dict[
            "INVERT_SCENARIO"] / country / f"_scen_{country.lower()}_{path_dict['SUB_SCENARIO']}"

        energy_file = find_total_energy_demand_result(directory=source_directory)
        copy_file_to_disc_and_rename(source=energy_file,
                                     destination=output_folder,
                                     country_name=country)


def create_dict_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir()

if __name__ == "__main__":
    # user inputs:
    project_path = Path(r"E:/projects3/2021_ECEMF/invert")  # Path(r"E:\projects3\2022_NewTrends\invert")
    invert_scenario = r"output/output_ecemf_invert_eelab_secondround_231130_am_pm"  # r"output_new_trends_2022_12_20_2050"
    sub_scenario = "eff_high_elec_ab"  # "_res_hc_pw_alternative_1_ab"   # always has _sce_country before
    # invert input data ("input") folder
    invert_input_path = Path(
        r"W:\projects3\2021_ECEMF\invert\input\input_ecemf_invert_eelab_secondround_231115")
    # define path where data should be saved
    output_folder = Path(r"E:/projects3/2021_ECEMF/Philipp")
    years = [2020, 2030, 2040, 2050]

    paths = {"SOURCE_PATH": project_path,
             "INVERT_SCENARIO": invert_scenario,
             "SUB_SCENARIO": sub_scenario,
             "INVERT_INPUT": invert_input_path}

    create_dict_if_not_exists(output_folder)
    main(paths, output_folder)

