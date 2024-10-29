import shutil
from pathlib import Path


def is_descendant_of(path, potential_ancestor):
    """
    Checks if the given path is a descendant of the potential ancestor.

    :param path: The path to check.
    :param potential_ancestor: The path that might be an ancestor.
    :return: True if 'path' is a descendant of 'potential_ancestor', False otherwise.
    """
    # Compare parts of the paths (each part represents a directory level)
    return path.parts[:len(potential_ancestor.parts)] == potential_ancestor.parts


def copy_files_and_folders(source: Path, destination: Path, file_type: list=None, file_name_contains: str=None):
    """
    Copies files  from the source directory to the destination directory.
    If file_type is specified, only files of that type are copied within any folder. Also files from sub directories
    are copied to the destination path.

    :param source: Path to the source directory
    :param destination: Path to the destination directory
    :param file_type: Optional. Extension of the files to copy (e.g., 'txt' for text files). provide as list
    Default is None, which copies all files.
    """

    if destination == source or is_descendant_of(destination, source):
        raise ValueError("Destination path must not be the same as or a subdirectory of the source path.")
    # Check and create the destination directory if it does not exist
    if not destination.exists():
        print("destination does not exist yet, creating the folder")
        destination.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        if item.is_dir():
            # Recursively copy the folder and its content
            copy_files_and_folders(item, destination, file_type, file_name_contains)
        elif item.is_file():
            # if "DEU_2050" in item.stem:
            #     continue
            # Copy file if no file_type is specified or if it matches the file_type
            type_match = file_type is None or item.suffix[1:] in file_type
            name_match = file_name_contains is None or file_name_contains in item.name
            if type_match and name_match:
                # check if item already exists in
                if not (destination / f"{item.stem}{item.suffix}").exists():
                    # Copy file if it matches the file_type and/or file_name_contains criteria
                    shutil.copy(item, destination)
                    print(f"copied {item}")



if __name__ == "__main__":
    source_path = Path(r"E:\projects\Philipp\FLEX_public\projects")
    destination_path = Path(r"C:\Users\mascherbauer\OneDrive\TU\for songmin")
    copy_files_and_folders(
        source=source_path,
        destination=destination_path,
        file_type=["sqlite"],  # sqlite  csv
        file_name_contains="Summary"#"Summary", None
    )