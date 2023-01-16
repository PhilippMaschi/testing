import shutil
from pathlib import Path


def copy_files(source: Path, destination: Path):
    """Copy a file from src to dst, creating any missing directories in the destination path."""
    # Check if the destination file already exists
    for item in source.rglob("*.gzip"):
        subfolder = destination / item.parent.name
        if not subfolder.exists():
            subfolder.mkdir()
        shutil.copy(item, subfolder / item.name)

        print(f"copied {item.name} to local disk as {destination / item.parent.name}")


def delete_old_files(destination_folder):
    for item in destination_folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)


source_folder = Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\building_data")
destination_folder = Path(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\NewTrends_D5.4")
# delete all files in the new trends project folder
delete_old_files(destination_folder)
# copy new files to NewTrends
copy_files(source_folder, destination_folder)






