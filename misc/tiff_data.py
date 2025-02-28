import json
from pathlib import Path

import tifffile

file_path = Path("/Users/natalieyeung/Downloads/Tyche-F2_2023_02_27_1__00001_00001.tif")
absolute_path = file_path.resolve()

with tifffile.TiffFile(absolute_path) as tiff:
    # metadata = {tag.name: tag.value for tag in tiff.pages[0].tags.values()}
    metadata = tiff.scanimage_metadata

print(metadata)

output_directory = Path("/Users/natalieyeung/Documents/GitHub/sl-mesoscope/misc")
metadata_json = output_directory / "metadata.json"

with open(metadata_json, "w") as json_file:
    json.dump(metadata, json_file, indent=4)
