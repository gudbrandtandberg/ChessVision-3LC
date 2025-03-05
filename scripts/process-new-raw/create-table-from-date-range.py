#!/usr/bin/env python

from pathlib import Path

import tlc
import tqdm
from PIL import Image

data_dir = Path("../../output/nov-1-2-2024").absolute()
assert data_dir.exists()

table_writer = tlc.TableWriter(
    table_name="raw_data_nov_1_2_2024",
    dataset_name="chessvision-new-raw",
    project_name="chessvision-new-raw",
    column_schemas={
        "image": tlc.PILImage("image"),
    },
    if_exists="overwrite",
)

for file in tqdm.tqdm(list(data_dir.iterdir())):
    if not file.is_file() or file.suffix != ".JPG":
        assert False

    pil_img = Image.open(file)
    table_writer.add_row({"image": pil_img})
    break

table = table_writer.finalize()
print(table.url)
