import tlc

new_table = tlc.Table.from_names(
    "raw",
    "chessvision-new-raw",
    "chessvision-new-raw",
)
orig_table = tlc.Table.from_names(
    "fix-bad-sample",
    "chessboard-segmentation-train",
    "chessvision-segmentation",
)

new_table = tlc.Table.join_tables([orig_table, new_table], table_name=f"joined-{new_table.name}")
