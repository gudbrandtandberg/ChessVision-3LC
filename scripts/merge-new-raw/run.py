import tlc

new_table = tlc.Table.from_names(
    "add-17-masks",
    "chessvision-new-raw",
    "chessvision-new-raw",
)

filtered_table = tlc.FilteredTable(
    url=tlc.Url.create_table_url(f"filtered-{new_table.name}", new_table.dataset_name, new_table.project_name),
    input_table_url=new_table,
    filter_criterion=tlc.BoolFilterCriterion("mask", True),
)
filtered_table.write_to_url()

orig_table = tlc.Table.from_names(
    "train-cleaned-filtered",
    "chessboard-segmentation-train",
    "chessvision-segmentation",
)

joined_table = tlc.Table.join_tables(
    [orig_table, filtered_table],
    table_name=f"joined-{filtered_table.name}",
)

print(joined_table)
