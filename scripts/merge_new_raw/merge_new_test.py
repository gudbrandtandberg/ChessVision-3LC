import tlc

input_table = tlc.Table.from_names(
    table_name="select-16-test-samples",
    dataset_name="chessvision-new-raw",
    project_name="chessvision-new-raw",
)

filtered_table = tlc.FilteredTable(
    url=input_table.url.create_sibling("filtered"),
    filter_criterion=tlc.NumericRangeFilterCriterion(attribute="weight", min_value=1.1, max_value=3.0),
    input_table_url=input_table,
)
filtered_table.write_to_url()

edited_table = tlc.EditedTable(
    input_table_url=filtered_table,
    override_table_rows_schema={"values": {"weight": None, "mask": None}},
    url=filtered_table.url.create_sibling("remove-columns"),
)

edited_table.write_to_url()
