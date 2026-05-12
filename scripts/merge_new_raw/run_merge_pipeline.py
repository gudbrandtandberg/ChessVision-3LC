import tlc

# Filter criteria are private/dashboard-driven in 3.x; reach into private paths here.
from tlc._core.objects.tables.from_table.filtered_table import FilteredTable
from tlc._core.objects.tables.from_table.filtered_table_criteria.bool_filter_criterion import BoolFilterCriterion

new_table = tlc.Table.from_names(
    project_name="chessvision-new-raw",
    dataset_name="chessvision-new-raw",
    table_name="add-17-masks",
)

filtered_table = FilteredTable(
    url=tlc.helpers.ProjectLayout.table_url(
        project_name=new_table.project_name,
        dataset_name=new_table.dataset_name,
        table_name=f"filtered-{new_table.name}",
    ),
    input_table_url=new_table,
    filter_criterion=BoolFilterCriterion("mask", True),
)
filtered_table.write_to_url()

orig_table = tlc.Table.from_names(
    project_name="chessvision-segmentation",
    dataset_name="chessboard-segmentation-train",
    table_name="train-cleaned-filtered",
)

joined_table = tlc.Table.join_tables(
    [orig_table, filtered_table],
    table_name=f"joined-{filtered_table.name}",
)

print(joined_table)
