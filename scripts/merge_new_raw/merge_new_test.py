import tlc

from chessvision import constants

TEST_SET_DATASET_NAME = "test"
TEST_SET_PROJECT_NAME = "chessvision-testing"


def merge_new_test_batch(batch_folder_name: str) -> None:
    """Merges the new batch with the latest revision in the test set lineage.

    Assumes the new batch data has been prepared, and follows the naming
    convention of the existing test set batches.
    """

    input_table = tlc.Table.from_names(
        table_name="initial",
        dataset_name=TEST_SET_DATASET_NAME,
        project_name=TEST_SET_PROJECT_NAME,
    ).latest()

    print(f"Using latest revision of test dataset: {input_table.name}")

    new_data_folder = constants.DATA_ROOT / "test" / batch_folder_name / "raw"

    new_data_table = tlc.Table.from_image_folder(
        new_data_folder,
        include_label_column=False,
        dataset_name=TEST_SET_DATASET_NAME,
        table_name=new_data_folder.parent.name,
        project_name=TEST_SET_PROJECT_NAME,
        add_weight_column=False,
    )

    merged_table = tlc.Table.join_tables(
        [input_table, new_data_table],
        table_name=f"merged-{new_data_table.name}",
    )

    print(f"Merged table: {merged_table.name} ({len(merged_table)} images)")


if __name__ == "__main__":
    merge_new_test_batch("2024-11-04-2024-11-04")
