from datetime import datetime
from pathlib import Path
from pprint import pprint

from process_pipeline import run_pipeline

from chessvision.utils import DATA_ROOT

start_date = datetime.strptime("2024-11-03", "%Y-%m-%d").date()
end_date = datetime.strptime("2024-11-03", "%Y-%m-%d").date()

# Run pipeline
# results = run_pipeline(
#     start_date=start_date,
#     end_date=end_date,
#     skip_download=True,
#     skip_create_table=False,
#     skip_enrich=False,
#     dry_run=False,
# )

results = run_pipeline(
    start_date=start_date,
    end_date=end_date,
    output_folder=Path(DATA_ROOT) / "test" / "raw",
    skip_download=True,
    skip_create_table=True,
    skip_enrich=False,
    dry_run=False,
)

pprint(results, indent=4, width=120)
