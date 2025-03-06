from datetime import datetime
from pprint import pprint

from process_pipeline import run_pipeline

start_date = datetime.strptime("2024-11-03", "%Y-%m-%d").date()
end_date = datetime.strptime("2024-11-03", "%Y-%m-%d").date()

# Run pipeline
results = run_pipeline(
    start_date=start_date,
    end_date=end_date,
    skip_download=False,
    skip_create_table=False,
    skip_enrich=False,
    dry_run=False,
)

pprint(results, indent=4, width=120)
