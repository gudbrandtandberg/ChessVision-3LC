import os
from collections import defaultdict

import plotly.graph_objects as go
import tlc

project_name = "chessvision-segmentation"

run_names = [
    "messy-fader",
    "partial-producer",
    "calm-equity",
    "amiable-buck",
    "deafening-lens",
    "investor-tanager",
]


def get_run_names(run_folder: str) -> list[str]:
    return [f.name for f in os.scandir(run_folder) if f.is_dir()]


run_names = get_run_names("c:/Users/gudbrand/AppData/Local/3LC/3LC/projects/chessvision-segmentation/runs")


def collect_sweep_data(run_names: list[str]) -> dict[str, list[float]]:
    data = defaultdict(list)
    for run_name in run_names:
        run = tlc.Run.from_url(tlc.Url.create_run_url(run_name, project_name))
        parameters = run.constants["parameters"]
        data["Learning Rate"].append(parameters["learning_rate"])
        data["Sample Weights"].append(float(parameters["use_sample_weights"]))
        data["Threshold"].append(parameters["threshold"])
        data["Accuracy"].append(parameters["best_val_score"])
        data["Test Accuracy"].append(float(parameters["test_results"]["top_1_accuracy"]))
        data["Run Name"].append(run_name)
    return data


data = collect_sweep_data(run_names)

# Create parallel coordinates plot
fig = go.Figure(
    data=go.Parcoords(
        line={
            "color": data["Test Accuracy"],  # Color lines by test accuracy instead
            "colorscale": "Viridis",  # Use Viridis colorscale
            "showscale": True,  # Show colorbar
        },
        dimensions=[
            {
                "range": [min(data["Learning Rate"]), max(data["Learning Rate"])],
                "label": "Learning Rate",
                "values": data["Learning Rate"],
                "ticktext": ["1e-7", "1e-6", "1e-5", "1e-4"],
                "tickvals": [1e-7, 1e-6, 1e-5, 1e-4],
            },
            {
                "range": [0, 1],
                "label": "Sample Weights",
                "values": data["Sample Weights"],
                "ticktext": ["Off", "On"],
                "tickvals": [0, 1],
            },
            {
                "range": [min(data["Threshold"]), max(data["Threshold"])],
                "label": "Threshold",
                "values": data["Threshold"],
                "ticktext": ["0.3", "0.5", "0.7"],
                "tickvals": [0.3, 0.5, 0.7],
            },
            {
                "range": [min(data["Accuracy"]), max(data["Accuracy"])],
                "label": "Val Accuracy",  # Renamed to clarify it's validation accuracy
                "values": data["Accuracy"],
                "tickformat": ".3f",
            },
            {
                "range": [min(data["Test Accuracy"]), max(data["Test Accuracy"])],
                "label": "Test Accuracy",
                "values": data["Test Accuracy"],
                "tickformat": ".3f",
            },
            {
                "range": [0, len(data["Run Name"])],
                "label": "Run",
                "values": list(range(len(data["Run Name"]))),
                "ticktext": data["Run Name"],
                "tickvals": list(range(len(data["Run Name"]))),
            },
        ],
    ),
)


# Update layout
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    title="Hyperparameter Sweep Results",
    width=1000,
    height=600,
)

# Show plot in browser
fig.show()  # Opens in default browser

# Save as HTML (interactive)
fig.write_html("hyperparameter_sweep.html")

# Save as PNG (static)
fig.write_image("hyperparameter_sweep.png")
