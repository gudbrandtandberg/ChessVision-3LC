from collections import defaultdict

import plotly.graph_objects as go
import tlc

project_name = "chessvision-segmentation"


def collect_sweep_data(runs: list[str]) -> dict[str, list[float]]:
    data = defaultdict(list)
    for run_name in runs:
        run = tlc.Run.from_url(tlc.Url.create_run_url(run_name, project_name))
        parameters = run.constants["parameters"]
        data["Learning Rate"].append(parameters["learning_rate"])
        data["Sample Weights"].append(float(parameters["use_sample_weights"]))
        data["Accuracy"].append(parameters["best_val_score"])
        data["Run Name"].append(run_name)
    return data


runs = [
    "messy-fader",
    "partial-producer",
    "calm-equity",
    "amiable-buck",
    "deafening-lens",
    "investor-tanager",
]

data = collect_sweep_data(runs)

# Create parallel coordinates plot
fig = go.Figure(
    data=go.Parcoords(
        line={
            "color": data["Accuracy"],  # Color lines by accuracy
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
                "range": [min(data["Accuracy"]), max(data["Accuracy"])],
                "label": "Accuracy",
                "values": data["Accuracy"],
                "tickformat": ".3f",  # 3 decimal places
            },
            # Add run names as a dimension
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
    plot_bgcolor="white", paper_bgcolor="white", title="Hyperparameter Sweep Results", width=1000, height=600
)

# Show plot in browser
fig.show()  # Opens in default browser

# Save as HTML (interactive)
fig.write_html("hyperparameter_sweep.html")

# Save as PNG (static)
fig.write_image("hyperparameter_sweep.png")
