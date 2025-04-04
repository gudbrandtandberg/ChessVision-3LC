"""
The ChessVision developer webpage.
Allows upload to either a live instance, a local container, or a local webserver.
Will not include any analysis, but will contain logic for editing results.
"""

import argparse
import sys
import warnings

from flask import Flask, render_template

app = Flask(__name__)

endpoint = None


@app.route("/")
def home() -> str:
    return render_template("index.html", endpoint=endpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=str)
    args = parser.parse_args()
    if not args.local:
        endpoint = "lambda"
    elif args.local == "container":
        endpoint = "container"
    elif args.local == "server":
        endpoint = "server"
    else:
        warnings.warn("--local flag must be either 'container' or 'server'", stacklevel=2)
        sys.exit(1)

    app.run(host="127.0.0.1", port=5000)
