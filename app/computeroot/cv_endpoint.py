import argparse
import base64
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from functools import update_wrapper
from pathlib import Path

import chess
import cv2
import flask
import numpy as np
from flask import Flask, current_app, make_response, request
from werkzeug.utils import secure_filename

from chessvision.core import ChessVision

# Create necessary directories
app_root = Path(__file__).parent
uploads_folder = app_root / "user_uploads"
tmp_folder = app_root / "tmp"
logs_folder = app_root / "logs"

for folder in [uploads_folder, tmp_folder, logs_folder]:
    folder.mkdir(exist_ok=True, parents=True)

for subfolder in ["boards", "raw", "squares"]:
    (uploads_folder / subfolder).mkdir(exist_ok=True, parents=True)

# Create square type folders
piece_types = ["B", "_b", "N", "_n", "R", "_r", "Q", "_q", "K", "_k", "f", "P", "_p"]
for piece_type in piece_types:
    (uploads_folder / "squares" / piece_type).mkdir(exist_ok=True, parents=True)


# Set up logging
class RequestFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            record.url = request.url
            record.remote_addr = request.remote_addr
        except:
            record.url = "-"
            record.remote_addr = "-"
        return super().format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = RequestFormatter(
    "[%(asctime)s] %(remote_addr)s requested %(url)s\n%(levelname)s in %(module)s: %(message)s"
)

if not os.getenv("LOCAL"):
    log_file = logs_folder / "cv_endpoint.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def crossdomain(origin=None, methods=None, headers=None, max_age=21600, attach_to_all=True, automatic_options=True):
    if methods is not None:
        methods = ", ".join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ", ".join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ", ".join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers["allow"]

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == "OPTIONS":
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != "OPTIONS":
                return resp

            h = resp.headers

            h["Access-Control-Allow-Origin"] = origin
            h["Access-Control-Allow-Methods"] = get_methods()
            h["Access-Control-Max-Age"] = str(max_age)
            if headers is not None:
                h["Access-Control-Allow-Headers"] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


app = Flask(__name__)


def fen_2_json(fen: str) -> dict[str, str]:
    piecemap = chess.Board(fen=fen).piece_map()
    predictedPos = {}
    for square_index in piecemap:
        square = chess.SQUARE_NAMES[square_index]
        predictedPos[square] = str(piecemap[square_index].symbol())
    return predictedPos


# Initialize models globally
cv_model = None
with app.app_context():
    cv_model = ChessVision(lazy_load=False)


@app.route("/cv_algo/", methods=["POST", "OPTIONS"])
@crossdomain(origin="*", headers=["Content-Type"])
def cv_algo() -> tuple[dict[str, str], int]:
    """Process image from web interface."""
    global cv_model

    if cv_model is None:
        cv_model = ChessVision(lazy_load=False)

    if request.method == "OPTIONS":
        return {"success": True}

    # Get the image data from JSON
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return {"success": False, "error": "No image data provided"}, 400

        # Decode base64 image
        img_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Invalid image data"}, 400

        # Process the image with ChessVision
        result = cv_model.process_image(img)

        if result.position is None:
            return {"success": False, "error": "No chessboard detected"}, 400

        # Convert results to JSON
        response = {
            "success": True,
            "fen": result.position.fen,
            "position": fen_2_json(result.position.fen),
            "confidence_scores": result.position.confidence_scores,
            "board_confidence": result.board_extraction.confidence_scores,
            "processing_time": result.processing_time,
        }

        # Save results if needed
        if not os.getenv("LOCAL"):
            # Generate unique filename
            filename = f"{uuid.uuid4()}.jpg"

            # Save the original image
            cv2.imwrite(str(uploads_folder / "raw" / filename), img)

            # Save the extracted board if available
            if result.board_extraction.board_image is not None:
                cv2.imwrite(str(uploads_folder / "boards" / filename), result.board_extraction.board_image)

        return flask.jsonify(response)

    except Exception as e:
        logger.exception("Error processing image")
        return {"success": False, "error": str(e)}, 500


@app.route("/classify_image", methods=["POST"])
def classify_image() -> tuple[dict[str, str], int]:
    global cv_model

    if cv_model is None:
        cv_model = ChessVision(lazy_load=False)

    # Get the image from the POST request
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No image selected", 400

    # Save the image temporarily
    filename = secure_filename(file.filename)
    filepath = tmp_folder / filename
    file.save(filepath)

    # Read and process the image
    img = cv2.imread(str(filepath))
    if img is None:
        return "Invalid image", 400

    try:
        # Process the image with ChessVision
        result = cv_model.process_image(img)

        if result.position is None:
            return "No chessboard detected", 400

        # Convert results to JSON
        response = {
            "success": True,
            "fen": result.position.fen,
            "position": fen_2_json(result.position.fen),
            "confidence_scores": result.position.confidence_scores,
            "board_confidence": result.board_extraction.confidence_scores,
            "processing_time": result.processing_time,
        }

        # Save results if needed
        if not os.getenv("LOCAL"):
            # Save the original image
            cv2.imwrite(str(uploads_folder / "raw" / filename), img)

            # Save the extracted board if available
            if result.board_extraction.board_image is not None:
                cv2.imwrite(str(uploads_folder / "boards" / filename), result.board_extraction.board_image)

        return flask.jsonify(response)

    except Exception as e:
        logger.exception("Error processing image")
        return str(e), 500

    finally:
        # Clean up temporary file
        filepath.unlink(missing_ok=True)


@app.route("/feedback/", methods=["POST", "OPTIONS"])
@crossdomain(origin="*", headers=["Content-Type"])
def feedback():
    """Handle user feedback on predictions."""
    if request.method == "OPTIONS":
        return {"success": True}

    try:
        data = request.get_json()
        if not data:
            return json.dumps({"success": "false", "error": "No data provided"}), 400

        # Required fields
        if not all(k in data for k in ["position", "flip", "predictedFEN", "id"]):
            return json.dumps({"success": "false", "error": "Missing required fields"}), 400

        # Save feedback if not in local mode
        if not os.getenv("LOCAL"):
            feedback_id = str(uuid.uuid4())
            feedback_path = uploads_folder / "feedback" / f"{feedback_id}.json"
            feedback_path.parent.mkdir(exist_ok=True)

            with open(feedback_path, "w") as f:
                json.dump(
                    {
                        "id": data["id"],
                        "position": data["position"],
                        "flip": data["flip"],
                        "predicted_fen": data["predictedFEN"],
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

        return json.dumps({"success": "true"})

    except Exception as e:
        logger.exception("Error processing feedback")
        return json.dumps({"success": "false", "error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run in local mode")
    args = parser.parse_args()

    os.environ["LOCAL"] = "1" if args.local else "0"
    port = 7777 if args.local else 8080
    cv_model = ChessVision(lazy_load=False)

    app.run(host="127.0.0.1", port=port)
