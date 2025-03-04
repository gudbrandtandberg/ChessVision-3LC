import argparse
import base64
import json
import logging
import os
import uuid
from datetime import timedelta
from functools import update_wrapper
from pathlib import Path

import chess
import cv2
import flask
import numpy as np
from flask import Flask, current_app, make_response, request
from werkzeug.utils import secure_filename

from chessvision.predict.classify_raw import classify_raw
from chessvision.predict.extract_squares import extract_squares
from chessvision.utils import DATA_ROOT

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
    def format(self, record):
        try:
            record.url = request.url
            record.remote_addr = request.remote_addr
        except:
            record.url = "-"
            record.remote_addr = "-"
        return super().format(record)


formatter = RequestFormatter("[%(asctime)s] %(remote_addr)s requested %(url)s. %(levelname)s in %(name)s: %(message)s")

logger = logging.getLogger("chessvision")
file_handler = logging.FileHandler(str(logs_folder / "cv_endpoint.log"), "w")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def crossdomain(origin=None, methods=None, headers=None, max_age=21600, attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests."""
    if methods is not None:
        methods = ", ".join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ", ".join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ", ".join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """Determines which methods are allowed"""
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers["allow"]

    def decorator(f):
        """The decorator function"""

        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code"""
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
            h["Access-Control-Allow-Credentials"] = "true"
            h["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h["Access-Control-Allow-Headers"] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(uploads_folder)
app.config["TMP_FOLDER"] = str(tmp_folder)
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])

app.secret_key = "super secret key"


# Custom exception for board extraction errors
class BoardExtractionError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def json_string(self):
        return f'{{"error": "true", "message": "{self.message}"}}'


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_image_from_b64(b64string):
    buffer = base64.b64decode(b64string)
    nparr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route("/cv_algo/", methods=["POST", "OPTIONS"])
@crossdomain(origin="*")
def predict_img():
    logger.info("CV-Algo invoked")

    if flask.request.content_type == "application/json":
        print("Got data")
        data = json.loads(flask.request.data.decode("utf-8"))
        flipped = data.get("flip") == "true"
        image = read_image_from_b64(data.get("image", ""))

    if image is None:
        print("Did not get data")
        return flask.Response(response="Could not parse input!", status=415, mimetype="application/json")

    raw_id = str(uuid.uuid4())
    filename = secure_filename(raw_id) + ".JPG"
    tmp_loc = os.path.join(app.config["TMP_FOLDER"], filename)

    tmp_path = os.path.abspath(tmp_loc)
    cv2.imwrite(tmp_path, image)

    try:
        logger.info(f"Processing image {filename}")
        board_img, _, predictions, chessboard, FEN, squares, names = classify_raw(
            image, filename, board_model, sq_model, flip=flipped
        )

        if board_img is None:
            raise BoardExtractionError("Could not extract chessboard from image")

        # Move file to success raw folder
        os.rename(tmp_loc, os.path.join(app.config["UPLOAD_FOLDER"], "raw", filename))
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "boards/x_" + filename), board_img)

    except BoardExtractionError as e:
        # Move file to raw folder anyway
        os.rename(tmp_loc, os.path.join(app.config["UPLOAD_FOLDER"], "raw", filename))
        return e.json_string()

    except Exception as e:
        logger.debug(f"Unexpected error: {e}")
        return '{"error": "true", "message": "An unexpected error occurred"}'

    ret = f'{{ "FEN": "{FEN}", "id": "{raw_id}", "error": "false"}}'
    return ret


def expandFEN(FEN, tomove):
    return "{} {} - - 0 1".format(FEN, tomove)


piece2dir = {
    "R": "R",
    "r": "_r",
    "K": "K",
    "k": "_k",
    "Q": "Q",
    "q": "_q",
    "N": "N",
    "n": "_n",
    "P": "P",
    "p": "_p",
    "B": "B",
    "b": "_b",
    "f": "f",
}

dict = {
    "wR": "R",
    "bR": "r",
    "wK": "K",
    "bK": "k",
    "wQ": "Q",
    "bQ": "q",
    "wN": "N",
    "bN": "n",
    "wP": "P",
    "bP": "p",
    "wB": "B",
    "bB": "b",
    "f": "f",
}


def convertPosition(position):
    new = {}
    for key in position:
        new[key] = dict[position[key]]
    return new


def FEN2JSON(fen):
    piecemap = chess.Board(fen=fen).piece_map()
    predictedPos = {}
    for square_index in piecemap:
        square = chess.SQUARE_NAMES[square_index]
        predictedPos[square] = str(piecemap[square_index].symbol())
    return predictedPos


@app.route("/feedback/", methods=["POST"])
@crossdomain(origin="*")
def receive_feedback():
    res = '{"success": "false"}'

    data = json.loads(flask.request.data.decode("utf-8"))

    if "id" not in data or "position" not in data or "flip" not in data:
        logger.error("Missing form data, abort!")
        return res

    raw_id = data["id"]
    position = json.loads(data["position"])
    flip = data["flip"] == "true"
    predictedFEN = data["predictedFEN"]
    predictedPos = FEN2JSON(predictedFEN)
    position = convertPosition(position)
    board_filename = "x_" + raw_id + ".JPG"
    board_filename = os.path.join(app.config["UPLOAD_FOLDER"], "boards/", board_filename)

    if not os.path.isfile(board_filename):
        return res

    board = cv2.imread(board_filename, 0)
    squares, names = extract_squares(board, flip=flip)

    # Save each square using the 'position' dictionary from chessboard.js
    for sq, name in zip(squares, names):
        if name not in position:
            label = "f"
        else:
            label = position[name]

        if name not in predictedPos:
            predictedLabel = "f"
        else:
            predictedLabel = predictedPos[name]

        if predictedLabel == label:
            continue

        # Else, the prediction was incorrect, save it to learn from it later
        fname = str(uuid.uuid4()) + ".JPG"
        out_dir = os.path.join(app.config["UPLOAD_FOLDER"], "squares/", piece2dir[label])
        outfile = os.path.join(out_dir, fname)
        cv2.imwrite(outfile, sq)

    # remove the board file
    os.remove(board_filename)

    return '{ "success": "true" }'


@app.route("/ping/", methods=["GET"])
@crossdomain(origin="*")
def ping():
    return '{{ "success": "true"}}'


def load_models():
    from chessvision.predict.classify_raw import load_board_extractor, load_classifier

    sq_model = load_classifier()
    board_model = load_board_extractor()

    return board_model, sq_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    port = 7777 if args.local else 8080
    board_model, sq_model = load_models()

    app.run(host="0.0.0.0", port=port)
