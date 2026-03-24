from __future__ import annotations

import io
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, request

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model/model.joblib")
CLIP_MIN, CLIP_MAX = 0, 20

app = Flask(__name__)
model = None


def get_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/ping", methods=["GET"])
def ping():
    try:
        get_model()
        return Response(response="OK", status=200)
    except Exception as e:
        return Response(response=str(e), status=500)


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        loaded_model = get_model()

        content_type = request.content_type or ""

        if "text/csv" in content_type:
            csv_data = request.data.decode("utf-8")
            X = pd.read_csv(io.StringIO(csv_data))
        elif "application/json" in content_type:
            payload = request.get_json()

            if isinstance(payload, dict) and "instances" in payload:
                X = pd.DataFrame(payload["instances"])
            elif isinstance(payload, list):
                X = pd.DataFrame(payload)
            else:
                return Response(
                    response=json.dumps(
                        {"error": "JSON debe ser una lista o un dict con key 'instances'."}
                    ),
                    status=400,
                    mimetype="application/json",
                )
        else:
            return Response(
                response=json.dumps(
                    {"error": f"Content-Type no soportado: {content_type}"}
                ),
                status=415,
                mimetype="application/json",
            )
   
for col in ["item_cnt_month", "ID"]:
    if col in X.columns:
        X = X.drop(columns=[col])
        
        preds = loaded_model.predict(X)
        preds = np.clip(preds, CLIP_MIN, CLIP_MAX)

        return Response(
            response=json.dumps({"predictions": preds.tolist()}),
            status=200,
            mimetype="application/json",
        )

    except Exception as e:
        return Response(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )
