import argparse
from pathlib import Path

import gradio as gr
import torch
from torch import nn
import numpy as np
from PIL import Image

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(1,32,(5,5),padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,(5,5),padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,10)
        )
    def forward(self,x):
        head_input = self.cnn_stack(x)
        logits = self.head(head_input)
        return logits

_parser = argparse.ArgumentParser(description="MNIST CNN demo")
_parser.add_argument(
    "--weights",
    type=Path,
    default=Path(__file__).resolve().parent.parent / "cnn_weights.pt",
    help="path to the model weights",
)
model_path = _parser.parse_args().weights

model = NeuralNetwork()
model.load_state_dict(torch.load(model_path,map_location = "cpu"))

def preprocess(sketch):
    if sketch is None:
        return None
    img = sketch.get("composite") if isinstance(sketch, dict) else sketch
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 4:
            rgba = Image.fromarray(img, mode="RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            pil = Image.alpha_composite(white, rgba).convert("L")
        else:
            pil = Image.fromarray(img).convert("L")
    else:
        pil = img.convert("L")
    arr = np.array(pil, dtype=np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    if arr.max() < 10:
        return None
    coords = np.argwhere(arr > 20)
    if len(coords) == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]
    h, w = cropped.shape
    scale = 20 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.array(
        Image.fromarray(cropped.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float32,
    )
    ys, xs = np.where(resized > 0)
    if len(ys) > 0:
        cy, cx = ys.mean(), xs.mean()
    else:
        cy, cx = new_h / 2, new_w / 2
    y_off = max(0, min(28 - new_h, int(round(14 - cy))))
    x_off = max(0, min(28 - new_w, int(round(14 - cx))))
    canvas28 = np.zeros((28, 28), dtype=np.float32)
    canvas28[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas28 / 255.0

def predict(sketch):
    arr = preprocess(sketch)
    if arr is None:
        return {str(i): 0.0 for i in range(10)}
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return {str(i): float(probs[i]) for i in range(10)}


import io
import socket
import threading
import time
import webbrowser

import anyio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent / "static"
# 7860 is Gradio's default; staying off it avoids colliding with a stale server.
PORT = 7864
URL = f"http://127.0.0.1:{PORT}"

def decode_canvas(png_bytes):
    """Flatten a canvas PNG onto white and hand back a PIL image for preprocess()."""
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode in ("RGBA", "LA", "P"):
        img = Image.alpha_composite(
            Image.new("RGBA", img.size, (255, 255, 255, 255)), img.convert("RGBA")
        )
    return img.convert("RGB")


app = FastAPI(title="MNIST CNN")


@app.post("/predict")
async def predict_endpoint(request: Request):
    body = await request.body()
    if not body:
        return JSONResponse({"probs": [0.0] * 10, "top": None})
    # Torch releases the GIL but still blocks; keep it off the event loop thread.
    scores = await anyio.to_thread.run_sync(lambda: predict(decode_canvas(body)))
    probs = [scores[str(i)] for i in range(10)]
    top = max(range(10), key=probs.__getitem__) if max(probs) > 0.0 else None
    return JSONResponse({"probs": probs, "top": top})


@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(
                f"Port {PORT} is already in use by another process.\n"
                f"Stop it first:  lsof -nP -iTCP:{PORT} -sTCP:LISTEN"
            )

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    )

    def open_when_ready():
        while not server.started:
            time.sleep(0.05)
        webbrowser.open(URL)

    threading.Thread(target=open_when_ready, daemon=True).start()
    print(f"MNIST CNN  ->  {URL}   (ctrl-c to stop)")
    server.run()
