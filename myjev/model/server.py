"""Drop-in Jev endpoint: POST /v1/systemone with a Jev request body, returns Jev-shaped answers.

MYJEV_CKPT=runs/q4b_10k/best uvicorn myjev.model.server:app --host 0.0.0.0 --port 8000
Then point jev-ultrafast's endpoint at http://<host>:8000/v1/systemone (auth header is ignored).
"""

import os

from fastapi import FastAPI

from .predict import Predictor

app = FastAPI()
predictor = Predictor(os.environ["MYJEV_CKPT"])


@app.post("/v1/systemone")
def systemone(body: dict):
    answers = predictor.answer([body])[0]
    return {"answers": answers, "model": predictor.config["base_model"] + "+myjev"}
