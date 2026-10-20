import base64
import hmac
import io
import math
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageOps, UnidentifiedImageError

WORKSPACE = "devopss-workspace-nms22"
WORKFLOW = "idvision-sam-3-card-detection-1792484696123"
API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_CARDS = 4
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(title="IDVision", docs_url=None, redoc_url=None, openapi_url=None)
security = HTTPBasic(auto_error=False)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if FACE_CASCADE.empty():
    raise RuntimeError("OpenCV face detector is unavailable")


@app.middleware("http")
async def response_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; connect-src 'self'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )
    return response


def require_auth(credentials: HTTPBasicCredentials | None = Depends(security)):
    username, password = os.getenv("APP_USERNAME"), os.getenv("APP_PASSWORD")
    if not username or not password:
        raise HTTPException(503, "App authentication is not configured")
    valid = (
        credentials is not None
        and hmac.compare_digest(credentials.username, username)
        and hmac.compare_digest(credentials.password, password)
    )
    if not valid:
        raise HTTPException(401, "Authentication required",
                            headers={"WWW-Authenticate": 'Basic realm="IDVision"'})


@app.get("/")
def index(_=Depends(require_auth)):
    return FileResponse("static/index.html", media_type="text/html")


@app.get("/app.js")
def javascript(_=Depends(require_auth)):
    return FileResponse("static/app.js", media_type="text/javascript")


@app.get("/health")
def health():
    return {"status": "ok"}


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def extract_prediction_list(result):
    if isinstance(result, list):
        if len(result) != 1 or not isinstance(result[0], dict):
            raise ValueError("Unexpected number or shape of Workflow outputs")
        result = result[0]
    if not isinstance(result, dict) or "predictions" not in result:
        raise ValueError("Workflow did not return predictions")
    payload = result["predictions"]
    metadata = {}
    if isinstance(payload, dict):
        metadata = payload.get("image") or {}
        payload = payload.get("predictions")
    if not isinstance(payload, list):
        raise ValueError("Workflow predictions have an unexpected shape")
    return payload, metadata


def prediction_box(prediction, source_width, source_height, metadata):
    if not isinstance(prediction, dict):
        return None
    x, y = finite_number(prediction.get("x")), finite_number(prediction.get("y"))
    width = finite_number(prediction.get("width"))
    height = finite_number(prediction.get("height"))
    if None in (x, y, width, height) or width <= 0 or height <= 0:
        return None
    pw = finite_number(metadata.get("width")) if isinstance(metadata, dict) else None
    ph = finite_number(metadata.get("height")) if isinstance(metadata, dict) else None
    sx = source_width / pw if pw else 1.0
    sy = source_height / ph if ph else 1.0
    x1 = max(0, min(source_width, math.floor((x-width/2)*sx)))
    y1 = max(0, min(source_height, math.floor((y-height/2)*sy)))
    x2 = max(0, min(source_width, math.ceil((x+width/2)*sx)))
    y2 = max(0, min(source_height, math.ceil((y+height/2)*sy)))
    return [x1, y1, x2, y2] if x2-x1 >= 10 and y2-y1 >= 10 else None


def image_data_url(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def ocr_card(card):
    rgb = np.asarray(card.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    config = "--oem 3 --psm 11"
    text = pytesseract.image_to_string(gray, lang="eng", config=config).strip()
    data = pytesseract.image_to_data(
        gray, lang="eng", config=config, output_type=pytesseract.Output.DICT
    )
    scores = []
    for value, word in zip(data["conf"], data["text"]):
        score = finite_number(value)
        if word.strip() and score is not None and score >= 0:
            scores.append(score)
    return text, round(sum(scores)/len(scores), 1) if scores else None


FIELD_PATTERNS = {
    "full_name": re.compile(r"^\s*(?:full\s*name|name)\s*[:\-]\s*(.{2,80}?)\s*$", re.I),
    "document_number": re.compile(
        r"^\s*(?:document|card|identity|id)\s*(?:number|no\.?|#)\s*[:\-]\s*"
        r"([A-Z0-9][A-Z0-9\- ]{3,31})\s*$", re.I),
    "date_of_birth": re.compile(
        r"^\s*(?:date\s*of\s*birth|birth\s*date|dob)\s*[:\-]\s*(.{4,30}?)\s*$", re.I),
    "expiry_date": re.compile(
        r"^\s*(?:expiry|expiration|expires|valid\s*until)\s*(?:date)?\s*"
        r"[:\-]\s*(.{4,30}?)\s*$", re.I),
}


def extract_fields(text):
    fields = {key: "" for key in FIELD_PATTERNS}
    for line in text.splitlines():
        for key, pattern in FIELD_PATTERNS.items():
            if not fields[key]:
                match = pattern.match(line)
                if match:
                    fields[key] = match.group(1).strip()
    return fields


def detected_portrait(card):
    rgb = np.asarray(card.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32)
    )
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda face: face[2]*face[3])
    px, py = max(8, int(width*.25)), max(8, int(height*.25))
    crop = card.crop((max(0, x-px), max(0, y-py),
                      min(card.width, x+width+px), min(card.height, y+height+py)))
    return image_data_url(crop)


@app.post("/api/extract")
def extract(file: UploadFile = File(...), _=Depends(require_auth)):
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise HTTPException(503, "Roboflow API key is not configured")
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 12 MB upload limit")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        image.load()
        image = image.convert("RGB")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(413, "Image has too many pixels")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Image could not be decoded")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            temp_path = temp.name
            image.save(temp, format="PNG")
        client = InferenceHTTPClient(api_url=API_URL, api_key=api_key)
        result = client.run_workflow(
            workspace_name=WORKSPACE, workflow_id=WORKFLOW,
            images={"image": temp_path}, use_cache=False
        )
    except Exception:
        raise HTTPException(502, "Roboflow detection request failed")
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    try:
        predictions, metadata = extract_prediction_list(result)
    except ValueError:
        raise HTTPException(502, "Roboflow returned an unexpected prediction format")

    candidates = []
    for prediction in predictions:
        box = prediction_box(prediction, image.width, image.height, metadata)
        if box is None:
            continue
        candidates.append({
            "box": box,
            "confidence": finite_number(prediction.get("confidence")),
            "class_name": str(prediction.get("class") or prediction.get("class_name") or "ID card"),
        })
    candidates.sort(key=lambda item: item["confidence"]
                    if item["confidence"] is not None else -1, reverse=True)

    detections = []
    for candidate in candidates[:MAX_CARDS]:
        x1, y1, x2, y2 = candidate["box"]
        pad = max(4, round(min(x2-x1, y2-y1)*.015))
        card = image.crop((max(0, x1-pad), max(0, y1-pad),
                           min(image.width, x2+pad), min(image.height, y2+pad)))
        try:
            raw_text, ocr_confidence = ocr_card(card)
            portrait = detected_portrait(card)
        except Exception:
            raise HTTPException(500, "Local OCR or face detection failed")
        detections.append({
            **candidate, "crop": image_data_url(card), "portrait": portrait,
            "ocr_text": raw_text, "ocr_mean_confidence": ocr_confidence,
            "fields": extract_fields(raw_text),
        })
    return {"image_width": image.width, "image_height": image.height, "detections": detections}
