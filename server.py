
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


# --------------------------------------------------
# IDVision configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

WORKSPACE = "devopss-workspace-nms22"
WORKFLOW = "idvision-sam-3-card-detection-1792484696123"

API_URL = os.getenv(
    "ROBOFLOW_API_URL",
    "https://serverless.roboflow.com",
).rstrip("/")

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_CARDS = 4

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="IDVision",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

security = HTTPBasic(auto_error=False)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if FACE_CASCADE.empty():
    raise RuntimeError("OpenCV face detector could not be loaded")


# --------------------------------------------------
# Security and authentication
# --------------------------------------------------

@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    )
    return response


def require_auth(
    credentials: HTTPBasicCredentials | None = Depends(security),
):
    expected_user = os.getenv("APP_USERNAME", "")
    expected_password = os.getenv("APP_PASSWORD", "")

    if not expected_user or not expected_password:
        raise HTTPException(
            status_code=503,
            detail="Application authentication is not configured",
        )

    valid = (
        credentials is not None
        and hmac.compare_digest(credentials.username, expected_user)
        and hmac.compare_digest(credentials.password, expected_password)
    )

    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": 'Basic realm="IDVision"'},
        )


# --------------------------------------------------
# Frontend routes
# Files are stored in the repository root.
# --------------------------------------------------

@app.get("/")
def index(_=Depends(require_auth)):
    path = BASE_DIR / "index.html"

    if not path.is_file():
        raise HTTPException(500, "index.html is missing")

    return FileResponse(path, media_type="text/html")


@app.get("/app.js")
def javascript(_=Depends(require_auth)):
    path = BASE_DIR / "app.js"

    if not path.is_file():
        raise HTTPException(500, "app.js is missing")

    return FileResponse(path, media_type="text/javascript")


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Numeric and prediction helpers
# --------------------------------------------------

def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def extract_prediction_list(result):
    if isinstance(result, list):
        if len(result) != 1 or not isinstance(result[0], dict):
            raise ValueError("Unexpected workflow result")

        result = result[0]

    if not isinstance(result, dict) or "predictions" not in result:
        raise ValueError("Workflow did not return predictions")

    payload = result["predictions"]
    metadata = {}

    if isinstance(payload, dict):
        metadata = payload.get("image") or {}
        payload = payload.get("predictions")

    if not isinstance(payload, list):
        raise ValueError("Unexpected predictions format")

    return payload, metadata


def prediction_box(prediction, image_width, image_height, metadata):
    if not isinstance(prediction, dict):
        return None

    x = finite_number(prediction.get("x"))
    y = finite_number(prediction.get("y"))
    width = finite_number(prediction.get("width"))
    height = finite_number(prediction.get("height"))

    if None in (x, y, width, height):
        return None

    if width <= 0 or height <= 0:
        return None

    if not isinstance(metadata, dict):
        metadata = {}

    prediction_width = finite_number(metadata.get("width"))
    prediction_height = finite_number(metadata.get("height"))

    scale_x = (
        image_width / prediction_width
        if prediction_width and prediction_width > 0
        else 1.0
    )
    scale_y = (
        image_height / prediction_height
        if prediction_height and prediction_height > 0
        else 1.0
    )

    x1 = max(0, min(image_width, math.floor((x - width / 2) * scale_x)))
    y1 = max(0, min(image_height, math.floor((y - height / 2) * scale_y)))
    x2 = max(0, min(image_width, math.ceil((x + width / 2) * scale_x)))
    y2 = max(0, min(image_height, math.ceil((y + height / 2) * scale_y)))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return [x1, y1, x2, y2]


def image_data_url(image):
    output = io.BytesIO()
    image.save(output, format="PNG")

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------
# OCR processing
# --------------------------------------------------

def ocr_card(card):
    rgb = np.asarray(card.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    config = "--oem 3 --psm 11"

    text = pytesseract.image_to_string(
        gray,
        lang="eng",
        config=config,
    ).strip()

    data = pytesseract.image_to_data(
        gray,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    scores = []

    for confidence, word in zip(data["conf"], data["text"]):
        score = finite_number(confidence)

        if word.strip() and score is not None and score >= 0:
            scores.append(score)

    mean_confidence = (
        round(sum(scores) / len(scores), 1)
        if scores
        else None
    )

    return text, mean_confidence


# --------------------------------------------------
# Structured field extraction
# OCR results should be reviewed by the user.
# --------------------------------------------------

FIELD_PATTERNS = {
    "full_name": re.compile(
        r"^\s*(?:full\s*name|name)\s*[:\-]\s*(.{2,80}?)\s*$",
        re.I,
    ),
    "document_number": re.compile(
        r"^\s*(?:document|card|identity|id)\s*"
        r"(?:number|no\.?|#)\s*[:\-]\s*"
        r"([A-Z0-9][A-Z0-9\- ]{3,31})\s*$",
        re.I,
    ),
    "date_of_birth": re.compile(
        r"^\s*(?:date\s*of\s*birth|birth\s*date|dob)\s*"
        r"[:\-]\s*(.{4,30}?)\s*$",
        re.I,
    ),
    "expiry_date": re.compile(
        r"^\s*(?:expiry|expiration|expires|valid\s*until)\s*"
        r"(?:date)?\s*[:\-]\s*(.{4,30}?)\s*$",
        re.I,
    ),
}


def extract_fields(text):
    fields = {name: "" for name in FIELD_PATTERNS}

    for line in text.splitlines():
        for name, pattern in FIELD_PATTERNS.items():
            if fields[name]:
                continue

            match = pattern.match(line)

            if match:
                fields[name] = match.group(1).strip()

    return fields


# --------------------------------------------------
# Face / portrait detection
# --------------------------------------------------

def detected_portrait(card):
    rgb = np.asarray(card.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(32, 32),
    )

    if len(faces) == 0:
        return None

    x, y, width, height = max(
        faces,
        key=lambda face: face[2] * face[3],
    )

    pad_x = max(8, int(width * 0.25))
    pad_y = max(8, int(height * 0.25))

    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(card.width, x + width + pad_x)
    bottom = min(card.height, y + height + pad_y)

    portrait = card.crop((left, top, right, bottom))

    return image_data_url(portrait)


# --------------------------------------------------
# Main ID-card extraction endpoint
# --------------------------------------------------

@app.post("/api/extract")
def extract(
    file: UploadFile = File(...),
    _=Depends(require_auth),
):
    api_key = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Roboflow API key is not configured",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image",
        )

    data = file.file.read(MAX_UPLOAD_BYTES + 1)

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 12 MB upload limit",
        )

    if not data:
        raise HTTPException(400, "The uploaded image is empty")

    try:
        with Image.open(io.BytesIO(data)) as original:
            image = ImageOps.exif_transpose(original)
            image.load()
            image = image.convert("RGB")

        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail="Image has too many pixels",
            )

    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be decoded",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as temp:
            temp_path = temp.name
            image.save(temp, format="PNG")

        client = InferenceHTTPClient(
            api_url=API_URL,
            api_key=api_key,
        )

        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW,
            images={"image": temp_path},
            use_cache=False,
        )

    except Exception:
        # Avoid exposing credentials or provider internals.
        raise HTTPException(
            status_code=502,
            detail="Roboflow detection request failed. Check the API key, workflow and server logs.",
        )

    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    try:
        predictions, metadata = extract_prediction_list(result)
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Roboflow returned an unexpected prediction format",
        )

    candidates = []

    for prediction in predictions:
        box = prediction_box(
            prediction,
            image.width,
            image.height,
            metadata,
        )

        if box is None:
            continue

        candidates.append({
            "box": box,
            "confidence": finite_number(
                prediction.get("confidence")
            ),
            "class_name": str(
                prediction.get("class")
                or prediction.get("class_name")
                or "ID card"
            ),
        })

    candidates.sort(
        key=lambda item: (
            item["confidence"]
            if item["confidence"] is not None
            else -1
        ),
        reverse=True,
    )

    detections = []

    for candidate in candidates[:MAX_CARDS]:
        x1, y1, x2, y2 = candidate["box"]

        padding = max(
            4,
            round(min(x2 - x1, y2 - y1) * 0.015),
        )

        card = image.crop((
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(image.width, x2 + padding),
            min(image.height, y2 + padding),
        ))

        try:
            raw_text, ocr_confidence = ocr_card(card)
            portrait = detected_portrait(card)

        except Exception:
            raise HTTPException(
                status_code=500,
                detail="OCR or face detection failed. Check the server logs.",
            )

        detections.append({
            **candidate,
            "crop": image_data_url(card),
            "portrait": portrait,
            "ocr_text": raw_text,
            "ocr_mean_confidence": ocr_confidence,
            "fields": extract_fields(raw_text),
        })

    return {
        "image_width": image.width,
        "image_height": image.height,
        "detections": detections,
    }
    
