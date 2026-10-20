# IDVision — real card detection + OCR web app

This is an implementation, not a static mockup. It calls the saved Roboflow Workflow
`idvision-sam-3-card-detection-1792484696123` for card detection, then performs cropping,
Tesseract OCR, and face/portrait detection locally in the backend. The user can edit
recognized fields and export the selected result to JSON or CSV.

## Important current limitations
- The Roboflow workflow is saved and specification-validated, but live detection accuracy
  has not yet been verified. The Roboflow project currently has no labeled training images.
- Hosted Roboflow receives each uploaded image for card detection. Do not upload real IDs
  unless you have approval and accept that data flow. OCR and face detection run locally
  inside this app's backend.
- Generic field extraction uses printed labels; different card formats may need specific rules.
- This project has not been deployed or end-to-end tested in this environment.

## Run with Docker
1. Install Docker.
2. Copy `.env.example` to `.env` and set your real Roboflow API key and a strong app password.
3. Build and run:

```bash
docker build -t idvision-app .
docker run --rm -p 8080:8080 --env-file .env idvision-app
```

4. Open `http://localhost:8080`. For remote deployment, use HTTPS and the hosting provider's
   secret manager. Never commit `.env` or put the Roboflow key in frontend JavaScript.

## Roboflow workflow
Workspace: `devopss-workspace-nms22`
Workflow: `idvision-sam-3-card-detection-1792484696123`
Workflow editor: https://app.roboflow.com/s6QfoD2rXTQFFBElPXUKy4ggeaa2/solutions/chat?workflowUrl=idvision-sam-3-card-detection-1792484696123

The workflow is separate from the existing Project Deployment, which remains unconfigured
with a trained model. This app calls the saved workflow directly.

## Endpoints
- `GET /health` — health status (no secrets exposed)
- `POST /api/extract` — image upload, hosted card detection, local OCR and portrait detection

Use authorized, nonpersonal test images first.
