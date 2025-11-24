import os
import shutil
import uuid
import json

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ===== Correct Utils Imports =====
from backend.utils.detect_faces_from_video import detect_faces_from_video
from backend.utils.identify_person import find_best_person
from backend.utils.frontalize_local import frontalize_local

# ===== Directories =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Create necessary folders
for d in [UPLOAD_DIR, TEMP_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAST_MAPPING_PATH = os.path.join(TEMP_DIR, "last_faces_map.json")


# ============================================================
# 1) UPLOAD VIDEO → DETECT FACES
# ============================================================
@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    try:
        filename = f"{uuid.uuid4().hex}_{video.filename}"
        video_path = os.path.join(UPLOAD_DIR, filename)

        # Save video
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Reset face folder
        face_dir = os.path.join(TEMP_DIR, "faces")
        if os.path.exists(face_dir):
            shutil.rmtree(face_dir)
        os.makedirs(face_dir)

        # Detect faces
        unique_paths = detect_faces_from_video(video_path, face_dir)

        # Build mapping
        mapping = {}
        for idx, p in enumerate(unique_paths):
            tid = f"face_{idx:04d}"
            mapping[tid] = {
                "track_id": tid,
                "img_path": p,
                "thumb": f"/temp/faces/{os.path.basename(p)}",
                "match": "Unknown",
                "score": None,
                "frontalized_image": None,
            }

        # Save mapping
        with open(LAST_MAPPING_PATH, "w") as f:
            json.dump(mapping, f)

        return {"faces": list(mapping.values())}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 2) IDENTIFY → STATIC FRONTAL (Option A)
# ============================================================
@app.post("/frontalize")
async def frontalize(track_id: str = Form(...)):

    if not os.path.exists(LAST_MAPPING_PATH):
        return {"error": "Upload video first"}

    with open(LAST_MAPPING_PATH) as f:
        mapping = json.load(f)

    if track_id not in mapping:
        return {"error": "Invalid track_id"}

    entry = mapping[track_id]
    face_path = entry["img_path"]

    # Identify person
    best_person, score, frontal_paths = find_best_person(face_path)

    entry["match"] = best_person
    entry["score"] = score

    if not frontal_paths:
        return {
            "frontalized_image": None,
            "match": best_person,
            "score": score,
            "error": "No frontal image found"
        }

    # Copy frontal.jpg
    out_path = frontalize_local(frontal_paths[0], RESULTS_DIR, track_id)
    out_url = f"/results/{os.path.basename(out_path)}"

    entry["frontalized_image"] = out_url

    # Save updated mapping
    with open(LAST_MAPPING_PATH, "w") as f:
        json.dump(mapping, f)

    return {
        "frontalized_image": out_url,
        "match": best_person,
        "score": score,
    }


# ============================================================
# STATIC ROUTES (Frontend)
# ============================================================
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# 👉 Serve welcome.html as the homepage
@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "welcome.html"))

# 👉 Serve all frontend files normally
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
