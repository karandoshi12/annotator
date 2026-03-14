# VLA Frame Annotator

Extract frames from a video and annotate each frame with structured AI-generated labels for robotics / Vision-Language-Action (VLA) model training — specifically designed for **furniture manufacturing** processes.

---

## What It Does

1. Upload any video (MP4, MOV, WebM, AVI)
2. Extract frames at a fixed interval (default: every 10 seconds)
3. Annotate each frame using **Claude Vision** — producing structured JSON covering:
   - Action & manufacturing process state
   - Robot-actionable instructions (target action, end effector, force profile, success condition)
   - 3D spatial understanding (workspace type, depth layers, spatial relationships)
   - Object & tool detection (bounding boxes, material, state)
   - Human pose & action (grip type, force estimate, PPE worn)
   - Safety compliance (hazards, human-robot proximity, robot stop flag)
4. Export all annotations as a `.jsonl` file — ready for OpenVLA / RT-2 training pipelines

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

---

## Setup

Add your Anthropic API key to `backend/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running

### 1. Start the backend

```bash
cd backend
python main.py
```

The server starts at:

```
http://localhost:8765
```

### 2. Open the frontend

The backend serves the frontend automatically. Open your browser and go to:

```
http://localhost:8765
```

---

## Usage

| Step | Action |
|------|--------|
| 1 | Drag & drop or click to upload a video |
| 2 | Set the frame interval (seconds) |
| 3 | Click **Extract Frames** |
| 4 | Click **Annotate All Frames** — Claude Vision analyses each frame |
| 5 | Click **Export JSONL** to download the training dataset |

Each frame card has 6 annotation tabs:

- **Action** — operation name, phase, progress %, operator grip/force, quality indicators
- **Robot** — target action verb, end effector, force profile, success condition, next action
- **Scene 3D** — workspace type, camera perspective, depth layers, spatial relationships
- **Objects** — detected tools and parts with bounding boxes, material, and state
- **Safety** — compliance level, hazards, missing PPE, human-robot proximity
- **JSON** — raw annotation JSON (copy or inspect)

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/annotate` | Annotate a single frame (base64 image) |
| `PATCH` | `/api/annotate/{session_id}/{frame_index}` | Update / correct an annotation |
| `GET` | `/api/export/{session_id}` | Download all annotations as `.jsonl` |
| `GET` | `/api/sessions/{session_id}/status` | Check how many frames are annotated |

---

## Project Structure

```
newfolder/
├── index.html                        # Frontend (served by backend)
├── README.md
└── backend/
    ├── main.py                       # FastAPI app entry point
    ├── requirements.txt
    ├── .env                          # ANTHROPIC_API_KEY goes here
    ├── routes/
    │   └── annotate.py               # API route handlers
    ├── schemas/
    │   └── vla_schema.py             # Pydantic models for VLA annotation
    ├── services/
    │   └── claude_vision.py          # Claude Vision API integration
    └── data/
        └── annotations/              # Saved session JSON + exported JSONL files
```

---

## Export Format

Each line in the exported `.jsonl` file is one frame annotation:

```json
{
  "frame_id": "uuid",
  "timestamp_seconds": 10.0,
  "process": { "operation_name": "dowel_drilling", "operation_category": "joining", ... },
  "robot_instruction": { "target_action": "DRILL_HOLE", "end_effector_required": "drill_chuck", ... },
  "safety": { "overall_compliance": "compliant", "robot_stop_required": false, ... },
  ...
}
```

Compatible with **OpenVLA**, **RT-2**, and any framework that consumes JSONL datasets.
