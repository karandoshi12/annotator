import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, FileResponse
from pydantic import ValidationError

from schemas.vla_schema import AnnotateRequest, AnnotateResponse, VLAAnnotation
from services.claude_vision import annotate_frame

router = APIRouter()

ANNOTATIONS_DIR = Path(__file__).parent.parent / "data" / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Top-level keys in VLAAnnotation that are optional/nullable when invalid
OPTIONAL_TOP_LEVEL = {
    "scene_geometry", "human_presence", "objects",
    "process", "robot_instruction", "safety", "scene_context",
}

# Sensible empty defaults to substitute when a block fails validation
_DEFAULTS: dict[str, Any] = {
    "scene_geometry": {
        "workspace_type": "unknown",
        "camera_perspective": "unknown",
        "depth_layers": [],
        "reference_objects": [],
        "occlusions": [],
        "spatial_relationships": [],
    },
    "human_presence": {"count": 0, "operators": []},
    "objects": [],
    "process": {
        "operation_name": "unknown",
        "operation_category": "setup",
        "process_phase": "in_progress",
        "estimated_completion_pct": 0,
    },
    "robot_instruction": {
        "target_action": "OBSERVE_ONLY",
        "end_effector_required": "none",
        "target_location_description": "unknown",
        "success_condition": "unknown",
        "prerequisite_state": "unknown",
        "estimated_duration_seconds": 0.0,
    },
    "safety": {
        "overall_compliance": "compliant",
        "ppe_compliance": True,
        "human_robot_proximity": "no_humans",
        "robot_stop_required": False,
        "housekeeping_state": "clear",
    },
    "scene_context": {
        "lighting_quality": "adequate",
        "image_quality": "sharp",
        "environment": "indoor_workshop",
    },
}


def _session_file(session_id: str) -> Path:
    return ANNOTATIONS_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> dict:
    f = _session_file(session_id)
    if f.exists():
        return json.loads(f.read_text())
    return {}


def _save_session(session_id: str, data: dict):
    _session_file(session_id).write_text(json.dumps(data, indent=2))


def _try_validate_partial(raw: dict) -> tuple[VLAAnnotation, list[str]]:
    """
    Try to build a VLAAnnotation from raw. If validation fails, drop the
    offending top-level blocks one by one (replacing with empty defaults)
    and keep trying. Returns (annotation, list_of_skipped_field_paths).
    """
    skipped: list[str] = []
    attempt = dict(raw)

    for _ in range(len(OPTIONAL_TOP_LEVEL) + 1):
        try:
            return VLAAnnotation(**attempt), skipped
        except ValidationError as exc:
            # Collect the top-level field names that caused errors
            bad_tops: set[str] = set()
            field_details: list[str] = []

            for err in exc.errors():
                loc = err.get("loc", ())
                top = loc[0] if loc else None
                sub = ".".join(str(x) for x in loc)
                field_details.append(f"{sub}: {err['msg']}")
                if top and str(top) in OPTIONAL_TOP_LEVEL:
                    bad_tops.add(str(top))

            if not bad_tops:
                # Error is in a non-optional field — can't recover
                raise RuntimeError(
                    "Validation failed on required fields: " + "; ".join(field_details)
                ) from exc

            for field in bad_tops:
                skipped.append(field)
                attempt[field] = _DEFAULTS.get(field, None)

    raise RuntimeError("Could not produce a valid annotation after dropping optional blocks.")


@router.post("/annotate", response_model=AnnotateResponse)
async def annotate(req: AnnotateRequest):
    try:
        raw = annotate_frame(
            image_base64=req.frame_base64,
            mime_type=req.mime_type,
            frame_index=req.frame_index,
            timestamp_s=req.timestamp_seconds,
            interval_s=req.capture_interval_seconds,
            video_filename=req.video_filename,
        )

        # Inject mandatory identity fields
        raw["frame_id"] = raw.get("frame_id") or str(uuid.uuid4())
        raw["video_filename"] = req.video_filename
        raw["frame_index"] = req.frame_index
        raw["timestamp_seconds"] = req.timestamp_seconds
        raw["capture_interval_seconds"] = req.capture_interval_seconds
        raw["annotated_at"] = datetime.now(timezone.utc).isoformat()

        annotation, skipped = _try_validate_partial(raw)

        # Persist to session file
        session = _load_session(req.session_id)
        session[str(req.frame_index)] = annotation.model_dump()
        _save_session(req.session_id, session)

        return AnnotateResponse(
            ok=True,
            annotation=annotation,
            partial=bool(skipped),
            skipped_fields=skipped,
        )

    except Exception as e:
        return AnnotateResponse(ok=False, error=str(e))


@router.patch("/annotate/{session_id}/{frame_index}")
async def update_annotation(session_id: str, frame_index: int, body: dict):
    """Persist human-edited annotation corrections."""
    session = _load_session(session_id)
    key = str(frame_index)
    if key not in session:
        return JSONResponse(status_code=404, content={"error": "Frame not found"})
    session[key].update(body)
    _save_session(session_id, session)
    return {"ok": True}


@router.get("/export/{session_id}")
async def export_jsonl(session_id: str):
    """
    Returns a .jsonl file — one JSON object per line.
    Standard format for VLA training datasets (OpenVLA, RT-2, etc.).
    """
    session = _load_session(session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    # Sort by frame_index for deterministic output
    records = [session[k] for k in sorted(session.keys(), key=int)]
    jsonl_content = "\n".join(json.dumps(r) for r in records)

    out_path = ANNOTATIONS_DIR / f"{session_id}.jsonl"
    out_path.write_text(jsonl_content)

    return FileResponse(
        path=str(out_path),
        media_type="application/x-ndjson",
        filename=f"vla_annotations_{session_id[:8]}.jsonl",
    )


@router.get("/sessions/{session_id}/status")
async def session_status(session_id: str):
    session = _load_session(session_id)
    return {
        "session_id": session_id,
        "annotated_frames": len(session),
        "frame_indices": sorted(int(k) for k in session.keys()),
    }
