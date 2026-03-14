from __future__ import annotations
from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


# ── Group A: Frame Identity ───────────────────────────────────────────────────

class FrameMeta(BaseModel):
    frame_id: str
    video_filename: str
    frame_index: int
    timestamp_seconds: float
    capture_interval_seconds: float
    annotated_at: str
    annotator: str = "claude-3-5-sonnet"
    annotation_version: str = "1.0.0"


# ── Group B: 3D Spatial Understanding ────────────────────────────────────────

class SpatialRelationship(BaseModel):
    subject: str
    relation: str   # above, below, left_of, right_of, in_contact_with, inside, holding, adjacent_to
    object: str

class SceneGeometry(BaseModel):
    workspace_type: str              # workbench, floor_assembly, overhead_crane_area, conveyor, unknown
    camera_perspective: str          # top_down, side_view, isometric, first_person, oblique
    camera_height_estimate_m: Optional[float] = None
    estimated_workspace_dimensions_m: Optional[dict] = None   # {width, depth, height}
    depth_layers: List[str] = []     # foreground → background ordered list
    reference_objects: List[str] = []
    occlusions: List[str] = []
    spatial_relationships: List[SpatialRelationship] = []

    @field_validator("reference_objects", "depth_layers", "occlusions", mode="before")
    @classmethod
    def coerce_list_items_to_str(cls, v):
        """Claude sometimes returns dicts instead of strings — coerce them."""
        if not isinstance(v, list):
            return v
        result = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # Flatten dict to a readable string e.g. "oak_panel (~600mm)"
                parts = [str(val) for val in item.values() if val]
                result.append(" ".join(parts) if parts else str(item))
            else:
                result.append(str(item))
        return result


# ── Group C: Human Pose and Action ───────────────────────────────────────────

class FramePosition(BaseModel):
    x_pct: float   # 0-1 normalised
    y_pct: float

class Operator(BaseModel):
    id: str                          # operator_1, operator_2 …
    body_region_visible: List[str] = []
    dominant_hand: str = "unknown"   # left, right, both, unknown
    pose_state: str                  # standing, crouching, kneeling, sitting, leaning_forward
    gaze_direction: str              # at_workpiece, at_tool, at_display, away, unknown
    action_verb: str                 # drilling, measuring, clamping, inspecting …
    action_phase: str                # approaching, active_manipulation, releasing, inspecting, idle, transitioning
    grip_type: str                   # power_grip, pinch_grip, hook_grip, bimanual, none
    force_estimate: str              # light_touch, moderate, heavy_force, unknown
    ppe_worn: List[str] = []
    approximate_position_in_frame: Optional[FramePosition] = None

class HumanPresence(BaseModel):
    count: int = 0
    operators: List[Operator] = []


# ── Group D: Object & Tool Detection ─────────────────────────────────────────

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class DetectedObject(BaseModel):
    id: str
    category: str   # workpiece, hand_tool, power_tool, fastener, fixture, measuring_instrument,
                    # safety_equipment, material, waste
    label: str
    bounding_box_pct: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    state: str                       # in_use, idle, being_transported, assembled, disassembled, damaged
    material: Optional[str] = None
    surface_finish_visible: Optional[str] = None
    approximate_size_class: Optional[str] = None
    held_by: Optional[str] = None    # operator id or null
    contact_surface: Optional[str] = None


# ── Group E: Manufacturing Process State ─────────────────────────────────────

class Defect(BaseModel):
    type: str
    severity: str   # cosmetic, functional, critical
    location_description: str

class Process(BaseModel):
    operation_name: str
    operation_category: str     # cutting, joining, shaping, finishing, assembly, inspection,
                                # material_handling, setup
    process_phase: str          # setup, in_progress, paused, quality_check, cleanup, complete
    estimated_completion_pct: int = Field(ge=0, le=100)
    workpiece_state_before: Optional[str] = None
    workpiece_state_after: Optional[str] = None
    quality_indicators: List[str] = []
    defects_visible: List[Defect] = []
    production_sequence_tag: Optional[str] = None
    machine_settings_visible: Optional[dict] = None
    consumables_present: List[str] = []


# ── Group F: Robot-Actionable Instructions ────────────────────────────────────

class GraspPoseHint(BaseModel):
    approach_vector: List[float] = []   # [x, y, z] unit vector
    grasp_orientation: str              # top_down, side, angled_45
    pregrasp_offset_mm: int = 50

class ForceTorqueProfile(BaseModel):
    max_force_N: Optional[float] = None
    contact_force_N: Optional[float] = None
    approach_speed: str = "normal"      # slow, normal, fast

class RobotInstruction(BaseModel):
    target_action: str                  # DRILL_HOLE, APPLY_GLUE_BEAD, PLACE_PANEL, TIGHTEN_CLAMP …
    action_parameters: dict = {}
    end_effector_required: str          # drill_chuck, parallel_gripper, soft_gripper,
                                        # glue_dispenser, vacuum_cup, none
    grasp_pose_hint: Optional[GraspPoseHint] = None
    target_object_id: Optional[str] = None
    target_location_description: str
    force_torque_profile: Optional[ForceTorqueProfile] = None
    success_condition: str
    prerequisite_state: str
    safety_clearance_mm: int = 500
    estimated_duration_seconds: float
    human_handoff_required: bool = False
    next_logical_action: Optional[str] = None


# ── Group G: Safety & Compliance ─────────────────────────────────────────────

class Hazard(BaseModel):
    type: str
    severity: str   # low, medium, high, critical
    description: str

class Safety(BaseModel):
    overall_compliance: str     # compliant, minor_violation, major_violation, critical_hazard
    ppe_compliance: bool
    missing_ppe: List[str] = []
    hazards_detected: List[Hazard] = []
    hazard_types: List[str] = []
    human_robot_proximity: str  # safe_>1m, caution_0.5-1m, danger_<0.5m, no_humans
    robot_stop_required: bool
    housekeeping_state: str     # clear, minor_clutter, obstructed_walkway, hazardous_clutter
    lockout_tagout_visible: bool = False
    emergency_stop_accessible: Optional[bool] = None


# ── Group H: Scene Context ────────────────────────────────────────────────────

class SceneContext(BaseModel):
    lighting_quality: str   # adequate, poor_shadows, glare, overexposed, underexposed
    image_quality: str      # sharp, motion_blur, out_of_focus, occluded_critical_area
    environment: str        # indoor_workshop, cleanroom, outdoor, loading_bay
    ambient_conditions: Optional[dict] = None   # {dust_level, chips_present}
    concurrent_operations: List[str] = []
    annotation_flags: List[str] = []            # low_confidence, needs_human_review …
    free_text_notes: Optional[str] = None


# ── Root VLA Annotation ───────────────────────────────────────────────────────

class VLAAnnotation(BaseModel):
    # Group A
    frame_id: str
    video_filename: str
    frame_index: int
    timestamp_seconds: float
    capture_interval_seconds: float
    annotated_at: str
    annotator: str = "claude-3-5-sonnet"
    annotation_version: str = "1.0.0"

    # Groups B–H
    scene_geometry: SceneGeometry
    human_presence: HumanPresence
    objects: List[DetectedObject]
    process: Process
    robot_instruction: RobotInstruction
    safety: Safety
    scene_context: SceneContext


# ── Request / Response wrappers ───────────────────────────────────────────────

class AnnotateRequest(BaseModel):
    frame_base64: str       # raw base64, no data-URL prefix
    mime_type: str = "image/jpeg"
    frame_index: int
    timestamp_seconds: float
    capture_interval_seconds: float
    video_filename: str
    session_id: str

class AnnotateResponse(BaseModel):
    ok: bool
    annotation: Optional[VLAAnnotation] = None
    error: Optional[str] = None
    partial: bool = False                    # True when some fields were skipped
    skipped_fields: List[str] = []          # fields that failed validation and were dropped
