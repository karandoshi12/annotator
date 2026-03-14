"""
Calls Claude Vision API to produce a structured VLA annotation for a single frame.
"""
import json
import re
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# Allowed robot action vocabulary (constrained for robot executability)
ROBOT_ACTIONS = [
    "DRILL_HOLE", "DRIVE_SCREW", "APPLY_GLUE_BEAD", "SPREAD_GLUE",
    "PLACE_PANEL", "POSITION_COMPONENT", "ALIGN_COMPONENT",
    "CLAMP_WORKPIECE", "RELEASE_CLAMP", "TIGHTEN_FASTENER", "LOOSEN_FASTENER",
    "CUT_MATERIAL", "SAND_SURFACE", "INSPECT_SURFACE", "MEASURE_DIMENSION",
    "PICK_TOOL", "PLACE_TOOL", "PICK_PART", "PLACE_PART",
    "APPLY_FINISH", "WIPE_SURFACE", "FLIP_WORKPIECE", "ROTATE_WORKPIECE",
    "TRANSPORT_WORKPIECE", "OBSERVE_ONLY"
]

SYSTEM_PROMPT = """You are an expert VLA (Vision-Language-Action) robotics annotation specialist
for furniture manufacturing environments. Your job is to analyse a single video frame and produce
a richly structured JSON annotation that a robot or VLA model can directly consume for imitation
learning, task planning, and safety-aware execution.

You must output ONLY a single valid JSON object — no markdown fences, no prose, no comments.
Fill every field as accurately as possible from visual evidence. If a field is genuinely
unobservable, use a sensible default or null as specified.

Be precise about spatial relationships and robot-actionable instructions. Use SI units throughout."""


def _build_user_prompt(frame_index: int, timestamp_s: float, video_filename: str,
                       interval_s: float) -> str:
    return f"""Analyse the provided video frame and produce a complete VLA annotation JSON.

Frame context:
- Source video: {video_filename}
- Frame index: {frame_index}
- Timestamp: {timestamp_s:.2f}s  (captured every {interval_s}s)
- Domain: furniture manufacturing

Output a single JSON object with EXACTLY these top-level keys (all required):
  frame_id, video_filename, frame_index, timestamp_seconds, capture_interval_seconds,
  annotated_at, annotator, annotation_version,
  scene_geometry, human_presence, objects, process, robot_instruction, safety, scene_context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD SPECIFICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

frame_id: generate a UUID v4 string
video_filename: "{video_filename}"
frame_index: {frame_index}
timestamp_seconds: {timestamp_s}
capture_interval_seconds: {interval_s}
annotated_at: current ISO-8601 UTC datetime
annotator: "claude-3-5-sonnet"
annotation_version: "1.0.0"

── scene_geometry ──
workspace_type: one of [workbench, floor_assembly, overhead_crane_area, conveyor, unknown]
camera_perspective: one of [top_down, side_view, isometric, first_person, oblique]
camera_height_estimate_m: float (best estimate from visual cues)
estimated_workspace_dimensions_m: {{width: float, depth: float, height: float}} or null
depth_layers: ordered array of strings from foreground to background
  e.g. ["operator_hands", "power_drill", "oak_panel", "workbench", "tool_rack", "wall"]
reference_objects: array of known-size objects for scale calibration
occlusions: array of "X occluded by Y" strings
spatial_relationships: array of {{subject, relation, object}} where relation is one of
  [above, below, left_of, right_of, in_contact_with, inside, holding, adjacent_to]

── human_presence ──
count: integer
operators: array of operator objects, each with:
  id: "operator_1" etc.
  body_region_visible: array from [head, torso, left_arm, right_arm, left_hand, right_hand, legs, feet]
  dominant_hand: one of [left, right, both, unknown]
  pose_state: one of [standing, crouching, kneeling, sitting, leaning_forward, reaching]
  gaze_direction: one of [at_workpiece, at_tool, at_display, at_colleague, away, unknown]
  action_verb: specific present-participle verb e.g. "drilling", "measuring", "aligning"
  action_phase: one of [approaching, active_manipulation, releasing, inspecting, idle, transitioning]
  grip_type: one of [power_grip, pinch_grip, hook_grip, bimanual, no_contact]
  force_estimate: one of [light_touch, moderate, heavy_force, unknown]
  ppe_worn: array from [safety_glasses, gloves, ear_protection, hard_hat, hi_vis_vest, apron, steel_toe_boots]
  approximate_position_in_frame: {{x_pct: 0-1, y_pct: 0-1}} torso center

── objects ──
Array of ALL significant objects. Each object:
  id: "obj_001" etc.
  category: one of [workpiece, hand_tool, power_tool, fastener, fixture, measuring_instrument,
                    safety_equipment, material, furniture_component, waste, machine]
  label: specific label e.g. "cordless_drill_dewalt", "M8_hex_bolt", "oak_side_panel_600x400mm"
  bounding_box_pct: {{x: left, y: top, width: w, height: h}} all 0-1 normalised
  confidence: 0.0-1.0
  state: one of [in_use, idle, being_transported, assembled, partially_assembled, disassembled, damaged]
  material: e.g. "oak_solid_wood", "steel_zinc_plated", "ABS_plastic"
  surface_finish_visible: one of [raw, rough_sanded, fine_sanded, painted, lacquered, stained, anodized]
  approximate_size_class: one of [small_<10cm, medium_10-50cm, large_>50cm]
  held_by: operator id string or null
  contact_surface: what the object rests on, or null

── process ──
operation_name: specific name e.g. "mortise_drilling", "dowel_insertion", "panel_glue_up",
  "tenon_cutting", "surface_sanding_120grit", "drawer_slide_installation", "quality_inspection"
operation_category: one of [cutting, joining, shaping, finishing, assembly, inspection,
                            material_handling, setup]
process_phase: one of [setup, in_progress, paused, quality_check, cleanup, complete]
estimated_completion_pct: 0-100 integer
workpiece_state_before: text description of workpiece state before operation
workpiece_state_after: text description after (if transition visible), else null
quality_indicators: array of observable quality signals
  e.g. ["drill_perpendicular_to_surface", "glue_bead_consistent_width", "clean_cut_edge"]
defects_visible: array of {{type, severity (cosmetic|functional|critical), location_description}}
production_sequence_tag: e.g. "STEP_04_DRILL_DOWEL_HOLES_SIDE_PANEL_LEFT"
machine_settings_visible: any readable settings {{rpm, depth_mm, feed_rate_mm_min}} or null
consumables_present: array of consumables visible e.g. ["sandpaper_120grit", "PVA_wood_glue"]

── robot_instruction ──
target_action: ONE of these exact strings:
  {json.dumps(ROBOT_ACTIONS, indent=2)}
action_parameters: operation-specific dict e.g.
  for DRILL_HOLE: {{diameter_mm, depth_mm, rpm, drill_axis, pilot_hole_required, tolerance_mm}}
  for APPLY_GLUE_BEAD: {{bead_width_mm, bead_length_mm, open_time_s, glue_type}}
  for DRIVE_SCREW: {{screw_size, torque_Nm, depth_flush_mm, pre_drill_required}}
end_effector_required: one of [drill_chuck, parallel_gripper, soft_gripper,
                                glue_dispenser, vacuum_cup, screw_driver_bit, sanding_pad, none]
grasp_pose_hint: {{approach_vector: [x,y,z], grasp_orientation: top_down|side|angled_45,
                   pregrasp_offset_mm: int}} or null
target_object_id: id of primary object being acted on
target_location_description: precise text e.g. "marked hole position on left rail, 45mm from top edge"
force_torque_profile: {{max_force_N: float, contact_force_N: float, approach_speed: slow|normal|fast}}
success_condition: observable completion criterion e.g. "hole_depth_equals_20mm_by_depth_stop"
prerequisite_state: what must be true first e.g. "panel_secured_in_drill_press_fixture"
safety_clearance_mm: minimum clearance from humans (integer)
estimated_duration_seconds: realistic time estimate (float)
human_handoff_required: boolean — true if human must intervene before/after
next_logical_action: the next target_action string in the sequence

── safety ──
overall_compliance: one of [compliant, minor_violation, major_violation, critical_hazard]
ppe_compliance: boolean
missing_ppe: array of PPE items that should be worn but aren't visible
hazards_detected: array of {{type, severity (low|medium|high|critical), description}}
hazard_types: array from [pinch_point, flying_debris, sharp_edge, electrical, noise,
                           chemical, ergonomic, trip_hazard, fire]
human_robot_proximity: one of [safe_>1m, caution_0.5-1m, danger_<0.5m, no_humans]
robot_stop_required: boolean — true if any critical hazard requires robot halt
housekeeping_state: one of [clear, minor_clutter, obstructed_walkway, hazardous_clutter]
lockout_tagout_visible: boolean
emergency_stop_accessible: boolean or null

── scene_context ──
lighting_quality: one of [adequate, poor_shadows, glare, overexposed, underexposed]
image_quality: one of [sharp, motion_blur, out_of_focus, occluded_critical_area]
environment: one of [indoor_workshop, cleanroom, outdoor, loading_bay, assembly_line]
ambient_conditions: {{dust_level: low|medium|high, chips_present: bool, coolant_visible: bool}}
concurrent_operations: array of other operations visible in background
annotation_flags: array from [low_confidence, needs_human_review, ambiguous_action,
                               novel_scenario, partial_visibility, complex_interaction]
free_text_notes: any important observations not captured by structured fields

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY the JSON object. Start your response with {{ and end with }}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def annotate_frame(
    image_base64: str,
    mime_type: str,
    frame_index: int,
    timestamp_s: float,
    interval_s: float,
    video_filename: str,
) -> dict:
    """
    Sends a frame to Claude Vision and returns a raw annotation dict.
    Raises on API errors.
    """
    user_prompt = _build_user_prompt(frame_index, timestamp_s, video_filename, interval_s)

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }
        ],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if Claude adds them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)
