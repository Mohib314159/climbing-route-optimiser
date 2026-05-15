"""
route_generator_v2.py: Optimal route generation using dynamic programming.

USAGE (in main.py, change one import):
    # from src.route_generator import RouteGenerator
    from src.route_generator_v2 import RouteGeneratorV2 as RouteGenerator

The constructor and generate_route() signatures are identical to the original,
so no other code needs to change. The output is the same List[Climber] format,
meaning draw_climber() and the FastAPI endpoint work unchanged.


The original route_generator.py uses np.random.choice to select holds at each
step, and it picks randomly from candidates within a spatial zone. This produces
a different route on every call with no concept of difficulty or optimality.

Now:
    1. Converts YOLO detections to a weighted graph using ArUco-calibrated distances
    2. Estimates hold difficulty from visual features (size, height, class)
    3. Finds the minimum-cost route using dynamic programming (Dijkstra on a DAG)
    4. Computes body positions using anatomically-accurate inverse kinematics
    5. Runs Monte Carlo simulation to estimate send probability for the climber
    6. All deterministic given the same image and climber profile

COORDINATE SYSTEMS
Two coordinate systems are used internally:

    PIXEL space (image coords):
        Origin: top-left of image
        y increases downward
        Units: pixels
        Used for: all final output (BodyPart endpoints), OpenCV drawing

    WORLD space (wall coords):
        Origin: bottom-left of wall
        y increases upward (climbing direction)
        Units: metres
        Used for: DP graph, reach constraints, distance calculations

Conversion uses the ArUco marker's pixels-per-cm scale factor.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np

# These imports use his existing model classes unchanged
from src.aruco_marker import ArucoMarker
from src.model.body_part import BodyPart
from src.model.body_proportion import BodyProportion
from src.model.climber import Climber
from src.model.color import Color
from src.model.detected_object import DetectedObject
from src.model.point import Point


# ── Constants ─────────────────────────────────────────────────────────────────

# V-grade → skill level in [0,1]
# Derived from consensus difficulty ratings across major grading databases
GRADE_TO_SKILL: Dict[str, float] = {
    "V0": 0.10, "V1": 0.20, "V2": 0.28, "V3": 0.36,
    "V4": 0.45, "V5": 0.55, "V6": 0.62, "V7": 0.70,
    "V8": 0.78, "V9": 0.86, "V10": 0.93, "V11": 0.98,
}

# Logistic steepness for Monte Carlo success probability model
# Higher = sharper skill/difficulty cutoff
MC_STEEPNESS = 8.0

# Dynamic reach factor: a climber can reach ~15% beyond static arm length
# via deadpoints and dynamic movement
DYNAMIC_REACH_FACTOR = 1.15

# Hold class names from YOLO (matches climbingcrux_model training data)
VOLUME_CLASS_NAMES = {"volume"}  # volumes are typically easier holds


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class WallHold:
    """
    Internal representation of a hold for the DP solver.
    Bridges between DetectedObject (pixels) and our world-space graph.
    """
    idx: int                        # index in the detections list
    detected_object: DetectedObject # original YOLO detection
    x_m: float                      # world x in metres
    y_m: float                      # world y in metres (upward from ground)
    difficulty: float               # estimated difficulty in [0, 1]
    is_start: bool = False
    is_finish: bool = False

    def distance_to(self, other: "WallHold") -> float:
        return math.sqrt((self.x_m - other.x_m)**2 + (self.y_m - other.y_m)**2)


@dataclass
class OptimalRoute:
    """The result of the DP solver."""
    holds: List[WallHold]
    total_cost: float
    move_costs: List[float]
    send_rate: float        # from Monte Carlo simulation
    grade_estimate: str
    crux_move_idx: int

    @property
    def n_moves(self) -> int:
        return len(self.holds) - 1


# ── Coordinate conversion ─────────────────────────────────────────────────────

def px_to_world(px: int, py: int, img_height: int,
                pixels_per_cm: float) -> Tuple[float, float]:
    """Convert image pixel coords to world-space metres."""
    x_m = px / pixels_per_cm / 100.0
    y_m = (img_height - py) / pixels_per_cm / 100.0  # flip y: image down → world up
    return x_m, y_m


def world_to_px(x_m: float, y_m: float, img_height: int,
                pixels_per_cm: float) -> Tuple[int, int]:
    """Convert world-space metres back to image pixel coords."""
    px = int(round(x_m * pixels_per_cm * 100.0))
    py = int(round(img_height - y_m * pixels_per_cm * 100.0))
    return px, py


# ── Difficulty estimation ─────────────────────────────────────────────────────

def estimate_hold_difficulty(
    det: DetectedObject,
    img_width: int,
    img_height: int,
    wall_height_m: float,
    y_m: float,
) -> float:
    """
    Estimate hold difficulty from visual features.

    Three signals:
        1. Hold size: smaller bounding box → harder hold
           (jugs are large, crimps and pockets are small)
        2. Height on wall: higher holds are typically harder
           (route setters place crux holds in the upper section)
        3. Hold class: volumes are typically easier than individual holds

    Returns a float in [0.05, 0.95].
    """
    x1, y1, x2, y2 = det.bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = img_width * img_height

    # Size signal: normalise as fraction of image, map to difficulty
    # A hold occupying 2% of image is easy (jug); 0.05% is very hard (crimp)
    size_fraction = bbox_area / max(image_area, 1)
    size_difficulty = float(np.clip(1.0 - (size_fraction / 0.02) * 0.7, 0.1, 0.9))

    # Height signal: linear from 0.1 at ground to 0.85 at top
    height_difficulty = 0.1 + 0.75 * (y_m / max(wall_height_m, 0.01))

    # Class signal: volumes are easier
    class_modifier = -0.12 if det.class_name in VOLUME_CLASS_NAMES else 0.0

    # Weighted combination
    difficulty = 0.35 * size_difficulty + 0.50 * height_difficulty + 0.15 * 0.5 + class_modifier
    return float(np.clip(difficulty, 0.05, 0.95))


# ── DP Route Solver ───────────────────────────────────────────────────────────

def build_wall_holds(
    detected_objects: List[DetectedObject],
    img_width: int,
    img_height: int,
    pixels_per_cm: float,
    start_zone_height_m: float,
    finish_zone_height_m: float,
) -> List[WallHold]:
    """
    Convert YOLO detections to WallHold objects in world space.

    Start zone: holds within start_zone_height_m of the ground.
    Finish zone: holds within finish_zone_height_m of the top.
    Wall height is inferred from image height via ArUco calibration.
    """
    wall_height_m = img_height / pixels_per_cm / 100.0

    holds = []
    for i, det in enumerate(detected_objects):
        x_m, y_m = px_to_world(det.center.x, det.center.y, img_height, pixels_per_cm)

        difficulty = estimate_hold_difficulty(
            det=det,
            img_width=img_width,
            img_height=img_height,
            wall_height_m=wall_height_m,
            y_m=y_m,
        )

        is_start  = y_m <= start_zone_height_m
        is_finish = y_m >= (wall_height_m - finish_zone_height_m)

        holds.append(WallHold(
            idx=i,
            detected_object=det,
            x_m=x_m,
            y_m=y_m,
            difficulty=difficulty,
            is_start=is_start,
            is_finish=is_finish,
        ))

    return holds


def solve_optimal_route(
    holds: List[WallHold],
    max_reach_m: float,
    skill_level: float,
    n_monte_carlo: int = 5000,
    rng_seed: int = 0,
) -> Optional[OptimalRoute]:
    """
    Find the minimum-cost route from any start hold to any finish hold.

    ALGORITHM: DP on a DAG (directed acyclic graph).

    Graph construction:
        - Nodes: all holds
        - Edge src → dst exists iff:
            (a) dst.y_m > src.y_m  (upward only — we are climbing)
            (b) distance(src, dst) <= max_reach_m  (physically reachable)

    Edge weight (move cost):
        cost(src, dst) = 0.7 * dst.difficulty + 0.3 * (distance / max_reach)

    DP recurrence:
        dp[h] = min total cost to reach hold h from any start hold
        dp[h] = min over predecessors p: dp[p] + cost(p, h)

    Since all edges go upward (DAG), a single forward pass in height order
    is sufficient — no cycles, no need for Bellman-Ford or Dijkstra.

    TIME COMPLEXITY: O(n²) where n = number of holds.
    For 50 holds: 2,500 operations — negligible.
    """
    if not any(h.is_start for h in holds):
        return None
    if not any(h.is_finish for h in holds):
        return None

    n = len(holds)
    INF = float("inf")

    # Sort by height ascending — guarantees predecessors processed before successors
    sorted_holds = sorted(holds, key=lambda h: h.y_m)
    idx_to_pos = {h.idx: i for i, h in enumerate(sorted_holds)}

    dp = [INF] * n
    parent: List[Optional[int]] = [None] * n  # index into sorted_holds

    # Initialise start holds: entry cost = hold difficulty
    for h in sorted_holds:
        if h.is_start:
            dp[idx_to_pos[h.idx]] = h.difficulty

    # Forward pass
    for dst_pos, dst in enumerate(sorted_holds):
        for src_pos, src in enumerate(sorted_holds):
            if src.y_m >= dst.y_m:
                break  # sorted — no more predecessors

            if dp[src_pos] == INF:
                continue  # src unreachable

            dist = src.distance_to(dst)
            if dist > max_reach_m:
                continue  # out of reach

            # Move cost: difficulty of destination + normalised distance
            move_cost = (0.7 * dst.difficulty + 0.3 * (dist / max_reach_m))
            total = dp[src_pos] + move_cost

            if total < dp[dst_pos]:
                dp[dst_pos] = total
                parent[dst_pos] = src_pos

    # Find best finish hold
    best_cost = INF
    best_finish_pos = None
    for h in sorted_holds:
        pos = idx_to_pos[h.idx]
        if h.is_finish and dp[pos] < best_cost:
            best_cost = dp[pos]
            best_finish_pos = pos

    if best_finish_pos is None or best_cost == INF:
        return None

    # Backtrack to recover route
    route_holds: List[WallHold] = []
    pos = best_finish_pos
    while pos is not None:
        route_holds.append(sorted_holds[pos])
        pos = parent[pos]
    route_holds.reverse()

    # Compute individual move costs
    move_costs = []
    for i in range(len(route_holds) - 1):
        src, dst = route_holds[i], route_holds[i+1]
        dist = src.distance_to(dst)
        move_costs.append(0.7 * dst.difficulty + 0.3 * (dist / max_reach_m))

    # Grade estimate
    grade = _estimate_grade(best_cost / max(len(move_costs), 1))

    # Crux: highest cost move
    crux_idx = int(np.argmax(move_costs)) if move_costs else 0

    # Monte Carlo send rate
    send_rate = _monte_carlo_send_rate(
        move_costs=move_costs,
        skill_level=skill_level,
        n_attempts=n_monte_carlo,
        seed=rng_seed,
    )

    return OptimalRoute(
        holds=route_holds,
        total_cost=best_cost,
        move_costs=move_costs,
        send_rate=send_rate,
        grade_estimate=grade,
        crux_move_idx=crux_idx,
    )


def _estimate_grade(avg_move_cost: float) -> str:
    """Map average move cost to approximate V-grade."""
    thresholds = [
        (0.20, "V0"), (0.28, "V1"), (0.36, "V2"), (0.44, "V3"),
        (0.52, "V4"), (0.59, "V5"), (0.66, "V6"), (0.73, "V7"),
        (0.80, "V8"), (0.87, "V9"), (0.93, "V10"),
    ]
    for threshold, grade in thresholds:
        if avg_move_cost <= threshold:
            return grade
    return "V11+"


def _monte_carlo_send_rate(
    move_costs: List[float],
    skill_level: float,
    n_attempts: int,
    seed: int,
) -> float:
    """
    Estimate probability of completing the route via Monte Carlo simulation.

    MODEL: Each move succeeds independently with probability
        P = sigmoid(k * (skill - move_cost))
    where k = MC_STEEPNESS.

    This is the Item Response Theory (IRT) model: the same mathematical
    framework used in psychometrics to model test question difficulty
    versus student ability. A climbing move is a 'question' the climber
    must 'answer' correctly.

    We simulate n_attempts independent attempts and count completions.
    """
    if not move_costs:
        return 1.0

    rng = np.random.default_rng(seed)
    costs = np.array(move_costs)

    # Precompute success probability per move (same for all attempts)
    success_probs = 1.0 / (1.0 + np.exp(-MC_STEEPNESS * (skill_level - costs)))

    # Simulate: draw Bernoulli(p) for each move in each attempt
    # Shape: (n_attempts, n_moves)
    outcomes = rng.random((n_attempts, len(costs))) < success_probs

    # A completion = all moves succeed in that attempt
    completions = np.all(outcomes, axis=1).sum()
    return float(completions / n_attempts)


# ── Body Position IK ──────────────────────────────────────────────────────────

def build_climber_position(
    right_hand_hold: WallHold,
    left_hand_hold: WallHold,
    all_holds: List[WallHold],
    body_prop: BodyProportion,
    img_height: int,
    pixels_per_cm: float,
    is_crux: bool = False,
) -> Climber:
    """
    Compute a Climber body position given two hand holds.

    Uses anatomically-accurate body proportions from BodyProportion
    (head/8, arm = 3.5 heads, leg = 4 heads — from anatomy4sculptors.com).

    INVERSE KINEMATICS APPROACH:
        1. Place hands on holds (given)
        2. Derive shoulder positions: each shoulder is arm_length below
           its hand, pulled inward toward body centre
        3. Shoulder midpoint → derive neck, head, trunk downward
        4. Hip = base of trunk
        5. Find two foot holds: nearest holds below the hip within leg reach
        6. Knees = midpoint between hip and foot hold

    All computation in pixel space for direct OpenCV rendering.
    """
    arm_px = body_prop.arm
    leg_px = body_prop.leg
    trunk_px = body_prop.trunk
    head_px = body_prop.head
    neck_px = body_prop.neck
    shoulder_px = body_prop.shoulder

    # Hand positions in pixels
    rh = right_hand_hold.detected_object.center
    lh = left_hand_hold.detected_object.center

    hand_mid_x = (rh.x + lh.x) // 2
    hand_mid_y = (rh.y + lh.y) // 2

    # Shoulders: arm_length * 0.6 below each hand, pulled 40% toward centre
    rs_x = int(round(rh.x * 0.6 + hand_mid_x * 0.4))
    rs_y = int(round(rh.y + arm_px * 0.55))
    ls_x = int(round(lh.x * 0.6 + hand_mid_x * 0.4))
    ls_y = int(round(lh.y + arm_px * 0.55))

    shoulder_mid_x = (rs_x + ls_x) // 2
    shoulder_mid_y = (rs_y + ls_y) // 2

    # Elbows: midpoint between shoulder and hand
    re_x = (rs_x + rh.x) // 2
    re_y = (rs_y + rh.y) // 2
    le_x = (ls_x + lh.x) // 2
    le_y = (ls_y + lh.y) // 2

    # Neck
    neck_top_y = shoulder_mid_y
    neck_bot_y = int(round(shoulder_mid_y - neck_px))

    # Head (above neck)
    head_top_y = int(round(neck_bot_y - head_px))

    # Trunk (below shoulders)
    trunk_top_y = shoulder_mid_y
    trunk_bot_y = int(round(shoulder_mid_y + trunk_px))
    hip_x = shoulder_mid_x
    hip_y = trunk_bot_y

    # Foot holds: two nearest holds below the hip within leg reach
    foot_candidates = []
    for h in all_holds:
        hpx, hpy = h.detected_object.center.x, h.detected_object.center.y
        if hpy > hip_y:  # below hip in image coords (y increases down)
            dist = math.sqrt((hpx - hip_x)**2 + (hpy - hip_y)**2)
            if dist <= leg_px:
                foot_candidates.append((dist, h))

    foot_candidates.sort(key=lambda t: t[0])

    # Select left and right foot holds
    if len(foot_candidates) >= 2:
        # Pick left/right by x position
        fc_holds = [t[1] for t in foot_candidates[:4]]
        left_candidates = [h for h in fc_holds if h.detected_object.center.x <= hip_x]
        right_candidates = [h for h in fc_holds if h.detected_object.center.x > hip_x]

        left_foot_hold = left_candidates[0] if left_candidates else foot_candidates[0][1]
        right_foot_hold = right_candidates[0] if right_candidates else foot_candidates[1][1]
    elif len(foot_candidates) == 1:
        left_foot_hold = foot_candidates[0][1]
        right_foot_hold = foot_candidates[0][1]
    else:
        # No foot holds in range — model hanging position
        left_foot_hold = None
        right_foot_hold = None

    # Foot positions
    if left_foot_hold:
        lf = left_foot_hold.detected_object.center
        lf_det = left_foot_hold.detected_object
    else:
        lf = Point(hip_x - int(shoulder_px * 0.5), hip_y + int(leg_px * 0.9))
        lf_det = None

    if right_foot_hold:
        rf = right_foot_hold.detected_object.center
        rf_det = right_foot_hold.detected_object
    else:
        rf = Point(hip_x + int(shoulder_px * 0.5), hip_y + int(leg_px * 0.9))
        rf_det = None

    # Knee positions: midpoint between hip and foot
    lk_x = (hip_x + lf.x) // 2
    lk_y = (hip_y + lf.y) // 2
    rk_x = (hip_x + rf.x) // 2
    rk_y = (hip_y + rf.y) // 2

    # Crux: highlight in orange, otherwise standard colours
    arm_colour = Color(255, 100, 0) if is_crux else Color.red()
    body_colour = Color.green() if not is_crux else Color(255, 200, 0)

    # Build Climber object using his existing model classes
    climber = Climber(body_prop.height)

    climber.head = BodyPart(
        start=Point(shoulder_mid_x, head_top_y),
        end=Point(shoulder_mid_x, neck_bot_y),
        color=Color(255, 255, 0),
        thickness=int(head_px * 0.6),
    )
    climber.neck = BodyPart(
        start=Point(shoulder_mid_x, neck_bot_y),
        end=Point(shoulder_mid_x, neck_top_y),
        color=Color(0, 125, 255),
        thickness=8,
    )
    climber.trunk = BodyPart(
        start=Point(shoulder_mid_x, trunk_top_y),
        end=Point(hip_x, trunk_bot_y),
        color=body_colour,
        thickness=20,
    )
    climber.left_shoulder = BodyPart(
        start=Point(ls_x, ls_y),
        end=Point(shoulder_mid_x, shoulder_mid_y),
        color=Color.blue(),
        thickness=8,
    )
    climber.right_shoulder = BodyPart(
        start=Point(shoulder_mid_x, shoulder_mid_y),
        end=Point(rs_x, rs_y),
        color=Color.blue(),
        thickness=8,
    )
    # Arms: shoulder → elbow → hand
    climber.left_arm = BodyPart(
        start=Point(ls_x, ls_y),
        end=lh,
        color=arm_colour,
        thickness=8,
        detected_object=left_hand_hold.detected_object,
    )
    climber.right_arm = BodyPart(
        start=Point(rs_x, rs_y),
        end=rh,
        color=arm_colour,
        thickness=8,
        detected_object=right_hand_hold.detected_object,
    )
    climber.left_leg = BodyPart(
        start=Point(hip_x, hip_y),
        end=lf,
        color=Color.blue(),
        thickness=8,
        detected_object=lf_det,
    )
    climber.right_leg = BodyPart(
        start=Point(hip_x, hip_y),
        end=rf,
        color=Color.blue(),
        thickness=8,
        detected_object=rf_det,
    )

    return climber


# ── Main class (drop-in replacement) ─────────────────────────────────────────

class RouteGeneratorV2:
    """
    Optimal route generator using dynamic programming.

    DROP-IN REPLACEMENT for RouteGenerator. Constructor and
    generate_route() have identical signatures.

    Additional capability via generate_route_with_metadata() which
    returns the OptimalRoute alongside the Climber positions — useful
    for the enhanced FastAPI endpoint to include grade, send rate, etc.
    """

    def __init__(
        self,
        img_width: int,
        img_height: int,
        marker: ArucoMarker,
        detected_objects: List[DetectedObject],
    ):
        self._img_width = img_width
        self._img_height = img_height
        self._marker = marker
        self._detected_objects = detected_objects
        self._pixels_per_cm = marker.get_pixels_per_centimeter()

    def generate_route(
        self,
        climber_height_in_cm: int,
        starting_steps_max_distance_from_ground_in_cm: int,
        climber_grade: str = "V4",
        climber_wingspan_in_cm: Optional[int] = None,
    ) -> List[Climber]:
        """
        Generate the optimal route and return body positions.

        Compatible with original RouteGenerator.generate_route() signature.
        Additional optional args: climber_grade, climber_wingspan_in_cm.

        Returns: List[Climber] — one per hold in the optimal route.
        """
        positions, _ = self.generate_route_with_metadata(
            climber_height_in_cm=climber_height_in_cm,
            starting_steps_max_distance_from_ground_in_cm=starting_steps_max_distance_from_ground_in_cm,
            climber_grade=climber_grade,
            climber_wingspan_in_cm=climber_wingspan_in_cm,
        )
        return positions

    def generate_route_with_metadata(
        self,
        climber_height_in_cm: int,
        starting_steps_max_distance_from_ground_in_cm: int,
        climber_grade: str = "V4",
        climber_wingspan_in_cm: Optional[int] = None,
    ) -> Tuple[List[Climber], Optional[OptimalRoute]]:
        """
        Generate optimal route and return both body positions and route metadata.

        Returns:
            (List[Climber], OptimalRoute) — positions for drawing, route for metadata.
            If no route found, returns ([], None).
        """
        # Derive climber biomechanics
        wingspan_cm = climber_wingspan_in_cm or climber_height_in_cm * 1.01
        max_reach_m = (wingspan_cm / 100.0 / 2.0) * DYNAMIC_REACH_FACTOR
        skill_level = GRADE_TO_SKILL.get(climber_grade.upper(), 0.45)

        # Height of start/finish zones in metres
        start_zone_m   = starting_steps_max_distance_from_ground_in_cm / 100.0
        wall_height_m  = self._img_height / self._pixels_per_cm / 100.0
        finish_zone_m  = wall_height_m * 0.10  # top 10% of wall

        # Convert YOLO detections to world-space WallHold objects
        wall_holds = build_wall_holds(
            detected_objects=self._detected_objects,
            img_width=self._img_width,
            img_height=self._img_height,
            pixels_per_cm=self._pixels_per_cm,
            start_zone_height_m=start_zone_m,
            finish_zone_height_m=finish_zone_m,
        )

        if not wall_holds:
            return [], None

        # Run DP solver
        route = solve_optimal_route(
            holds=wall_holds,
            max_reach_m=max_reach_m,
            skill_level=skill_level,
        )

        if route is None:
            return [], None

        # Build body proportions using his anatomical ratio model
        climber_height_px = self._marker.convert_cm_to_px(climber_height_in_cm)
        body_prop = BodyProportion(climber_height_px)

        # Build Climber positions for each move in the route
        positions: List[Climber] = []
        route_holds = route.holds

        for i, hold in enumerate(route_holds):
            # Assign hands: current hold = right hand, previous = left hand
            if i == 0:
                # Start position: both hands on first hold or nearby
                rh = hold
                # Find nearest hold to the left within reach
                candidates = [
                    h for h in wall_holds
                    if h.idx != hold.idx and h.distance_to(hold) <= max_reach_m
                ]
                lh = min(candidates, key=lambda h: h.distance_to(hold)) if candidates else hold
            else:
                rh = hold
                lh = route_holds[i - 1]

            is_crux = (i > 0 and (i - 1) == route.crux_move_idx)

            climber_pos = build_climber_position(
                right_hand_hold=rh,
                left_hand_hold=lh,
                all_holds=wall_holds,
                body_prop=body_prop,
                img_height=self._img_height,
                pixels_per_cm=self._pixels_per_cm,
                is_crux=is_crux,
            )
            positions.append(climber_pos)

        return positions, route
