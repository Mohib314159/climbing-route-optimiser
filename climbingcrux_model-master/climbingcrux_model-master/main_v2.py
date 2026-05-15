"""
main_v2.py — Enhanced FastAPI application.

REPLACES main.py in climbingcrux_model.

Changes from original:
    1. /boulder/generate now accepts climber_grade and climber_wingspan_cm
       as optional query parameters
    2. New endpoint /boulder/analyse returns JSON metadata (grade estimate,
       send rate, crux move, route hold sequence) alongside the image
    3. RouteGeneratorV2 replaces RouteGenerator — all logic is identical
       from the API caller's perspective

USAGE:
    uvicorn main_v2:app --reload --port 8000
    # Swagger docs: http://localhost:8000/docs
"""

import io
import cv2
import imutils
import numpy as np
import base64

from fastapi import FastAPI, UploadFile, HTTPException, Query, status
from starlette.responses import StreamingResponse, JSONResponse

from src import config, image_utils, objects_detector
from src.aruco_marker import ArucoMarker
from src.route_generator_v2 import RouteGeneratorV2
from src.colour_filter import (
    classify_all_holds, filter_by_colour,
    get_colour_summary, draw_colour_annotations,
)

app = FastAPI(
    title="Climbing Crux Route Generator v2",
    description=(
        "Optimal route generation using dynamic programming and Monte Carlo simulation. "
        "Drop-in upgrade for climbingcrux_model with personalised biomechanics."
    ),
    version="2.0.0",
)


@app.post("/boulder/generate")
async def generate_boulder(
    file: UploadFile,
    climber_height_cm: int = Query(default=170, description="Climber height in cm"),
    climber_wingspan_cm: int = Query(default=None, description="Wingspan in cm (optional, estimated from height if omitted)"),
    climber_grade: str = Query(default="V4", description="Current grade e.g. V0, V4, V8"),
) -> StreamingResponse:
    """
    Generate an optimal boulder route from an image.

    Returns annotated image with stick figure body positions at each hold.
    Crux moves are highlighted in orange.
    """
    contents = await file.read()
    validate_file(file, contents)

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=1216)

    try:
        marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No ArUco marker detected")

    detected_objects = objects_detector.detect(img)

    generator = RouteGeneratorV2(
        img_width=img.shape[1],
        img_height=img.shape[0],
        marker=marker,
        detected_objects=detected_objects,
    )

    positions = generator.generate_route(
        climber_height_in_cm=climber_height_cm,
        starting_steps_max_distance_from_ground_in_cm=config.STARTING_STEPS_MAX_DISTANCE_FROM_GROUND_IN_CM,
        climber_grade=climber_grade,
        climber_wingspan_in_cm=climber_wingspan_cm,
    )

    if not positions:
        raise HTTPException(status_code=404, detail="No route found — check hold density or reach parameters")

    for climber_position in positions:
        img = image_utils.draw_climber(
            img=img,
            climber=climber_position,
            draw_labels=False,
            draw_centers=False,
        )

    _, im_png = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.post("/boulder/analyse")
async def analyse_boulder(
    file: UploadFile,
    climber_height_cm: int = Query(default=170),
    climber_wingspan_cm: int = Query(default=None),
    climber_grade: str = Query(default="V4"),
) -> JSONResponse:
    """
    Full analysis: optimal route + grade estimate + send rate + crux + annotated image.

    Returns JSON with:
        - route metadata (grade, send_rate, n_moves, crux_move)
        - annotated image as base64-encoded PNG
    """
    contents = await file.read()
    validate_file(file, contents)

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=1216)

    try:
        marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)
    except ValueError:
        raise HTTPException(400, "No ArUco marker detected")

    detected_objects = objects_detector.detect(img)

    generator = RouteGeneratorV2(
        img_width=img.shape[1],
        img_height=img.shape[0],
        marker=marker,
        detected_objects=detected_objects,
    )

    positions, route = generator.generate_route_with_metadata(
        climber_height_in_cm=climber_height_cm,
        starting_steps_max_distance_from_ground_in_cm=config.STARTING_STEPS_MAX_DISTANCE_FROM_GROUND_IN_CM,
        climber_grade=climber_grade,
        climber_wingspan_in_cm=climber_wingspan_cm,
    )

    if not positions or route is None:
        raise HTTPException(404, "No route found")

    for climber_position in positions:
        img = image_utils.draw_climber(img=img, climber=climber_position, draw_labels=False, draw_centers=False)

    # Overlay grade + send rate text on image
    _draw_route_info(img, route, climber_grade)

    _, im_png = cv2.imencode(".png", img)
    img_b64 = base64.b64encode(im_png.tobytes()).decode("utf-8")

    return JSONResponse({
        "route": {
            "n_moves": route.n_moves,
            "total_cost": round(route.total_cost, 3),
            "move_costs": [round(c, 3) for c in route.move_costs],
            "grade_estimate": route.grade_estimate,
            "crux_move_index": route.crux_move_idx,
            "hold_indices": [h.idx for h in route.holds],
        },
        "climber": {
            "height_cm": climber_height_cm,
            "wingspan_cm": climber_wingspan_cm or int(climber_height_cm * 1.01),
            "grade": climber_grade,
            "send_rate": round(route.send_rate, 3),
        },
        "annotated_image_png_b64": img_b64,
    })


@app.post("/wall/colours")
async def detect_wall_colours(file: UploadFile) -> JSONResponse:
    """
    Detect all hold colours on the wall without generating a route.

    Useful first step — call this to discover which colours are on the wall
    before calling /boulder/generate with a specific route_colour.

    Returns: colour counts + annotated image showing each hold's detected colour.
    """
    contents = await file.read()
    validate_file(file, contents)

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=1216)

    detected_objects = objects_detector.detect(img)
    annotated = classify_all_holds(img, detected_objects)
    summary = get_colour_summary(annotated)

    annotated_img = draw_colour_annotations(img, annotated)
    _, im_png = cv2.imencode(".png", annotated_img)
    img_b64 = base64.b64encode(im_png.tobytes()).decode("utf-8")

    return JSONResponse({
        "colours_detected": summary,
        "total_holds": len(detected_objects),
        "annotated_image_png_b64": img_b64,
    })


@app.post("/boulder/generate-colour")
async def generate_boulder_colour(
    file: UploadFile,
    route_colour: str = Query(..., description="Colour of the route holds e.g. red, blue, yellow, green, purple, orange"),
    climber_height_cm: int = Query(default=170),
    climber_wingspan_cm: int = Query(default=None),
    climber_grade: str = Query(default="V4"),
) -> StreamingResponse:
    """
    Generate optimal route using ONLY holds of the specified colour.

    This is how real climbing works — routes are colour-coded.
    Call /wall/colours first to see which colours are on the wall.
    """
    contents = await file.read()
    validate_file(file, contents)

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=1216)

    try:
        marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)
    except ValueError:
        raise HTTPException(400, "No ArUco marker detected")

    # Detect all holds, classify colours, filter to target colour
    all_objects = objects_detector.detect(img)
    annotated_holds = classify_all_holds(img, all_objects)
    colour_objects = filter_by_colour(annotated_holds, route_colour)

    if len(colour_objects) < 3:
        raise HTTPException(404, f"Only {len(colour_objects)} '{route_colour}' holds found — not enough for a route. "
                                 f"Available colours: {get_colour_summary(annotated_holds)}")

    # Draw faded non-route holds first
    img = draw_colour_annotations(img, annotated_holds, target_colour=route_colour)

    generator = RouteGeneratorV2(
        img_width=img.shape[1],
        img_height=img.shape[0],
        marker=marker,
        detected_objects=colour_objects,  # only the filtered colour
    )

    positions, route = generator.generate_route_with_metadata(
        climber_height_in_cm=climber_height_cm,
        starting_steps_max_distance_from_ground_in_cm=config.STARTING_STEPS_MAX_DISTANCE_FROM_GROUND_IN_CM,
        climber_grade=climber_grade,
        climber_wingspan_in_cm=climber_wingspan_cm,
    )

    if not positions:
        raise HTTPException(404, f"No complete route found through '{route_colour}' holds")

    for climber_position in positions:
        img = image_utils.draw_climber(img=img, climber=climber_position, draw_labels=False, draw_centers=False)

    if route:
        _draw_route_info(img, route, climber_grade)

    _, im_png = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.post("/boulder/compare")
async def compare_climbers(
    file: UploadFile,
    climber1_height_cm: int = Query(default=165, description="First climber height"),
    climber1_grade: str = Query(default="V3", description="First climber grade"),
    climber2_height_cm: int = Query(default=185, description="Second climber height"),
    climber2_grade: str = Query(default="V6", description="Second climber grade"),
    route_colour: str = Query(default=None, description="Optional: filter by colour"),
) -> JSONResponse:
    """
    Compare optimal routes for two different climbers on the same wall.

    Shows how height, wingspan, and grade affect route selection.
    Returns both routes overlaid on the same image in different colours.
    """
    contents = await file.read()
    validate_file(file, contents)

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=1216)

    try:
        marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)
    except ValueError:
        raise HTTPException(400, "No ArUco marker detected")

    all_objects = objects_detector.detect(img)

    # Optionally filter by colour
    if route_colour:
        annotated = classify_all_holds(img, all_objects)
        use_objects = filter_by_colour(annotated, route_colour)
    else:
        use_objects = all_objects

    results = []
    # Draw climber 1 (red tones) then climber 2 (blue tones)
    for i, (height, grade) in enumerate([
        (climber1_height_cm, climber1_grade),
        (climber2_height_cm, climber2_grade),
    ]):
        gen = RouteGeneratorV2(
            img_width=img.shape[1],
            img_height=img.shape[0],
            marker=marker,
            detected_objects=use_objects,
        )
        positions, route = gen.generate_route_with_metadata(
            climber_height_in_cm=height,
            starting_steps_max_distance_from_ground_in_cm=config.STARTING_STEPS_MAX_DISTANCE_FROM_GROUND_IN_CM,
            climber_grade=grade,
        )

        if positions and route:
            for pos in positions:
                img = image_utils.draw_climber(img=img, climber=pos, draw_labels=False, draw_centers=False)

            results.append({
                "climber": i + 1,
                "height_cm": height,
                "grade": grade,
                "route_grade_estimate": route.grade_estimate,
                "send_rate": round(route.send_rate, 3),
                "n_moves": route.n_moves,
                "hold_indices": [h.idx for h in route.holds],
                "shares_holds_with_other": None,  # filled below
            })

    # Check how many holds the two routes share
    if len(results) == 2:
        set1 = set(results[0]["hold_indices"])
        set2 = set(results[1]["hold_indices"])
        shared = len(set1 & set2)
        results[0]["shares_holds_with_other"] = shared
        results[1]["shares_holds_with_other"] = shared

    # Label each climber on image
    colours_text = [(0, 80, 220), (220, 80, 0)]
    for i, r in enumerate(results):
        label = f"C{i+1}: {r['height_cm']}cm {r['grade']} | {r['route_grade_estimate']} | {r['send_rate']:.0%} send"
        y_pos = 40 + i * 40
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (15, y_pos - th - 5), (15 + tw + 5, y_pos + 5), (0, 0, 0), -1)
        cv2.putText(img, label, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colours_text[i], 2)

    _, im_png = cv2.imencode(".png", img)
    img_b64 = base64.b64encode(im_png.tobytes()).decode("utf-8")

    return JSONResponse({
        "climbers": results,
        "annotated_image_png_b64": img_b64,
    })


def _draw_route_info(img, route, climber_grade: str) -> None:
    """Draw grade estimate and send rate onto the image."""
    text_lines = [
        f"Grade: {route.grade_estimate}",
        f"Send rate ({climber_grade}): {route.send_rate:.0%}",
        f"Moves: {route.n_moves}",
        f"Crux: move #{route.crux_move_idx + 1}",
    ]
    x, y = 20, 40
    for line in text_lines:
        # Dark background for readability
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += th + 15


def validate_file(file: UploadFile, contents: bytes) -> None:
    if file.content_type not in config.ACCEPTED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    if len(contents) > config.MAXIMUM_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
