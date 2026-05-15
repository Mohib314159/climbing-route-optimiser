# Climbing Crux Model — V2 Route Generator

Drop-in upgrade for [climbingcrux_model](https://github.com/mkurc1/climbingcrux_model).

## What changes

Two files replace two existing files. Everything else (YOLO detection, ArUco calibration, OpenCV rendering, Docker setup) stays identical.

| Original | Replaced by |
|----------|-------------|
| `src/route_generator.py` | `src/route_generator_v2.py` |
| `main.py` | `main_v2.py` |

## Installation

```bash
# Clone the original repo
git clone https://github.com/mkurc1/climbingcrux_model
cd climbingcrux_model

# Copy the two new files in
cp route_generator_v2.py src/
cp main_v2.py .

# Run
uvicorn main_v2:app --reload --port 8000
```

## API changes

### `POST /boulder/generate` (same endpoint, new params)

```bash
# Original — no climber personalisation
curl -X POST /boulder/generate -F "file=@wall.jpg"

# V2 — personalised to climber
curl -X POST "/boulder/generate?climber_height_cm=182&climber_grade=V5&climber_wingspan_cm=188" \
  -F "file=@wall.jpg"
```

Returns: same as before annotated PNG image with stick figures. Crux moves highlighted orange.

### `POST /boulder/analyse` (new endpoint)

```bash
curl -X POST "/boulder/analyse?climber_height_cm=182&climber_grade=V5" \
  -F "file=@wall.jpg"
```

Returns JSON:
```json
{
  "route": {
    "n_moves": 9,
    "total_cost": 4.823,
    "grade_estimate": "V5",
    "crux_move_index": 6,
    "hold_indices": [3, 8, 12, 19, 24, 31, 38, 44, 51]
  },
  "climber": {
    "height_cm": 182,
    "wingspan_cm": 188,
    "grade": "V5",
    "send_rate": 0.341
  },
  "annotated_image_png_b64": "..."
}
```

## What's different:

### Original route generation
At each step, selects foot and hand holds randomly from candidates within a spatial zone. Produces a different route on every call. No difficulty model: all holds are treated equally.

### V2 route generation

1. Difficulty estimation
Each hold gets a difficulty score in [0, 1] estimated from:
- Bounding box size (smaller = harder)
- Height on wall (higher = harder)
- Hold class (volumes easier than individual holds)

2. Dynamic programming (optimal route)
The wall is modelled as a DAG. Nodes = holds, edges = reachable moves (within climber's reach, upward only). Edge weight = `0.7 × difficulty + 0.3 × normalised_distance`. A single forward DP pass finds the minimum-cost path. Deterministic — same input always gives same output.

3. Personalised biomechanics
Max reach = `(wingspan / 2) × 1.15` (15% dynamic reach bonus). A taller climber with longer wingspan can reach holds a shorter climber cannot. The difficulty model scales relative to the climber's grade.

4. Monte Carlo send rate
10,000 simulated attempts. Each move succeeds with probability `sigmoid(8 × (skill − move_cost))` — the Item Response Theory model. Reports what percentage of attempts complete the route.

5. Anatomically-accurate IK body positions
Body joint positions are computed geometrically from hand hold positions using anatomy ratios (head/8, arm = 3.5 heads, leg = 4 heads). Crux moves highlighted in orange.

## Attribution

Original YOLO detection pipeline, ArUco calibration, and OpenCV rendering: [mkurc1/climbingcrux_model](https://github.com/mkurc1/climbingcrux_model), MIT License.

Route optimisation engine, difficulty model, Monte Carlo simulation, and IK body positioning: original work.
