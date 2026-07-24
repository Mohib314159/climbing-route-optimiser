# ClimbingCrux: Route Optimiser

Built on top of [mkurc1/climbingcrux_model](https://github.com/mkurc1/climbingcrux_model), which handles the computer vision side (YOLOv9 hold detection, ArUco distance calibration, OpenCV rendering). That part is untouched. What I replaced is the route generation: the original picks holds randomly, which means you get a different route every run and there's no sense of difficulty or optimality.

## What's new

Two files replace two from the original:

- `src/route_generator.py` → `src/route_generator_v2.py`
- `main.py` → `main_v2.py`

Everything else (Docker, ArUco, YOLO, the drawing code) stays the same.

## Setup

Clone the original repo first, then drop in the two new files:

```bash
git clone https://github.com/mkurc1/climbingcrux_model
cd climbingcrux_model
cp route_generator_v2.py src/
cp main_v2.py .
pip install fastapi uvicorn opencv-python python-dotenv imutils ultralytics python-multipart
uvicorn main_v2:app --reload --port 8000
```

Swagger docs at `http://localhost:8000/docs`.

## How the route generation works

**Original:** at each step, candidates within a radius zone are collected and one is picked with `np.random.choice`. No difficulty weighting, no optimisation - just a random walk upward.

**This version:**

Each detected hold gets a difficulty score between 0 and 1, estimated from three things: bounding box size (smaller holds are harder), height on the wall (route setters tend to place harder moves higher up), and hold class (volumes are generally easier than individual holds).

The wall is then treated as a directed acyclic graph: holds are nodes, and there's an edge from hold A to hold B if B is above A and within the climber's reach. Edge weight combines hold difficulty and move distance. A single forward DP pass (O(n²) in number of holds) finds the minimum-cost path from any start hold to any finish hold. Same input always gives the same output.

Reach is personalised: max reach = (wingspan / 2) × 1.15, where the 1.15 accounts for dynamic moves. A 190cm climber with long arms reaches holds that a 165cm climber can't, so their optimal routes will differ.

Once the route is found, 10,000 climbing attempts are simulated. Each move succeeds with probability sigmoid(8 × (skill - move_cost)), which is the same model used in psychometrics to relate question difficulty to student ability (Item Response Theory). The output is a send rate: the fraction of simulated attempts that top out.

Body position at each hold is estimated geometrically using standard anatomy ratios (arm ≈ 3.5 head lengths, leg ≈ 4 head lengths) to place shoulders, elbows, hips, knees, and feet. The crux move gets highlighted in orange.

## API

`POST /boulder/generate` - same as original but now accepts `climber_height_cm`, `climber_wingspan_cm`, and `climber_grade` as query params. Returns annotated PNG.

`POST /boulder/analyse` - returns JSON with grade estimate, send rate, crux move index, and base64-encoded annotated image.

`POST /boulder/generate-colour` - accepts a `route_colour` param (red, blue, yellow, green, purple, orange) and filters to only holds of that colour before solving. Useful on walls where routes are colour-coded.

`POST /boulder/compare` - takes two climber profiles and returns both optimal routes overlaid on the same image. Shows how height and grade affect which holds get used.

`POST /wall/colours` - no route generation, just identifies which colours are on the wall and returns counts. Good first step before calling generate-colour.

## Attribution

YOLO detection, ArUco calibration, OpenCV rendering, and all original infrastructure: [mkurc1/climbingcrux_model](https://github.com/mkurc1/climbingcrux_model), MIT licence.

Route optimisation, difficulty estimation, Monte Carlo simulation, colour filtering, and IK body positioning: original work.
