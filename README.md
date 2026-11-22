# FormBack

Running form analysis from a single side-on video. Upload a clip and FormBack
measures your cadence, stride timing, ground contact, trunk lean, knee motion
and left/right balance, then explains the numbers in plain language.

FormBack runs on your computer. It does not upload your videos anywhere.

> FormBack provides general movement information estimated from video. It is
> not medical advice or a diagnosis.

## Quick start

### Option 1: Docker (recommended)

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/output:/app/output" \
  ghcr.io/alexisnewell/formback:latest
```

Then open **http://localhost:8000**. Results are saved in an `output` folder
next to where you ran the command. On Windows PowerShell, use `${PWD}` instead
of `$PWD`.

If you have this repository, `docker compose up` does the same thing.

### Option 2: Download the zip

1. Download the latest `formback-v*.zip` from the
   [Releases page](https://github.com/alexisnewell/FormBack/releases) and unzip it.
2. Install Python 3.11, 3.12 or 3.13 if you don't have it.
3. Start the app:
   - macOS / Linux: `./start.sh`
   - Windows: double-click `start.bat`
4. Open **http://localhost:8000**.

The first run downloads dependencies, which takes a few minutes and needs an
internet connection. Later runs start quickly. Press `Ctrl+C` to stop.

## Filming a good video

Quality in, quality out. For the most reliable numbers:

- Film from the **side**, with the whole body in frame from head to feet.
- Keep the camera **steady** (a tripod or a fixed spot is best), at about hip height.
- Capture **at least 10 strides per leg**, ideally 20 or more. Clips of 10 to 60
  seconds work best.
- Use good, even lighting and a plain background. Fitted clothing helps.
- Record the original video if you can. Screen recordings of other videos are
  re-compressed and tracked less accurately.
- Higher frame rates (60 fps or more) give better timing measurements.

## Using the app

1. Open http://localhost:8000.
2. Drop an **MP4** onto the page, or click **Choose video** (up to 500 MB).
3. Wait while it uploads and analyzes. This can take a minute or more depending
   on your computer and the clip length.
4. Read your results (below). Click **Analyze another video** to start over.

Not an MP4? Convert it with [ffmpeg](https://ffmpeg.org/):

```bash
ffmpeg -i clip.mov -c:v libx264 -crf 18 -pix_fmt yuv420p -an clip.mp4
```

Keep the original frame rate. If the clip is a screen recording (variable frame
rate), add `-r 60`.

## Reading your results

**Analysis quality** shows how trustworthy the run was: how often the body was
tracked, how many strides were valid, and any warnings. If the result is marked
low quality, the numbers are hidden or unreliable. Re-film using the tips above.

**Metrics tab**

- **Overall:** cadence (steps per minute), foot strike pattern, running
  direction, and clip length and frame rate.
- **Left vs. right:** the median for each leg, the spread (IQR), the number of
  strides measured (n), and the percentage difference between legs.
- **Confidence badges** (high, medium, low, unvalidated) show how much to trust
  each metric. Hover a badge for the reason. Treat low and unvalidated numbers
  as rough guides.

**Insights tab** turns the numbers into short notes, sorted from "worth a look"
to "looking good". They compare your values with typical ranges for recreational
runners (see [Where the ranges come from](#where-the-ranges-come-from)). Typical
does not mean ideal, and a difference is not a diagnosis. If you have pain or an
injury, see a physiotherapist or doctor.

Tips for reading it:

- Compare clips at a **similar pace**. Cadence, contact time and bounce all
  change with speed.
- Film a **second clip** before reading much into any single number, especially
  left/right differences.

## Where your files go

Every analysis writes files to the `output` folder, named after your video:

| File | Contents |
|---|---|
| `<name>_landmarks.csv` | Body landmark positions for every frame |
| `<name>_events.csv` | Detected foot strikes and toe-offs |
| `<name>_strides.csv` | One row per stride with all metrics |
| `<name>_summary.json` | Medians, spreads, symmetry, cadence and quality |
| `<name>_debug.png` | Signals with detected events, for checking |
| `<name>_cycles.png` | All valid strides overlaid |

Landmarks are cached. If you re-run a video after changing settings or the pose
model, delete its `_landmarks.csv` (or use `--force` on the command line) so it
is extracted again.

## Command line

Run from the repository root with the virtual environment active:

```bash
# Full analysis of a video
python -m src.pipeline videos/clip.mp4

# Re-extract landmarks even if cached
python -m src.pipeline videos/clip.mp4 --force

# Landmarks only, plus an annotated video with the skeleton drawn on
python -m src.video_processor videos/clip.mp4 --annotate

# Joint-angle time series (knee, hip, ankle, trunk lean)
python -m src.biomechanics output/clip_landmarks.csv --video videos/clip.mp4

# Contact sheet of frames around each detected event (needs the annotated video)
python -m src.gait_inspection_sequences output/clip_annotated.mp4
```

## Development

```bash
# Backend (from the repository root)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# Frontend, with hot reload (in a second terminal)
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to port 8000
```

To test the single-app build the way users run it:

```bash
cd frontend && npm run build && cd ..
uvicorn api.main:app --port 8000     # UI and API both at http://localhost:8000
```

Run the tests with `pytest`.

## Troubleshooting

- **"Can't reach the FormBack API":** the backend isn't running, or it's on a
  different port. Start it and reload the page.
- **The page shows an error or nothing at `localhost:8000`:** the frontend hasn't
  been built. Run `npm run build` inside `frontend`.
- **Port 8000 is already in use:** use another port, for example
  `uvicorn api.main:app --port 8001`, or with Docker `-p 127.0.0.1:8001:8000`.
- **Results didn't change after I changed the code or model:** cached landmarks
  are being reused. Delete `output/<name>_landmarks.csv` and run again.
- **"Low-quality analysis":** the clip is too short, the framing is off, or the
  body was hard to track. See [Filming a good video](#filming-a-good-video).
- **`start.sh` says the Python version is wrong:** install Python 3.11, 3.12 or 3.13.
- **Analysis feels slow:** pose detection is the heavy step. Shorter clips and a
  smaller video resolution are faster.

## Where the ranges come from

The Insights tab uses simple rules of thumb based on published studies of
recreational runners. They show what is typical, not what is ideal, and they
vary with speed, age, sex and population. Key sources:

- Malisoux et al. (2023), *Reference Values and Determinants of Spatiotemporal
  and Kinetic Variables in Recreational Runners*, 860 runners.
  [Full text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10588426/)
- Malisoux et al. (2024), *Gait asymmetry in spatiotemporal and kinetic
  variables does not increase running-related injury risk*, 800+ runners.
  [Full text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10773390/)
- Teng & Powers (2014), *Sagittal plane trunk posture influences patellofemoral
  joint stress during running*, JOSPT.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/25155651/)

Measurements from a single video are less precise than lab equipment, which is
why several metrics carry a lower-confidence badge.

## Third-party software and license

FormBack uses MediaPipe and its Pose Landmarker model. See
[NOTICE](NOTICE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/) for details.