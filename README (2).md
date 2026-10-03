# Mathematical Face Sketch

A Streamlit computer-vision project that transforms a portrait into a mathematical face sketch using **MediaPipe Face Mesh, OpenCV contours, Fourier-based parametric curves, and an animated tracing pointer**.

> **Photo → facial geometry → mathematical strokes → equations → animated mathematical portrait**

## Demo concept

The interface is inspired by graphing/calculator-style mathematical art. Each extracted feature becomes an independent mathematical stroke. During animation, the current equation is displayed beside the graph while a pointer traces its curve.

## Features

- Portrait upload with EXIF orientation handling.
- MediaPipe Face Mesh for feature-first facial geometry.
- OpenCV Canny contours for selected additional facial/hair detail.
- Independent mathematical strokes for face contour, eyebrows, eyes, lips, nose, and image contours.
- Fourier-smoothed parametric curves for the animated drawing.
- Live equation panel showing the equation currently being traced.
- Smooth pointer animation at an intentionally fast tracing speed.
- Restrained professional multi-color palette for active equations.
- Graph-paper mathematical workspace.
- Replay tracing.
- GIF export.
- MP4 export when the local OpenCV build supports the codec.
- Export of generated equations as text for further exploration in Desmos.

## Project structure

```text
Mathematical-Face-Sketch/
├── app.py
├── math_engine.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── assets/
    └── demo.gif              # optional; add your own demo
```

## How it works

### 1. Portrait input

The application accepts JPG, JPEG, PNG, and WEBP images. EXIF orientation is corrected before processing.

### 2. Facial geometry

MediaPipe Face Mesh provides landmark geometry for major facial structures. The current engine explicitly defines landmark loops for:

- Face contour
- Left eyebrow
- Right eyebrow
- Left eye
- Right eye
- Lips
- Nose

The feature-first ordering is intentional so the portrait is not dominated by arbitrary edge lengths.

### 3. Additional contours

OpenCV Canny edge detection supplies additional contours inside a region around the detected face. Small/noisy contours are filtered before they become mathematical strokes.

### 4. Mathematical curves

Each stroke is normalized into a mathematical coordinate system. The engine then uses Fourier reconstruction to create a smooth sampled curve for the animation.

The displayed equations use a parametric Fourier representation:

```text
x(t) = a0 + a1 cos(t) + b1 sin(t) + ...
y(t) = c0 + c1 cos(t) + d1 sin(t) + ...

0 ≤ t ≤ 2π
```

### 5. Animated construction

The browser-side animation traces one mathematical stroke at a time:

1. The current equation becomes active.
2. Its professional accent color is shown.
3. The pointer moves along the reconstructed curve.
4. Completed strokes remain visible in graphite.
5. The equation counter advances until the portrait is complete.

This makes the **mathematical construction** itself the visual output rather than generating an AI replacement portrait.

## Installation

### Requirements

Recommended:

- Python 3.11
- Windows 10/11
- 64-bit Python environment

The project intentionally pins MediaPipe to `0.10.21` because the engine uses the legacy `mp.solutions.face_mesh` API.

### Clone this repository

```powershell
git clone https://github.com/Muhammad-Tayyab-official/Mathematical-Face-Sketch.git
cd Mathematical-Face-Sketch
```

### Create and activate the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## MediaPipe compatibility

The project uses:

```text
mediapipe==0.10.21
protobuf>=4.25.3,<5
numpy>=1.26,<2
```

Check the installation with:

```powershell
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

Expected:

```text
0.10.21
True
```

If MediaPipe was downloaded as a wheel manually on Python 3.11/Windows 64-bit, the expected filename pattern is:

```text
mediapipe-0.10.21-cp311-cp311-win_amd64.whl
```

## GitHub

This repository is ready to push after adding any optional demo assets.

```powershell
git init
git add .
git commit -m "Initial release: Mathematical Face Sketch"
git branch -M main
git remote add origin https://github.com/Muhammad-Tayyab-official/Mathematical-Face-Sketch.git
git push -u origin main
```

If the repository is already initialized and connected:

```powershell
git add .
git commit -m "Prepare Mathematical Face Sketch for GitHub"
git push
```

## Optional demo GIF

For a stronger GitHub landing page, add a short recording of the tracing animation:

```text
assets/demo.gif
```

Then place this in the README:

```markdown
![Mathematical Face Sketch Demo](assets/demo.gif)
```

Do not commit generated personal portrait exports or private images unless you intend them to be public.

## Troubleshooting

### `module 'mediapipe' has no attribute 'solutions'`

The installed MediaPipe version is not the version expected by this project.

```powershell
pip install "mediapipe==0.10.21"
```

### Protobuf `FieldDescriptor` error

Use protobuf 4.x:

```powershell
pip install "protobuf>=4.25.3,<5" --force-reinstall
```

### NumPy compatibility issue

Use NumPy 1.x:

```powershell
pip install "numpy>=1.26,<2" --force-reinstall
```

### `ImportError` involving functions from `math_engine`

Make sure `app.py` and `math_engine.py` are from this same repository version. Do not mix files from older prototypes.

### No face detected

Use a clear portrait with the face reasonably visible and preferably facing toward the camera.

### MP4 export unavailable

MP4 depends on codec support in the local OpenCV installation. GIF and equation export do not require MP4 codec support.

## License

MIT License. See `LICENSE`.
