
# Semantic Parking Detector

University project for **Image Processing and Computer Vision (IOC)**.

Detection of available and occupied parking spaces using the **PKLot dataset**, **Python**, **OpenCV**, and **Artificial Intelligence (YOLO)**.

The project uses a pretrained YOLO model for vehicle detection and computer vision algorithms to determine whether individual parking spaces are occupied.

The goal is to develop, improve, and evaluate parking occupancy detection algorithms using image processing and AI techniques.

> **Project status:** Initial implementation. Vehicle detection is available using YOLO. Parking occupancy detection requires parking-space coordinates to be configured for the input image.

---

## 1. Technologies

The following technologies are used:

| Technology | Purpose |
|---|---|
| Python 3.13 | Main programming language |
| PyCharm | Development environment |
| OpenCV | Image processing and computer vision |
| Ultralytics YOLO | AI-based vehicle detection |
| PyTorch | Deep learning framework |
| NumPy | Numerical operations and image masks |
| pytest | Automated testing |
| Git & GitHub | Version control and collaboration |

A dedicated GPU is not required to run the initial application. However, an NVIDIA GPU is recommended for training or fine-tuning larger AI models.

---

## 2. Prerequisites

Before setting up the project, install:

- **Python 3.13**
- PyCharm (Community or Professional)
- Git
- An internet connection for downloading dependencies and pretrained models

The following instructions are intended for **Windows 10/11 using PowerShell**.

### Verify Python installation

Open PowerShell and run:

```powershell
py -3.13 --version
```

Expected output:

```text
Python 3.13.x
```

Verify Git:

```powershell
git --version
```

---

## 3. Clone the Repository

Open PyCharm and select:

**Get from VCS → Git**

Enter the repository URL:

```text
https://github.com/andrei-mandroc/Semantic-Parking-Detector.git
```

Choose a local directory and click **Clone**.

Alternatively, clone using the terminal:

```powershell
git clone https://github.com/andrei-mandroc/Semantic-Parking-Detector.git
cd Semantic-Parking-Detector
```

Open the cloned project in PyCharm.

All following commands must be executed from the project root directory, containing `main.py` and `requirements.txt`.

---

## 4. Python Virtual Environment

Each developer should use a separate Python virtual environment.

The recommended Python version is **3.13**.

### Create the virtual environment

From the project root, run:

```powershell
py -3.13 -m venv .venv
```

This creates a `.venv` directory containing the project's Python interpreter and installed packages.

### Configure PyCharm Interpreter

Navigate to:

**File → Settings → Python → Interpreter**

Depending on the PyCharm version, select:

**Add Interpreter → Add Local Interpreter → Existing**

Select the interpreter:

```text
<project-directory>\.venv\Scripts\python.exe
```

Apply the configuration.

> If PyCharm has already created a Python 3.13 virtual environment, reuse the existing environment instead of creating another one.

The `.venv` directory is excluded from Git using `.gitignore`.

---

## 5. Install Dependencies

Open the terminal in PyCharm.

### Upgrade pip

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Install required libraries

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
opencv-python>=4.10,<5
numpy>=1.26,<3
ultralytics>=8.3,<9
pytest>=8,<10
```

Ultralytics installs the required AI dependencies, including PyTorch.

### Verify installation

Run:

```powershell
.\.venv\Scripts\python.exe -c "import cv2, torch, ultralytics; print('Setup OK')"
```

Expected output:

```text
Setup OK
```

### GPU Support (Optional)

For NVIDIA GPU acceleration, install a compatible CUDA-enabled version of PyTorch.

Official installation instructions:

https://pytorch.org/get-started/locally/

GPU configuration is optional for the initial development stage.

---

## 6. Project Structure

```text
Semantic-Parking-Detector/
│
├── configs/
│   └── parking_spaces.json
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── models/
│
├── outputs/
│
├── scripts/
│   └── README.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   └── __init__.py
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│   │
│   ├── parking/
│   │   ├── __init__.py
│   │   └── occupancy.py
│   │
│   └── visualization/
│       ├── __init__.py
│       └── render.py
│
├── tests/
│   └── test_occupancy.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Directory Responsibilities

| Directory | Description |
|---|---|
| `configs/` | Parking-space configurations and polygon coordinates |
| `data/raw/` | Original PKLot dataset |
| `data/processed/` | Preprocessed images and annotations |
| `data/samples/` | Sample images for local testing |
| `models/` | Trained AI models and weights |
| `outputs/` | Generated images and detection results |
| `scripts/` | Future training, preprocessing, and evaluation scripts |
| `src/data/` | Dataset loading and annotation parsing |
| `src/detection/` | AI-based vehicle detection |
| `src/parking/` | Parking occupancy classification algorithms |
| `src/visualization/` | Drawing detected vehicles and parking spaces |
| `tests/` | Automated tests |
| `main.py` | Application entry point |

Empty directories are preserved in Git through `.gitkeep` files.

---

## 7. Running the Application

### Step 1: Add a test image

Place a parking lot image inside:

```text
data/samples/
```

For example:

```text
data/samples/parking.jpg
```

### Step 2: Run vehicle detection

Execute:

```powershell
.\.venv\Scripts\python.exe main.py --image data/samples/parking.jpg
```

The application will:

1. Load the input image using OpenCV.
2. Initialize the pretrained YOLO model.
3. Detect vehicles in the image.
4. Evaluate configured parking spaces, if available.
5. Draw the detection results.
6. Save the annotated image.

The first execution may automatically download the pretrained `yolo11n.pt` model.

### Step 3: View the results

The annotated image is saved in:

```text
outputs/parking_annotated.jpg
```

The terminal displays the number of detected vehicles and, if parking spaces are configured, their occupancy statistics.

### Optional Parameters

```powershell
.\.venv\Scripts\python.exe main.py --image data/samples/parking.jpg --model yolo11n.pt --config configs/parking_spaces.json --threshold 0.15
```

Available arguments:

| Argument | Description |
|---|---|
| `--image` | Input image path (required) |
| `--model` | YOLO model weights |
| `--config` | Parking-space configuration file |
| `--threshold` | Minimum polygon coverage ratio for occupied classification |

The default occupancy threshold is `0.15` (15%).

> Without configured parking polygons, the application performs vehicle detection only.

### Inspect PKLot annotations

To visualize the parking-space polygons from a matching image and XML file without running YOLO:

```powershell
.\.venv\Scripts\python.exe -m scripts.visualize_annotations --image data/raw/PKLot/2013-04-14/2013-04-14_06_40_00.jpg --xml data/raw/PKLot/2013-04-14/2013-04-14_06_40_00.xml --mode geometry
```

Use `--mode ground-truth` to display the XML occupancy labels in green (free) and red (occupied). This view is for checking annotations and does not show AI predictions. Both modes save an image in `outputs/`.

---

## 8. Parking Space Configuration

Parking-space coordinates are stored in:

```text
configs/parking_spaces.json
```

Example:

```json
{
  "image_width": 1280,
  "image_height": 720,
  "spaces": [
    {
      "id": "A1",
      "polygon": [
        [50, 100],
        [150, 100],
        [160, 180],
        [40, 180]
      ]
    },
    {
      "id": "A2",
      "polygon": [
        [170, 100],
        [270, 100],
        [280, 180],
        [160, 180]
      ]
    }
  ]
}
```

Each parking space is represented by a polygon containing its image coordinates.

The coordinates in this example are illustrative and must be replaced with actual coordinates from the selected parking image.

Different camera views require different parking-space configurations.

### Occupancy Detection Algorithm

The initial occupancy algorithm works as follows:

1. Detect vehicles using YOLO.
2. Extract bounding boxes for detected vehicles.
3. Generate a binary mask for each parking-space polygon.
4. Calculate the intersection between each parking-space mask and the detected vehicle bounding boxes.
5. Calculate the coverage ratio.
6. Classify the parking space as free or occupied.

The coverage ratio is calculated using:

```text
coverage = overlapping_area / parking_space_area
```

The initial classification rule is:

```text
coverage >= threshold  → OCCUPIED
coverage < threshold   → FREE
```

This is an initial geometric heuristic and must be evaluated and improved using the PKLot dataset.

`classify_spaces` keeps the original polygon-overlap baseline with a configurable
threshold (default `0.15`). `coverage_scores` exposes each overlap ratio.
`classify_spaces_by_center` is a second rule that marks a space occupied when
the center of a vehicle box lies inside or on its polygon boundary. Both rules
use vehicle boxes and space geometry, never XML occupancy labels.

Large boxes and vehicles between spaces can produce incorrect associations.
Oblique views, occlusions, and vehicles missed by the detector can also cause
errors. The default threshold is a starting value, not a validated optimum.

Future versions should load the parking-space coordinates automatically from PKLot annotations.

---

## 9. PKLot Dataset

The project uses the **PKLot dataset**, containing parking lot images captured under different environmental conditions.

The dataset provides annotations that can be used to identify individual parking spaces and their occupancy status.

### Planned Dataset Processing

The dataset processing component will:

1. Read parking lot images.
2. Parse XML annotation files.
3. Extract parking-space coordinates.
4. Extract ground-truth occupancy labels.
5. Generate parking-space masks.
6. Prepare data for AI training and evaluation.

Original dataset files should be stored in:

```text
data/raw/
```

Processed data should be stored in:

```text
data/processed/
```

Dataset files are excluded from Git because of their size.

---

## 10. Running Tests

The project uses `pytest` for automated testing.

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run a specific test file:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_occupancy.py
```

Additional unit tests should be implemented as the project evolves.

Important components to test include:

- Polygon and bounding-box intersections
- Parking occupancy classification
- Threshold validation
- Dataset annotation parsing
- Prediction accuracy calculations

---

## 11. Development Roadmap

### Phase 1 — Project Setup

- Initialize the Python project.
- Configure PyCharm and Python 3.13.
- Install dependencies.
- Configure Git and GitHub.
- Verify YOLO vehicle detection.

### Phase 2 — Dataset Exploration

- Download and analyze PKLot.
- Understand XML annotations.
- Implement annotation parsing.
- Visualize parking polygons using OpenCV.

### Phase 3 — Parking Occupancy Detection

- Implement vehicle detection using YOLO.
- Associate vehicle detections with parking spaces.
- Develop geometric occupancy algorithms.
- Classify parking spaces as free or occupied.

### Phase 4 — Algorithm Improvements

- Optimize intersection and coverage calculations.
- Experiment with confidence thresholds.
- Explore semantic or instance segmentation.
- Investigate temporal filtering for sequential images.
- Improve robustness under different weather conditions.

### Phase 5 — AI Model Experiments

- Evaluate pretrained YOLO models.
- Experiment with model fine-tuning.
- Compare vehicle detection and segmentation approaches.
- Investigate custom occupancy classification models.

### Phase 6 — Evaluation

Evaluate the algorithms using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Training, validation, and testing data should be separated by capture session, day, or camera when appropriate to reduce data leakage.

Compare performance under different environmental conditions, including sunny, cloudy, and rainy weather.

---

## 12. Team Collaboration

The project is developed by two team members.

A suggested division of responsibilities is:

### AI and Model Development

- YOLO integration
- AI model selection
- Model training and fine-tuning
- Detection optimization
- AI model evaluation

### Computer Vision and Parking Logic

- PKLot annotation processing
- OpenCV preprocessing
- Parking-space polygon extraction
- Occupancy classification algorithms
- Result visualization

Both team members participate in integration, testing, optimization, and evaluation.

---

## 13. Git Workflow

Use separate branches for implementing new functionality.

### Create a feature branch

```powershell
git switch -c feature/pklot-parser
```

### Commit changes

```powershell
git add .
git commit -m "feat: add PKLot annotation parser"
```

### Push the branch

```powershell
git push -u origin feature/pklot-parser
```

Create a Pull Request on GitHub to review and merge changes into `main`.

### Important Git Rules

Do not commit:

- Python virtual environments
- Original PKLot dataset files
- Large AI model weights
- Generated detection results
- IDE-specific settings
- Temporary files

These files are excluded through `.gitignore`.

---

## 14. Troubleshooting

### Python 3.13 is not recognized

Verify the Python installation:

```powershell
py -3.13 --version
```

Ensure Python 3.13 is correctly installed and available through the Python launcher.

### ModuleNotFoundError

Verify the interpreter configured in PyCharm.

Reinstall dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Input image not found

Verify the image path and make sure the command is executed from the project root.

### YOLO model cannot be downloaded

Check the internet connection. The pretrained model must be downloaded before the first inference unless its weights already exist locally.

### Image dimensions do not match

Ensure `image_width`, `image_height`, and the parking polygons in the configuration correspond to the input image.

### GPU is not detected

GPU acceleration is optional for the starter application.

For NVIDIA acceleration, verify the installed PyTorch build and CUDA compatibility.

---

## 15. Project Repository

GitHub repository:

https://github.com/andrei-mandroc/Semantic-Parking-Detector

**University course:** Image Processing and Computer Vision (IOC)

**Project topic:** Detection of free and occupied parking spaces using the PKLot dataset, OpenCV, and AI.
