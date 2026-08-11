"""All the constants used in this repo."""

from pathlib import Path

# This repository's directory
REPO_DIR = Path(__file__).parent

# This repository's main necessary folders
FILTERS_PATH = REPO_DIR / "filters"
KEYS_PATH = REPO_DIR / ".fhe_keys"
CLIENT_TMP_PATH = REPO_DIR / "client_tmp"
SERVER_TMP_PATH = REPO_DIR / "server_tmp"

# Create the necessary folders
KEYS_PATH.mkdir(exist_ok=True)
CLIENT_TMP_PATH.mkdir(exist_ok=True)
SERVER_TMP_PATH.mkdir(exist_ok=True)

# All the filters currently available in the demo
# Ordered roughly by fingerprint-pipeline stage:
#   preprocessing  →  enhancement  →  feature extraction
AVAILABLE_FILTERS = [
    # ── Preprocessing ──────────────────────────
    "black and white",
    "blur",
    # ── Enhancement / ridge analysis ───────────
    "sharpen",
    "ridge detection",
    "fingerprint enhance",
    # ── Directional feature extraction ─────────
    "sobel horizontal",
    "sobel vertical",
    "laplacian",
]

# The input images' shape. Images with different input shapes will be cropped
# and resized by Gradio.  100×100 keeps FHE key-generation and execution times
# reasonable while preserving enough ridge detail for fingerprint work.
INPUT_SHAPE = (100, 100)

# Retrieve the input examples directory
INPUT_EXAMPLES_DIR = REPO_DIR / "input_examples"
INPUT_EXAMPLES_DIR.mkdir(exist_ok=True)

# List of all image examples suggested in the demo
EXAMPLES = [str(image) for image in INPUT_EXAMPLES_DIR.glob("**/*")]

# Store the server's URL
SERVER_URL = "http://localhost:8000/"