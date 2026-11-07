import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAVELENGTH = 1550e-9
DX = 8e-6
N = 300
DISTANCE = 8e-3

SUB_SIZE = 212
OVERLAP = 124
assert N == 2 * SUB_SIZE - OVERLAP, "canvas must be 2*subnet - overlap"

INPUT_SIZE_RAW = 28
INPUT_SCALE = 4
INPUT_SIZE = INPUT_SIZE_RAW * INPUT_SCALE

FULL_SIZE = N * DX

SPREAD_SCALE = 0.15
LIMIT_SCALE = 0.6
SIGMA_SCALE = 0.015

DATA_PATH = "./data"

REGIONS = ["NW", "NE", "SW", "SE"]
REGION_TASKS = {
    "NW": "mnist",
    "NE": "fashion_mnist",
    "SW": "emnist",
    "SE": "synthetic",
}
N_CLASSES = 10
CANVAS_L2_SQ = float(SUB_SIZE * SUB_SIZE)

PIXEL_INIT_STD = 0.2
