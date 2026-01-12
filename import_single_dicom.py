import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # Needed for resizing
import pydicom
import torchio as tio
from pathlib import Path
from diffdrr.visualization import plot_drr
import matplotlib.pyplot as plt


# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
patient_folder = Path.cwd() / "SJ"
ap_xray_path = patient_folder / "AP.dcm"

# --- 1. Load Data ---
# tio.ScalarImage loads data as 4D: (Channels, W, H, D) -> (1, 1024, 1024, 1)
ap_xray_obj = tio.ScalarImage(ap_xray_path)
raw_tensor = ap_xray_obj.data

# plt.imshow(raw_tensor,cmap='gray')
plot_drr(raw_tensor,"RAW AP",False,None)

plt.savefig(patient_folder/ "RAW_AP.png")


