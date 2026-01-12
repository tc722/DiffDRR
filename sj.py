import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import pydicom
import torchio as tio
from diffdrr.data import load_example_ct
from diffdrr.drr import DRR
from diffdrr.metrics import NormalizedCrossCorrelation2d
from diffdrr.metrics import MutualInformation
from diffdrr.pose import convert
from diffdrr.registration import Registration
from diffdrr.visualization import plot_drr
from diffdrr.data import read
from pathlib import Path

def read_single_dcm(dcm_path):
    dcm_ScalerImage = tio.ScalarImage(dcm_path)
    dcm_tensor = dcm_ScalerImage.data.permute(3,0,2,1)
    return dcm_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"


# Make the ground truth X-ray
SDD = 1020.0
HEIGHT = 100
DELX = 4.0

patient_folder = Path.cwd() / "SJ"

# import and read the AP dicom header
ap_xray_path = patient_folder / "AP.dcm"
ds = pydicom.dcmread(ap_xray_path)

ap_xray_column = ds.Columns
ap_xray_SDD = ds.DistanceSourceToDetector
ap_xray_pixel_size = ds.ImagerPixelSpacing[0]

total_size = ap_xray_column * ap_xray_pixel_size

raw_ap = read_single_dcm(ap_xray_path).to(device)
plot_drr(raw_ap)
plt.savefig(patient_folder /"raw_AP_DRR.png")

mri_volume_path = patient_folder / "MRI"

subject = read(mri_volume_path,
               orientation="AP")

true_params = {
    "sdr": ap_xray_SDD,
    "alpha": 0.0,
    "beta": 0.0,
    "gamma": 0.0,
    "bx": 0.0,
    "by": 850.0,
    "bz": 0.0,
}

drr = DRR(subject, sdd=ap_xray_SDD, height=ap_xray_column, delx=ap_xray_pixel_size, patch_size=250).to(device)
rotations = torch.tensor(
    [[true_params["alpha"], true_params["beta"], true_params["gamma"]]]
)
translations = torch.tensor([[true_params["bx"], true_params["by"], true_params["bz"]]])
gt_pose = convert(
    rotations, translations, parameterization="euler_angles", convention="ZXY"
).to(device)
est = drr(gt_pose)

plot_drr(est)
plt.savefig(patient_folder /"MRI_DRR.png")

# Initialize MI with calculated sigma
mi = MutualInformation(
    sigma=0.2, 
    num_bins=256, 
    epsilon=1e-10, 
    normalize=True
).to(device)

print("Initial MI=",mi(raw_ap,est))