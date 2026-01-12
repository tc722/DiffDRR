import torch
import pydicom
import torchio as tio
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F

# DiffDRR imports
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert
from diffdrr.visualization import plot_drr
from diffdrr.metrics import MutualInformation

# -----------------------------------------------------------------------------
# 1. Setup and Helper Functions
# -----------------------------------------------------------------------------

def load_and_normalize_dicom(dcm_path):
    """
    Reads a DICOM file using TorchIO, corrects dimensions, and normalizes to [0, 1].
    
    Args:
        dcm_path (Path): Path to the .dcm file.
        
    Returns:
        torch.Tensor: A tensor of shape (B, C, H, W) with values in [0, 1].
    """
    # Load image using TorchIO to handle I/O
    tio_image = tio.ScalarImage(dcm_path)
    
    # Permute dimensions to match DiffDRR expectation: (Batch, Channel, Height, Width)
    # .contiguous() is required for view/reshape operations in metrics later
    tensor = tio_image.data.permute(3, 0, 2, 1).contiguous()
    
    # --- Min-Max Normalization ---
    # Essential for Mutual Information, which expects inputs in the [0, 1] range.
    val_min = tensor.min()
    val_max = tensor.max()
    
    if val_max - val_min != 0:
        tensor = (tensor - val_min) / (val_max - val_min)
    else:
        tensor = torch.zeros_like(tensor)
        
    return tensor

def normalize_tensor(tensor):
    """
    Helper to normalize any tensor (e.g., generated DRR) to [0, 1].
    """
    val_min = tensor.min()
    val_max = tensor.max()
    if val_max - val_min != 0:
        return (tensor - val_min) / (val_max - val_min)
    return torch.zeros_like(tensor)

def downsample(tensor, size=(256, 256)):
    """Downsample image to save memory during MI calculation."""
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

# Select computation device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# -----------------------------------------------------------------------------
# 2. Initialization & Constants
# -----------------------------------------------------------------------------

# Paths
patient_dir = Path.cwd() / "SJ"
xray_path = patient_dir / "AP.dcm"
mri_dir = patient_dir / "MRI"

# Metric Initialization
# sigma=0.005 provides a sharp landscape for fine-tuning alignment.
# Ensure num_bins matches the bit-depth resolution you need (256 is standard).
mi_metric = MutualInformation(
    sigma=0.005, 
    num_bins=256, 
    epsilon=1e-10, 
    normalize=True
).to(device)

# -----------------------------------------------------------------------------
# 3. Load Ground Truth (Target X-Ray)
# -----------------------------------------------------------------------------

# Read DICOM metadata to get geometry constraints
dcm_meta = pydicom.dcmread(xray_path)
img_height = dcm_meta.Columns            # In DICOM, Columns corresponds to width/grid size
sdd = dcm_meta.DistanceSourceToDetector  # Source-to-Detector Distance
pixel_spacing = dcm_meta.ImagerPixelSpacing[0]

# Load the actual pixel data (Target)
target_xray = load_and_normalize_dicom(xray_path).to(device)

# Visualization
plot_drr(target_xray)
plt.savefig(patient_dir / "target_xray_drr.png")
plt.close()

# Sanity Check: Self-MI should be high (0.5 - 1.0)
print(f"Self-MI Check (Target vs Target): {mi_metric(target_xray, target_xray).item():.5f}")

# -----------------------------------------------------------------------------
# 4. DRR Generation (Prediction from MRI)
# -----------------------------------------------------------------------------

# Load 3D Volume
subject_mri = read(mri_dir, orientation="AP")

# Initialize DRR Projector with X-Ray geometry
drr_projector = DRR(
    subject_mri, 
    sdd=sdd, 
    height=img_height, 
    delx=pixel_spacing, 
    patch_size=250
).to(device)

# Define Test Pose Parameters (Arbitrary configuration for testing)
test_pose_params = {
    "alpha": 0.0, "beta": 0.0, "gamma": 0.0,  # Rotations (Euler angles)
    "bx": 0.0,    "by": 850.0, "bz": 0.0      # Translations (mm)
}

# Convert parameters to tensors expected by DiffDRR
test_rotations = torch.tensor([[
    test_pose_params["alpha"], test_pose_params["beta"], test_pose_params["gamma"]
]])
test_translations = torch.tensor([[
    test_pose_params["bx"], test_pose_params["by"], test_pose_params["bz"]
]])

# Create the Pose object (The spatial orientation for the simulation)
test_pose = convert(
    test_rotations, test_translations, 
    parameterization="euler_angles", 
    convention="ZXY"
).to(device)

# Generate Prediction and Apply Normalization
pred_drr = 1.0 - normalize_tensor(drr_projector(test_pose))

# Visualization
plot_drr(pred_drr)
plt.savefig(patient_dir / "predicted_mri_drr.png")
plt.close()

# -----------------------------------------------------------------------------
# 5. Final Metric Calculation
# -----------------------------------------------------------------------------

# Resize both images to 256x256 before calculating MI
# This reduces the memory requirement from 1GB to ~60MB
target_small = downsample(target_xray)
pred_small = downsample(pred_drr)

initial_score = mi_metric(target_small, pred_small)

print(f"Initial MI Score (Target vs Prediction): {initial_score.item():.5f}")