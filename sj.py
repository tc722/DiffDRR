import torch
import pydicom
import torchio as tio
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd

# DiffDRR imports
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert
from diffdrr.visualization import plot_drr
from diffdrr.metrics import MutualInformation
from diffdrr.registration import Registration
from diffdrr.visualization import plot_drr
from diffdrr.visualization import animate

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

def pose_from_carm(sid, tx, ty, alpha, beta, gamma):
    rot = torch.tensor([[alpha, beta, gamma]])
    xyz = torch.tensor([[tx, sid, ty]])
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")


def get_initial_parameters(true_params):
    alpha = true_params["alpha"] + np.random.uniform(-np.pi / 4, np.pi / 4)
    beta = true_params["beta"] + np.random.uniform(-np.pi / 4, np.pi / 4)
    gamma = true_params["gamma"] + np.random.uniform(-np.pi / 4, np.pi / 4)
    bx = true_params["bx"] + np.random.uniform(-30.0, 30.0)
    by = true_params["by"] + np.random.uniform(-30.0, 30.0)
    bz = true_params["bz"] + np.random.uniform(-30.0, 30.0)
    pose = pose_from_carm(by, bx, bz, alpha, beta, gamma).cuda()
    rotations, translations = pose.convert("euler_angles", "ZXY")
    return rotations, translations, pose

def get_unique_filename(base_name, ext, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    counter = 1
    
    while True:
        # Format as 3 digits: 001, 002, etc.
        filename = f"{output_dir}/{base_name}_{counter:03d}.{ext}"
        
        # If file doesn't exist, we found our unique name!
        if not os.path.exists(filename):
            return filename
        
        counter += 1


def animate_in_browser(drr, ground_truth, skip=1, max_length=None, duration=30):
    if max_length is not None:
        n = max_length - len(df)
        df = pd.concat([df, df.iloc[[-1] * n]]).iloc[::skip]
    else:
        pass

    out = animate(
        "<bytes>",
        df,
        drr,
        ground_truth=ground_truth,
        verbose=True,
        device=device,
        extension=".webp",
        duration=duration,
        parameterization="euler_angles",
        convention="ZXY",
    )
    # display(HTML(f"""<img src='{"data:img/gif;base64," + b64encode(out).decode()}'>"""))

    export_filename = get_unique_filename("optimization_animation", "webp")
    with open(export_filename,'wb') as f:
        f.write(out)

import torch
import torch.nn.functional as F

@torch.no_grad() # Automatically disables gradients for this entire function
def calc_mi_downsampled(ground_truth, estimate, mi_metric, scale_factor=0.25):
    """
    Downsamples images and calculates Mutual Information.
    Used for fast monitoring/logging without using memory for gradients.
    
    Args:
        ground_truth (Tensor): Target X-Ray (B, C, H, W)
        estimate (Tensor): Predicted DRR (B, C, H, W)
        mi_metric (MutualInformation): The metric object
        scale_factor (float): 0.25 means resize to 1/4th width/height (1/16th pixels).
    """
    
    # 1. Downsample both images using Bilinear Interpolation
    # align_corners=False is standard for image resizing
    gt_small = F.interpolate(
        ground_truth, scale_factor=scale_factor, mode='bilinear', align_corners=False
    )
    est_small = F.interpolate(
        estimate, scale_factor=scale_factor, mode='bilinear', align_corners=False
    )

    # 2. Normalize (Crucial step repeated because max/min changes after resizing)
    gt_min, gt_max = gt_small.min(), gt_small.max()
    est_min, est_max = est_small.min(), est_small.max()
    
    # Safe normalization for Ground Truth
    if gt_max - gt_min > 1e-6:
        gt_norm = (gt_small - gt_min) / (gt_max - gt_min)
    else:
        gt_norm = torch.zeros_like(gt_small)

    # Safe normalization for Estimate
    if est_max - est_min > 1e-6:
        est_norm = (est_small - est_min) / (est_max - est_min)
    else:
        est_norm = torch.zeros_like(est_small)

    # 3. Calculate Score
    return mi_metric(gt_norm, est_norm).item()

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
print(f"Self-MI Check (Target vs Target): {calc_mi_downsampled(target_xray, target_xray, mi_metric, scale_factor=0.25).item():.5f}")

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
    "bx": 0.0,    "by": sdd, "bz": 0.0      # Translations (mm)
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
plot_drr(1.0 -pred_drr)
plt.savefig(patient_dir / "predicted_mri_drr.png")
plt.close()

# -----------------------------------------------------------------------------
# 5. Final Metric Calculation
# -----------------------------------------------------------------------------


initial_score = calc_mi_downsampled(target_xray, 1.0 - pred_drr, mi_metric,scale_factor=0.25)

print(f"Initial MI Score (Target vs Prediction): {initial_score.item():.5f}")

def optimize_lbfgs(
    reg: Registration,
    ground_truth,
    mi_metric,       # Pass your MI object explicitly
    n_itrs=100,
    lr=1.0,          # L-BFGS usually works well with lr=1.0
    tol=1e-4,        # Stop if improvement < 0.0001
    patience=5       # Wait 5 steps to confirm plateau
):
    # L-BFGS with Strong Wolfe line search (helps find best step size automatically)
    optimizer = torch.optim.LBFGS(
        reg.parameters(), 
        lr=lr, 
        line_search_fn="strong_wolfe"
    )
    
    losses = []
    patience_counter = 0
    
    # Define Closure
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
            
        # 1. Forward Pass (Generate DRR)
        estimate = reg()
        
        # 2. Normalize (Crucial for MI)
        est_min, est_max = estimate.min(), estimate.max()
        if est_max - est_min != 0:
            est_norm = (estimate - est_min) / (est_max - est_min)
        else:
            est_norm = torch.zeros_like(estimate)

        # 3. Calculate Loss (Negative MI)
        # We want to MAXIMIZE MI, so we MINIMIZE Negative MI
        loss = -calc_mi_downsampled(ground_truth, 1.0 -est_norm, mi_metric, scale_factor=0.25)
        
        if loss.requires_grad:
            loss.backward()
            
        return loss

    # Optimization Loop
    print(f"Starting Optimization (Max Iters: {n_itrs})")
    for i in range(n_itrs):
        # Step returns the loss value from the closure
        loss_tensor = optimizer.step(closure)
        current_mi = -loss_tensor.item() # Flip sign back for readability
        losses.append(current_mi)
        
        print(f"Iter {i+1}: MI = {current_mi:.5f}")

        # --- Early Stopping Logic ---
        if i > 0:
            # Check improvement over previous step
            improvement = current_mi - losses[-2]
            
            if improvement < tol:
                patience_counter += 1
            else:
                patience_counter = 0 # Reset if we improved
            
            if patience_counter >= patience:
                print(f"Converged! Improvement < {tol} for {patience} steps.")
                break

    return losses

registration = Registration(
    drr_projector,
    test_rotations.clone(),
    test_translations.clone(),
    parameterization="euler_angles",
    convention="ZXY")

params_lbfgs = optimize_lbfgs(
    registration,
    target_xray,
    mi_metric,
    n_itrs=100,
    lr=1.0,
    tol=1e-4,
    patience=5
)

animate_in_browser(params_lbfgs, target_xray, skip=1, max_length=100, duration=30)

