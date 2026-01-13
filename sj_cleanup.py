# %%
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pydicom
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# DiffDRR imports
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert
from diffdrr.visualization import plot_drr, animate
from diffdrr.metrics import MutualInformation
from diffdrr.registration import Registration

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# MEMORY SAVING SETTINGS
# 1 = Full Resolution (1024x1024) -> High Memory
# 4 = Quarter Resolution (256x256) -> Recommended (Fast & Low VRAM)
SUBSAMPLE_FACTOR = 4 

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def load_and_preprocess_target(dcm_path, target_size=None):
    """
    Reads DICOM, ensures float type (CRITICAL for interpolation), 
    resizes to target_size, and normalizes.
    """
    tio_image = tio.ScalarImage(dcm_path)
    
    # 1. Permute to (Batch, Channel, Height, Width)
    # Shape logic: TorchIO usually loads (C, W, H, D). 
    # We map D->Batch(0), C->Channel(1), H->H(2), W->W(3) or similar.
    # Assuming dcm is 2D slice: (1, W, H, 1) -> Permute(3, 0, 2, 1) -> (1, 1, H, W)
    tensor = tio_image.data.permute(3, 0, 2, 1).contiguous()
    
    # 2. FIX: Force Float Type (Solves the upsample_bilinear2d error)
    tensor = tensor.float()

    # 3. Resize immediately (Downsampling upfront)
    if target_size is not None:
        tensor = F.interpolate(
            tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )

    # 4. Normalize to [0, 1]
    val_min, val_max = tensor.min(), tensor.max()
    if val_max - val_min > 1e-6:
        tensor = (tensor - val_min) / (val_max - val_min)
    else:
        tensor = torch.zeros_like(tensor)
        
    return tensor

def normalize_tensor(tensor):
    """Helper to normalize any tensor to [0, 1]."""
    val_min, val_max = tensor.min(), tensor.max()
    if val_max - val_min > 1e-6:
        return (tensor - val_min) / (val_max - val_min)
    return torch.zeros_like(tensor)

def get_unique_filename(base_name, ext, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    counter = 1
    while True:
        filename = f"{output_dir}/{base_name}_{counter:03d}.{ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1


# -----------------------------------------------------------------------------
# 3. Load Ground Truth & Configure Geometry
# -----------------------------------------------------------------------------

patient_dir = Path.cwd() / "SJ"
xray_path = patient_dir / "AP.dcm"
mri_dir = patient_dir / "MRI"

# Read Metadata
dcm_meta = pydicom.dcmread(xray_path)
orig_height = dcm_meta.Columns
orig_pixel_size = dcm_meta.ImagerPixelSpacing[0]
sdd = dcm_meta.DistanceSourceToDetector

# Calculate New Scaled Geometry
new_height = int(orig_height / SUBSAMPLE_FACTOR)
new_pixel_size = orig_pixel_size * SUBSAMPLE_FACTOR

print(f"Original Size: {orig_height}x{orig_height}")
print(f"Optimization Size: {new_height}x{new_height}")

# Load Target (Already resized to 256x256 via helper)
target_xray = load_and_preprocess_target(
    xray_path, 
    target_size=(new_height, new_height)
).to(device)

# Visualize
plot_drr(target_xray)
plt.savefig(patient_dir / "target_xray_downsampled.png")
plt.close()

# -----------------------------------------------------------------------------
# 4. Initialize Components
# -----------------------------------------------------------------------------

# Initialize MI Metric
mi_metric = MutualInformation(
    sigma=0.01, 
    num_bins=256, 
    epsilon=1e-10, 
    normalize=True
).to(device)

# Initialize DRR Projector with DOWNSAMPLED geometry
# This ensures the DRR output matches the resized target automatically.
subject_mri = read(mri_dir, orientation="AP")
drr_projector = DRR(
    subject_mri, 
    sdd=sdd, 
    height=new_height,       # e.g., 256
    delx=new_pixel_size,     # e.g., 4x larger pixel size
    patch_size=250           
).to(device)

# Define Test Pose
test_pose_params = {
    "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
    "bx": 0.0,    "by": sdd, "bz": 0.0
}

test_rotations = torch.tensor([[
    test_pose_params["alpha"], test_pose_params["beta"], test_pose_params["gamma"]
]], device=device)

test_translations = torch.tensor([[
    test_pose_params["bx"], test_pose_params["by"], test_pose_params["bz"]
]], device=device)

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
plt.show()
plt.savefig(patient_dir / "predicted_mri_drr.png")
plt.close()


print(f"Initial MI = {mi_metric(target_xray, target_xray).item():.5f}")

# %%
# -----------------------------------------------------------------------------
# 5. Optimization
# -----------------------------------------------------------------------------

def optimize_registration(reg_module, ground_truth, metric, lr=1e1, n_itrs=100, tol=1e-4, patience=5):
    """
    Optimizes the registration with early stopping.
    
    Args:
        tol (float): Minimum improvement required to reset patience.
        patience (int): Number of iterations to wait before stopping if no improvement.
    """
    optimizer = torch.optim.LBFGS(
        reg_module.parameters(), 
        lr=lr, 
        line_search_fn="strong_wolfe"
    )
    
    losses = []
    patience_counter = 0
    print(f"Starting Optimization...")
    
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        
        # 1. Forward Pass
        estimate = reg_module()
        
        # 2. Normalize & Invert 
        est_norm = normalize_tensor(estimate)
        est_inverted = 1.0 - est_norm
        
        # 3. Calculate Loss 
        loss = -metric(ground_truth, est_inverted)
        
        if loss.requires_grad:
            loss.backward()
        return loss

    for i in range(n_itrs):
        # Step
        loss_val = optimizer.step(closure)
        current_mi = -loss_val.item() # Flip sign back for readability
        losses.append(current_mi)

        print(f"Iter {i+1}: MI = {current_mi:.5f}")

        # --- EARLY STOPPING LOGIC ---
        if i > 0:
            # Compare current MI with the previous step's MI
            previous_mi = losses[-2]
            improvement = current_mi - previous_mi
            
            # Check if improvement is negligible
            if improvement < tol:
                patience_counter += 1
                # Optional: Print status to know why it might stop soon
                # print(f"   Plateau detected ({improvement:.1e} < {tol}). Patience: {patience_counter}/{patience}")
            else:
                patience_counter = 0  # Reset if we make progress
            
            if patience_counter >= patience:
                print(f"Converged! Improvement < {tol} for {patience} consecutive steps.")
                break
        # ----------------------------

    return losses

# Run
registration = Registration(
    drr_projector,
    test_rotations.clone(),
    test_translations.clone(),
    parameterization="euler_angles",
    convention="ZXY"
)

df_results = optimize_registration(
    registration,
    target_xray,
    mi_metric,
    n_itrs=50
)

# %%
# -----------------------------------------------------------------------------
# 6. Save Animation
# -----------------------------------------------------------------------------

def animate_in_browser(df, pred_drr, ground_truth, duration=30):

    out = animate(
        "<bytes>",
        df,
        pred_drr,
        ground_truth=ground_truth,
        verbose=True,
        extension=".webp",
        duration=duration,
        parameterization="euler_angles",
        convention="ZXY",
    )
    # display(HTML(f"""<img src='{"data:img/gif;base64," + b64encode(out).decode()}'>"""))

    export_filename = get_unique_filename("optimization_animation", "webp")
    with open(export_filename,'wb') as f:
        f.write(out)

animate_in_browser(df_results, pred_drr, target_xray, duration=30)


# %%
