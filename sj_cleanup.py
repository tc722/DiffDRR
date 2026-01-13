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
from tqdm import tqdm
from pathlib import Path
import imageio.v3 as iio
import io
import base64
import seaborn as sns

# DiffDRR imports
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert
from diffdrr.visualization import plot_drr, animate
from diffdrr.metrics import MutualInformation
from diffdrr.registration import Registration
from base64 import b64encode
from IPython.display import HTML, display
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


# %%
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
sdp = dcm_meta.DistanceSourceToPatient

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
plt.show()
plt.close()

# %%
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
    "bx": 0.0,    "by": sdp, "bz": 0.0
}

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

test_rotations, test_translations = test_pose.convert("euler_angles", "ZXY")

# Generate Prediction and Apply Normalization
with torch.no_grad():
    pred_drr = 1.0 - normalize_tensor(drr_projector(test_pose))

# Visualization
plot_drr(pred_drr)
plt.title(f"Initial MI = {mi_metric(target_xray, pred_drr).item():.5f}")
plt.show()
plt.savefig(patient_dir / "predicted_mri_drr.png")
plt.close()

del pred_drr, test_pose, drr_projector
# %%
# -----------------------------------------------------------------------------
# 5. Optimization
# -----------------------------------------------------------------------------
def optimize(
    reg: Registration,
    ground_truth,
    mi_metric,
    lr_rotations=5e-2,
    lr_translations=1e1,
    momentum=0,
    dampening=0,
    n_itrs=500,
    tol=1e-5,
    patience=5,
    optimizer="sgd",  # 'sgd' or `adam`
):
    # Initialize an optimizer with different learning rates
    # for rotations and translations since they have different scales
    if optimizer == "sgd":
        optim = torch.optim.SGD(
            [
                {"params": [reg._rotation], "lr": lr_rotations},
                {"params": [reg._translation], "lr": lr_translations},
            ],
            momentum=momentum,
            dampening=dampening,
            maximize=True,
        )
        optimizer = optimizer.upper()
    elif optimizer == "adam":
        optim = torch.optim.Adam(
            [
                {"params": [reg._rotation], "lr": lr_rotations},
                {"params": [reg._translation], "lr": lr_translations},
            ],
            maximize=True,
        )
        optimizer = optimizer.title()
    else:
        raise ValueError(f"Unrecognized optimizer {optimizer}")
    print(f"Using {optimizer} optimizer for registration.")
    params = []
    losses = [mi_metric(ground_truth, 1.0 - normalize_tensor(reg())).item()]
    patience_counter = 0
    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        # Save the current set of parameters
        alpha, beta, gamma = reg.rotation.squeeze().tolist()
        bx, by, bz = reg.translation.squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

        # Run the optimization loop
        optim.zero_grad()
        estimate = reg()
        est_norm = normalize_tensor(estimate)
        est_inverted = 1.0 - est_norm
        loss = mi_metric(ground_truth, est_inverted)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        pbar.set_description(f"MI = {loss.item():06f} |  Params = [{alpha/torch.pi*180.0:.1f}, {beta/torch.pi*180.0:.1f}, {gamma/torch.pi*180.0:.1f}, {bx:.1f}, {by:.1f}, {bz:.1f}]")
    
 # --- EARLY STOPPING LOGIC ---
        if itr > 0:
            # Compare current MI with the previous step's MI
            previous_mi = losses[-2]
            improvement = loss - previous_mi
            
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

    # Save the final estimated pose
    alpha, beta, gamma = reg.rotation.squeeze().tolist()
    bx, by, bz = reg.translation.squeeze().tolist()
    params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    return df

def optimize_registration(reg_module, ground_truth, metric, lr=1e-2, n_itrs=100, tol=1e-4, patience=5, line_search_fn =None):
    """
    Optimizes the registration with early stopping.
    
    Args:
        tol (float): Minimum improvement required to reset patience.
        patience (int): Number of iterations to wait before stopping if no improvement.
    """
    optimizer = torch.optim.LBFGS(
        reg_module.parameters(), 
        lr=lr, 
        line_search_fn=line_search_fn
    )
    
    patience_counter = 0
    
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

    params = []
    losses = [closure().abs().item()]
    print("Using L-BFGS optimizer for registration.")
    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        # Save the current set of parameters
        alpha, beta, gamma = reg_module.rotation.squeeze().tolist()
        bx, by, bz = reg_module.translation.squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

        # Run the optimization loop
        optimizer.step(closure)
        with torch.no_grad():
            loss = closure().abs().item()
            losses.append(loss)
            pbar.set_description(f"MI = {loss:06f}| Params = [{alpha/torch.pi*180.0:.1f}, {beta/torch.pi*180.0:.1f}, {gamma/torch.pi*180.0:.1f}, {bx:.1f}, {by:.1f}, {bz:.1f}]")

         # --- EARLY STOPPING LOGIC ---
        if itr > 0:
            # Compare current MI with the previous step's MI
            previous_mi = losses[-2]
            improvement = loss - previous_mi
            
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


    alpha, beta, gamma = reg_module.rotation.squeeze().tolist()
    bx, by, bz = reg_module.translation.squeeze().tolist()
    params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    return df

def plot_loss_curve(df):
    """
    Plots the loss (Negative Mutual Information) over iterations.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the curve
    plt.plot(df.index, df['loss'], marker='o', markersize=3, linestyle='-', color='#007acc', linewidth=2, label='Loss (Negative MI)')
    
    # Highlight the minimum (Best Result)
    max_loss = df['loss'].max()
    max_iter = df['loss'].idxmax()
    plt.scatter(max_iter, max_loss, color='red', s=100, zorder=5, label=f'Best: {max_loss:.5f}')
    
    # Styling
    plt.title("Optimization Convergence: Loss vs Iterations", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Negative Mutual Information", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Optional: Annotate start and end
    plt.annotate(f'Start: {df["loss"].iloc[0]:.4f}', (0, df["loss"].iloc[0]), textcoords="offset points", xytext=(10,10), ha='left')
    plt.annotate(f'End: {df["loss"].iloc[-1]:.4f}', (len(df)-1, df["loss"].iloc[-1]), textcoords="offset points", xytext=(-10,10), ha='right')
    
    plt.tight_layout()
    plt.show()


# Run
lr_rot = 0.01 * torch.pi / 180.0  # 0.1 degree in radians
lr_trans = 1e-1 # 1 mm
lr_lbfgs = 5e-2

niteration = 800
tolerance = 1e-5
sigma = 0.01

print("Current Sigma for MI Metric:", sigma)

# Define Test Pose
test_pose_params = {
    "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
    "bx": 0.0,    "by": sdp, "bz": 0.0
}

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

test_rotations, test_translations = test_pose.convert("euler_angles", "ZXY")

# Initialize MI Metric
mi_metric = MutualInformation(
    sigma=sigma, 
    num_bins=256, 
    epsilon=1e-10, 
    normalize=True
).to(device)

drr_projector = DRR(
    subject_mri, 
    sdd=sdd, 
    height=new_height,       # e.g., 256
    delx=new_pixel_size,     # e.g., 4x larger pixel size
    patch_size=250           
).to(device)

# %%
# # L-BFGS Optimization

# registration = Registration(
#     drr_projector,
#     test_rotations.clone(),
#     test_translations.clone(),
#     parameterization="euler_angles",
#     convention="ZXY"
# )

# df_results_lbfgs=[]
# df_results_lbfgs = optimize_registration(
#     registration,
#     target_xray,
#     mi_metric,
#     lr = lr_lbfgs,
#     n_itrs=niteration,
#     tol=tolerance,
# )
# del registration

# %%
# L-BFGS with Strong Wolfe Line Search
registration = Registration(
    drr_projector,
    test_rotations.clone(),
    test_translations.clone(),
    parameterization="euler_angles",
    convention="ZXY"
)
df_results_lbfgs_wolfe=[]
df_results_lbfgs_wolfe = optimize_registration(
    registration,
    target_xray,
    mi_metric,
    lr = lr_lbfgs,
    n_itrs=niteration,
    line_search_fn="strong_wolfe"
)
del registration

# %%
# Adam Optimization

registration = Registration(
    drr_projector,
    test_rotations.clone(),
    test_translations.clone(),
    parameterization="euler_angles",
    convention="ZXY"
)

df_results_adam =[]
df_results_adam = optimize(
    registration,
    target_xray,
    mi_metric,
    lr_rot,
    lr_trans,
    n_itrs=niteration,
    tol=tolerance,
    optimizer="adam",
    )

del registration

plot_loss_curve(df_results_adam)
animate_in_browser(df_results_adam, skip = 5, max_length=len(df_results_adam),duration=50)

# -----------------------------------------------------------------------------
# 6. Save Animation
# -----------------------------------------------------------------------------

# %%

def animate_in_browser(df, skip=1, max_length=50, duration=30):
    if max_length is not None:
        n = max_length - len(df)
        df = pd.concat([df, df.iloc[[-1] * n]]).iloc[::skip]
    else:
        pass

    out = animate(
        "<bytes>",
        df,
        drr_projector,
        ground_truth=target_xray,
        verbose=True,
        device=device,
        extension=".webp",
        duration=duration,
        parameterization="euler_angles",
        convention="ZXY",
    )
    display(HTML(f"""<img src='{"data:img/gif;base64," + b64encode(out).decode()}'>"""))

def display_optimization_evolution(
    df, 
    drr_projector, 
    target_img, 
    translation_scale=1.0, 
    fps=10, 
    downsample_display=1
):
    """
    Generates and displays an in-notebook animation using WEBP (High Quality).
    """
    
    # 1. Prepare Target (Red Channel)
    t_red = target_img.detach().cpu().squeeze()
    t_red = (t_red - t_red.min()) / (t_red.max() - t_red.min())
    t_red = t_red.numpy()
    
    if downsample_display > 1:
        t_red = t_red[::downsample_display, ::downsample_display]
    
    frames = []
    print(f"Rendering {len(df)} frames for notebook display...")

    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=100):
        
        # --- A. Reconstruct Pose (Float32 Fixed) ---
        rot_tensor = torch.tensor(
            [[row['alpha'], row['beta'], row['gamma']]], 
            dtype=torch.float32, 
            device=drr_projector.device
        )
        
        trans_tensor = torch.tensor(
            [[row['bx'], row['by'], row['bz']]], 
            dtype=torch.float32, 
            device=drr_projector.device
        ) * translation_scale
        
        pose = convert(
            rot_tensor, 
            trans_tensor, 
            parameterization="euler_angles", 
            convention="ZXY"
        )
        
        # --- B. Generate Prediction (Green Channel) ---
        with torch.no_grad():
            pred_raw = drr_projector(pose)
            
            val_min, val_max = pred_raw.min(), pred_raw.max()
            pred_norm = (pred_raw - val_min) / (val_max - val_min) if val_max > val_min else torch.zeros_like(pred_raw)
            pred_inverted = 1.0 - pred_norm
            
            t_green = pred_inverted.squeeze().cpu().numpy()
            
            if downsample_display > 1:
                t_green = t_green[::downsample_display, ::downsample_display]

        # --- C. Create RGB Overlay ---
        H, W = t_red.shape
        rgb_img = np.zeros((H, W, 3), dtype=np.float32)
        rgb_img[..., 0] = t_red      # Red
        rgb_img[..., 1] = t_green    # Green

        # --- D. Plot to Buffer ---
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(rgb_img)
        ax.axis('off')
        
        loss_val = row['loss'] if 'loss' in row else 0.0
        ax.set_title(f"Iter {idx}: Loss={loss_val:.4f}", color='white', backgroundcolor='black', fontsize=10)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=72)
        plt.close(fig)
        
        buf.seek(0)
        frames.append(iio.imread(buf, index=None))
        buf.close()

    # 3. Compile WEBP in Memory
    webp_buffer = io.BytesIO()
    
    # Note: loop=0 means infinite loop
    iio.imwrite(
        webp_buffer, 
        frames, 
        extension=".webp", 
        duration=1000/fps, 
        loop=0, 
        quality=90
    )
    webp_buffer.seek(0)
    
    # 4. Base64 Encode & Display
    b64_data = base64.b64encode(webp_buffer.read()).decode("utf-8")
    
    # Change mimetype to image/webp
    html_str = f'<img src="data:image/webp;base64,{b64_data}" style="width:50%"/>'
    
    display(HTML(html_str))


parameters = {}

# 2. Add entries conditionally
if df_results_lbfgs_wolfe is not None:
    parameters["L-BFGS (Wolfe)"] = (df_results_lbfgs_wolfe, "#ffd92f")

if df_results_lbfgs is not None:
    parameters["L-BFGS"] = (df_results_lbfgs, "#a6d854")

if df_results_adam is not None:
    # I added a color/label for Adam since your snippet cut off
    parameters["Adam"] = (df_results_adam, "#e78ac3")

with sns.axes_style("darkgrid"):
    plt.figure(figsize=(6, 4), dpi=150)
    
    for name, (data, color) in parameters.items():
        # CHECK: Is it a DataFrame or a List?
        if isinstance(data, pd.DataFrame):
            # It's a DataFrame: Access the "loss" column
            plt.plot(data["loss"], label=name, color=color, linewidth=2)
        elif isinstance(data, list):
            # It's a List: Plot it directly
            plt.plot(data, label=name, color=color, linewidth=2)
        else:
            print(f"Skipping {name}: Unknown data type {type(data)}")

    plt.xlabel("Iterations")
    plt.ylabel("Loss (Negative MI)")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

animate_in_browser(df_results_adam, skip = 5, max_length=len(df_results_adam),duration=50)
animate_in_browser(df_results_lbfgs_wolfe, max_length=len(df_results_lbfgs_wolfe),duration=50)
animate_in_browser(df_results_lbfgs, max_length=len(df_results_lbfgs),duration=50)

display_optimization_evolution(
    df_results_lbfgs, 
    drr_projector,  # Use the raw projector (not the registration wrapper)
    target_xray, 
    translation_scale=1.0, # Change to 1000.0 if you optimized in Meters!
    fps=5
)


# %%
def visualize_final_result(df, drr_projector, target_img, translation_scale=1.0):
    """
    Visualizes the BEST result from the optimization.
    
    Args:
        df (pd.DataFrame): Optimization history.
        drr_projector (DRR): The DRR module.
        target_img (Tensor): Ground truth X-ray.
        translation_scale (float): 1000.0 if you optimized in Meters, 1.0 otherwise.
    """
    
    # 1. Find the Best Parameters (Minimum Loss)
    best_idx = df['loss'].idxmax()
    best_row = df.loc[best_idx]
    
    print(f"Visualizing Best Result (Iter {best_idx})")
    print(f"Loss: {best_row['loss']:.5f}")
    
    # 2. Reconstruct Pose
    rot_tensor = torch.tensor(
        [[best_row['alpha'], best_row['beta'], best_row['gamma']]], 
        dtype=torch.float32, 
        device=device
    )
    
    trans_tensor = torch.tensor(
        [[best_row['bx'], best_row['by'], best_row['bz']]], 
        dtype=torch.float32, 
        device=device
    ) 
    
    pose = convert(
        rot_tensor, 
        trans_tensor, 
        parameterization="euler_angles", 
        convention="ZXY"
    )
    
    # 3. Generate Predicted DRR
    with torch.no_grad():
        pred_raw = drr_projector(pose)
        
        # Normalize to [0, 1]
        val_min, val_max = pred_raw.min(), pred_raw.max()
        pred_norm = (pred_raw - val_min) / (val_max - val_min)
        pred_inverted = 1.0 - pred_norm
        
        # Move to CPU / Numpy
        img_pred = pred_inverted.squeeze().cpu().numpy()
        
    # Prepare Target (Red)
    img_target = target_img.detach().cpu().squeeze()
    img_target = (img_target - img_target.min()) / (img_target.max() - img_target.min())
    img_target = img_target.numpy()

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    
    # --- Left Panel: RGB Overlay ---
    # Red = Target, Green = Prediction, Yellow = Overlap
    H, W = img_target.shape
    rgb_img = np.zeros((H, W, 3))
    rgb_img[..., 0] = img_target  # Red
    rgb_img[..., 1] = img_pred    # Green
    
    ax1.imshow(rgb_img)
    ax1.set_title(f"Overlay (Iter {best_idx})\nRed=Target, Green=Pred")
    ax1.axis('off')

    # --- Right Panel: Voxel-wise Difference ---
    # We use a Diverging colormap (Blue=Negative, Red=Positive, White=Zero)
    diff_map = img_pred - img_target
    
    # Use vmin/vmax to center the colormap at 0 for accurate difference perception
    im2 = ax2.imshow(diff_map, cmap='seismic', vmin=-0.5, vmax=0.5)
    ax2.set_title("Difference Map\n(Pred - Target)")
    ax2.axis('off')
    
    # Add colorbar for the difference map
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

visualize_final_result(
    df_results_adam, 
    drr_projector,  # Use the base projector
    target_xray, 
    translation_scale=1000.0 # Set to 1.0 if you didn't use the scaling trick
)
# %%
