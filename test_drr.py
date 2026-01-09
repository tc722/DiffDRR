import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from diffdrr.data import load_example_ct
from diffdrr.drr import DRR
from diffdrr.metrics import NormalizedCrossCorrelation2d
from diffdrr.pose import convert
from diffdrr.registration import Registration
from diffdrr.visualization import plot_drr


# Make the ground truth X-ray
SDD = 1020.0
HEIGHT = 100
DELX = 4.0

subject = load_example_ct()
true_params = {
    "sdr": SDD,
    "alpha": 0.0,
    "beta": 0.0,
    "gamma": 0.0,
    "bx": 0.0,
    "by": 850.0,
    "bz": 0.0,
}
device = "cuda" if torch.cuda.is_available() else "cpu"

drr = DRR(subject, sdd=SDD, height=HEIGHT, delx=DELX).to(device)
rotations = torch.tensor(
    [[true_params["alpha"], true_params["beta"], true_params["gamma"]]]
)
translations = torch.tensor([[true_params["bx"], true_params["by"], true_params["bz"]]])
gt_pose = convert(
    rotations, translations, parameterization="euler_angles", convention="ZXY"
).to(device)
ground_truth = drr(gt_pose)

# plot_drr(ground_truth)
# plt.savefig("ground_truth_drr.png")

# Make a random DRR
np.random.seed(1)


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


rotations, translations, pose = get_initial_parameters(true_params)
drr = DRR(subject, sdd=SDD, height=HEIGHT, delx=DELX).to(device)
with torch.no_grad():
    est = drr(pose)
# plot_drr(est)
# plt.savefig("initial_drr.png")

criterion = NormalizedCrossCorrelation2d()
criterion(ground_truth, est).item()



def optimize_lbfgs(
    reg: Registration,
    ground_truth,
    lr,
    line_search_fn=None,
    n_itrs=100,
):
    # Initialize the optimizer and define the closure function
    optim = torch.optim.LBFGS(reg.parameters(), lr, line_search_fn=line_search_fn)

    def closure():
        if torch.is_grad_enabled():
            optim.zero_grad()
        estimate = reg()
        loss = -criterion(ground_truth, estimate)
        if loss.requires_grad:
            loss.backward()
        return loss

    params = []
    losses = [closure().abs().item()]
    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        # Save the current set of parameters
        alpha, beta, gamma = reg.rotation.squeeze().tolist()
        bx, by, bz = reg.translation.squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

        # Run the optimization loop
        optim.step(closure)
        with torch.no_grad():
            loss = closure().abs().item()
            losses.append(loss)
            pbar.set_description(f"NCC = {loss:06f}")

        # Stop the optimization if the estimated and ground truth images are 99.9% correlated
        if loss > 0.999:
            if line_search_fn is not None:
                method = f"L-BFGS + strong Wolfe conditions"
            else:
                method = "L-BFGS"
            tqdm.write(f"{method} converged in {itr + 1} iterations")
            break

    # Save the final estimated pose
    alpha, beta, gamma = reg.rotation.squeeze().tolist()
    bx, by, bz = reg.translation.squeeze().tolist()
    params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    return df


# Keyword arguments for diffdrr.drr.DRR
kwargs = {
    "subject": subject,
    "sdd": SDD,
    "height": HEIGHT,
    "delx": DELX,
    "stop_gradients_through_grid_sample": True,  # Enables faster optimization
}

drr = DRR(**kwargs).to(device)
reg = Registration(
    drr,
    rotations.clone(),
    translations.clone(),
    parameterization="euler_angles",
    convention="ZXY",
)
params_lbfgs = optimize_lbfgs(
    reg, 
    ground_truth, 
    lr=1.0,  # LBFGS with line search usually works best with lr=1.0
    line_search_fn="strong_wolfe"
)

del drr

from base64 import b64encode

from IPython.display import HTML, display

from diffdrr.visualization import animate

MAX_LENGTH = max(
    map(
        len,
        [
            params_lbfgs,
        ],
    )
)
drr = DRR(subject, sdd=SDD, height=HEIGHT, delx=DELX).to(device)

def animate_in_browser(df, skip=1, max_length=MAX_LENGTH, duration=30):
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

    with open("optimization_animation.webp", "wb") as f:
        f.write(out)
    
animate_in_browser(params_lbfgs)