import gc
from time import perf_counter as timer

import click
import torch
import numpy as np

from datafusion.fusion import FusionAdam, FusionCG
from datafusion.utils import load_data, RESOURCES_PATH, ZENODO_URL, download_url


@click.command()
@click.option(
    "--n-runs",
    "n_runs",
    "-n",
    default=5,
    type=int,
    show_default=True,
    help="Number of runs to time.",
)
@click.option(
    "--device",
    "device",
    "-d",
    default="cuda",
    type=str,
    show_default=True,
    help="Device to run on.",
)
@click.option(
    "--method",
    "method",
    "-m",
    default="adam",
    type=click.Choice(["adam", "cg"]),
    show_default=True,
    help="Fusion method to use.",
)
@click.option(
    "--cr",
    default="0.50",
    show_default=True,
    help="Compression ratio string used to pick the data file (e.g. '0.50').",
)
@click.option(
    "--download",
    default=False,
    is_flag=True,
    show_default=True,
    help="Whether to download the data if not present.",
)
def run_benchmark(
    n_runs: int = 10,
    device: str = "cuda",
    method: str = "adam",
    cr: str = "0.50",
    download: bool = False,
) -> None:

    weights = {
        "spatial": 0.5,
        "spectro_temporal": 0.5,
    }

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    if download:
        download_url(ZENODO_URL, RESOURCES_PATH / "acquisitions.zip", unzip=True)

    spc, cmos, _, _ = load_data(
        path=RESOURCES_PATH / "acquisitions" / "cells" / f"cells_{cr}cr.npz",
        max_xy_size=128,
    )

    runtimes = np.zeros(n_runs)
    for i in range(n_runs):
        if method == "adam":
            fusion = FusionAdam(
                spc,
                cmos,
                weights=weights,
                init_type="baseline",
                mask_noise=False,
                tol=1e-6,
                total_energy=1,
                device=device,
                seed=42,
                verbose=False,
            )
            start = timer()
            fusion(max_iterations=100, lr=1e-8)
            end = timer()
            runtimes[i] = end - start

        elif method == "cg":
            fusion = FusionCG(
                spc,
                cmos,
                weights=weights,
                init_type="baseline",
                mask_noise=False,
                tol=1e-6,
                total_energy=1,
                device=device,
                seed=42,
                verbose=False,
            )
            start = timer()
            fusion(max_iterations=10)
            end = timer()
            runtimes[i] = end - start

        else:
            raise ValueError(f"Unknown method: {method}")

        # Clean everything from GPU memory and force garbage collection
        try:
            del fusion
            del x
        except NameError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"Mean runtime: {np.mean(runtimes):.1f} s")
    print(f"Std runtime: {np.std(runtimes):.4f} s")


if __name__ == "__main__":
    run_benchmark()
