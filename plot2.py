import h5py
import numpy as np
import scipy as sp
import skimage as ski
import matplotlib.pyplot as plt
import os

from fusion import optimize
from utils import hyperspectral2RGB, hyperspectral2RGBvolume, bin_data
from utils import RESOURCES_PATH
from ipywidgets import interact

x = np.load(RESOURCES_PATH / "kidney" / "520_kidneyCells_550_550_610_SPC_raw_proc_tlxy_fused.npy")

LAMBDA_PATH = RESOURCES_PATH / "kidney" / "610_Lambda_L16.mat"

DATA_RESOURCES_PATH = "/Users/federicosimoni/Library/Mobile Documents/com~apple~CloudDocs/Università/Tesi/Code/CS-FLIM_lab/20240703/kidney"
SPC_PATH = os.path.join(DATA_RESOURCES_PATH,"520_kidney2_550_550_610_SPC_raw_proc_tlxy.mat")

spc = sp.io.loadmat(SPC_PATH)["im"]
t = np.squeeze(sp.io.loadmat(SPC_PATH)["t"])
spc[:, :, 0, 0] = spc[:, :, 1, 0]

lam = np.squeeze(sp.io.loadmat(LAMBDA_PATH)["lambda"])

slices_rgb = hyperspectral2RGBvolume(lam, np.mean(x, axis=0))
slice_rgb = hyperspectral2RGB(lam, np.mean(x, axis=0)[:,9,:,:])

plt.imshow(slice_rgb)
plt.title("RGB image of a slice")
plt.show()