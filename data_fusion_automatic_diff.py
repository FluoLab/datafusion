#%%
import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import skimage as ski
from os.path import dirname, join as pjoin
import h5py
from PIL import Image
from torchvision import transforms
from hyperspectral2RGBv2 import hyperspectral2RGB,hyperspectral2RGBvolume

# def min_max_scale(x):
#     return (x - x.min()) / (x.max() - x.min())

def optimize(x: torch.tensor, spc, cmos) -> torch.tensor:   #x->(lam,z,x,y)
    resizer_256 = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)
    
    zw = torch.mean(cmos, dim=(1, 2))
    zw /= zw.max()
    
    # Starting point
    for z in range(x.shape[1]):
        x[:, z, :, :] = resizer_256(spc) * zw[z]
    
    x = torch.nn.Parameter(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.01)

    spectral_fidelity = torch.nn.MSELoss(reduction="mean")
    spatial_fidelity = torch.nn.MSELoss(reduction="mean")
    spectral_slice_fidelity = torch.nn.MSELoss(reduction="mean")
    intensity_fidelity = torch.nn.MSELoss(reduction="mean")
    non_neg_fidelity = torch.nn.MSELoss(reduction="mean")
    
    # spectral_slice_loss = 0
    # global_lambda_fidelity = torch.nn.MSELoss(reduction="mean")
    
    resizer_32 = torch.nn.AvgPool2d(8, 8)
    #resizer_128 = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR)

    for it in range(80):
        optimizer.zero_grad()

        # spectral_loss = spectral_fidelity(spc.flatten(), resizer_32(torch.mean(x, dim=1)).flatten())
        spectral_loss = spectral_fidelity(spc.flatten(), resizer_32(torch.mean(x, dim=1)).flatten())
        spatial_loss = 5 * spatial_fidelity(cmos.flatten(), torch.mean(x, dim=0).flatten())
        # intensity_loss = intensity_fidelity(torch.mean(cmos,dim=(1,2)).flatten(), torch.mean(x,dim=(0,2,3)).flatten())
        # spectral_slice_loss = spectral_slice_fidelity(spc.repeat(17,1,1,1).transpose(0,1).flatten(), resizer_32(x).flatten())
        # for i in range(cmos.size(0)):
        #     spectral_slice_loss += spectral_slice_fidelity(spc.flatten(), resizer_32(x[:,i,:,:]).flatten())
        # global_lambda_loss =  global_lambda_fidelity(torch.mean(spc, dim=(1, 2)), torch.mean(x, dim=(1,2,3)))
        non_neg_loss = non_neg_fidelity(x, torch.nn.functional.relu(x)) 

        loss = spectral_loss + spatial_loss + non_neg_loss # + intensity_loss # intensity_loss

        loss.backward()
        optimizer.step()

        print(
            f"Iteration {it + 1} | "
            f"Spectral: {spectral_loss.item():.4F} | "
            f"Spatial: {spatial_loss.item():.4F} | "
            f"Non Neg: {non_neg_loss.item():.4F} | "
            #f"Intensity: {intensity_loss.item():.4F} | "
            #f"Global: {global_lambda_loss.item():.4F}"
            )
        
        # spectral_slice_loss = 0

    return x

def optimize2d(x: torch.tensor, spc, cmos) -> torch.tensor:  #x->(lam,x,y)
    resizer_256 = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)
    
    # Starting point
    x = resizer_256(spc)
    
    # Adding noise 
    # x = x + torch.rand_like(x, requires_grad=False) * 0.5
    x = torch.nn.Parameter(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.01)

    spatial_fidelity = torch.nn.MSELoss(reduction="mean")
    spectral_fidelity = torch.nn.MSELoss(reduction="mean")
    global_lambda_fidelity = torch.nn.MSELoss(reduction="mean")
    
    resizer_32 = torch.nn.AvgPool2d(8, 8)
    
    for it in range(256):
        optimizer.zero_grad()

        spectral_loss = spectral_fidelity(spc.flatten(), resizer_32(x).flatten())
        spatial_loss = 5 * spatial_fidelity(cmos.flatten(), torch.mean(x, dim=0).flatten())
        global_lambda_loss = global_lambda_fidelity(torch.mean(spc, dim=(1, 2)), torch.mean(x, dim=(1,2)))

        loss = spatial_loss + spectral_loss # + global_lambda_loss

        loss.backward()
        optimizer.step()

        print(
            f"Iteration {it + 1} | "
            f"Spectral: {spectral_loss.item():.4F} | "
            f"Spatial: {spatial_loss.item():.4F} | "
            f"Global: {global_lambda_loss.item():.4F}"
            )

    return x


def main():
    data_dir = '/Users/federicosimoni/Library/Mobile Documents/com~apple~CloudDocs/Università/Tesi/Code/CS-FLIM_lab'
    day = '20240617'
    filenamecmos = 'kidney_cells_520_610_w4_rec_Hil2D_FOVcorrected.mat'
    mat_fname = pjoin(data_dir,day,filenamecmos)
    with h5py.File(mat_fname, "r") as f:
        mat_contents = h5py.File(mat_fname)

    dimFused = 256
    # print(list(mat_contents.keys()))
    data = mat_contents['I']
    cmos = np.array(data)
    # cmos = sp.io.loadmat("/Users/federicosimoni/Library/Mobile Documents/com~apple~CloudDocs/Università/Tesi/Code/CS-FLIM_lab/20240614/kidney_cells_2_520_w4_rec_Hil2D_FOVcorrected.mat")["I"]  # (x, y, z)
    if cmos.ndim==3:
        cmos = np.transpose(cmos, (1, 2, 0))
        zdim = cmos.shape[2]
        cmos = ski.transform.resize(cmos, (dimFused, dimFused, zdim))
        cmos = np.transpose(cmos, (2, 1, 0))
        cmos = cmos/cmos.max()
    else:
        cmos = ski.transform.resize(cmos, (dimFused, dimFused))
        cmos = np.transpose(cmos, (1, 0))
        cmos = cmos/cmos.max()
        plt.imshow(cmos)
        plt.title('Initial cmos')
        plt.show()
    
    # mask = cmos < 0.05
    # cmos[mask] = 0
    cmos = torch.from_numpy(cmos.astype(np.float32))


    filenamespc = '520_kidneyCells_550_550_610_SPC_raw_proc_tlxy.mat'
    mat_fname = pjoin(data_dir,day,filenamespc)
    spc = sp.io.loadmat(mat_fname)["im"]  # (time, lambda, img_dim, img_dim)
    spc[:,:,0,0] = spc[:,:,1,0]
    # sp.io.savemat('spcOriginal.mat', {'spc': spc})
    filenamelambda = "610_Lambda_L16.mat"
    mat_fname = pjoin(data_dir,"Calibrations",filenamelambda)
    lam = np.squeeze(sp.io.loadmat(mat_fname)["lambda"])
    # lam = np.linspace(550,650,16)
    spc = spc.mean(axis=0)  # (lambda, img_dim, img_dim)
    spc = spc/np.max(spc)
    
    # =========== PLOTS =============
    imageColor = hyperspectral2RGB(lam,spc)

    plt.plot(lam, np.mean(spc,axis=(1,2)))
    plt.title('Initial global spectrum')
    plt.show()

    plt.imshow(np.mean(spc,axis=0))
    plt.title('Initial SPC image')
    plt.show()
    
    plt.imshow(cmos[10,:,:])
    plt.title('Initial CMOS image')
    plt.show()
    
    plt.imshow(imageColor)
    plt.title('Initial colored SPC image')
    plt.show()
    
    plt.plot(lam,spc[:,10,11])
    plt.plot(lam,spc[:,9,16])
    plt.title('Initial spectrum specific point')
    plt.show()
    
    # plt.plot(spc[:,12,20])
    # plt.title('Initial spectrum small bead')
    # plt.show()
    # ==================================
    
    # for i in range(len(spc)):
    #     spc[i] = spc[i]/spc[i].max()
    # mask = ski.transform.resize(np.all(mask, axis=0), (32, 32))
    # spc[:, mask] = 0
    
    spc = torch.from_numpy(spc.astype(np.float32))

    if cmos.ndim==3:
        x = optimize(x=torch.zeros(16, zdim, dimFused, dimFused), spc=spc, cmos=cmos)
    else:
        x = optimize2d(x=torch.zeros(16, dimFused, dimFused), spc=spc, cmos=cmos)
    x = x.cpu().detach().numpy()
    
    # =========== PLOTS =============
    # PLOT IF THE IMAGE IS Z-STACK (lam,z,x,y)
    if x.ndim==4:
        zxy = np.sum(x, axis=0)
        zxy /= zxy.max()

        #maxCMOS = zxy.max()
        #minCMOS = zxy.min()
        dimPlot=int(np.ceil(np.sqrt(zdim)))
        _, axs = plt.subplots(dimPlot,dimPlot)
        for i in range(zdim):
            axs[int(i/dimPlot), int(i%dimPlot)].imshow(zxy[i], cmap="gray", vmin=0, vmax=1)
        plt.tight_layout(pad=0.4)
        plt.show()
        plt.close()

        # plt.plot(spc.cpu().detach().numpy()[:, 10, 10])
        # plt.tight_layout()
        # plt.show()

        for i in range(1, zdim, 2):
            plt.plot(lam, np.mean(x[:, i, :, :],axis=(1,2)), label=f"{i}")
        plt.legend()
        plt.tight_layout()
        plt.title('Global spectrum for i-th slice')
        plt.show()
        
        for i in range(1, zdim, 2):
            plt.plot(lam, x[:, i, 80, 88], label=f"{i}")
        plt.legend()
        plt.tight_layout()
        plt.title('Specific point spectrum for i-th slice')
        plt.show()
        
        for i in range(1, zdim, 2):
            plt.plot(lam, x[:, i, 72, 128], label=f"{i}")
        plt.legend()
        plt.tight_layout()
        plt.title('Specific point spectrum for i-th slice - no signal')
        plt.show()
        
        imageColorSlice = hyperspectral2RGB(lam,x[:,10,:,:])
        plt.imshow(imageColorSlice)
        plt.title('Colored SPC image of one slice')
        plt.show()
        
        slicesRGB = hyperspectral2RGBvolume(lam,x)
        
        # dimPlot=int(np.ceil(np.sqrt(zdim)))
        # _, axs = plt.subplots(dimPlot,dimPlot)
        # for i in range(zdim):
        #     axs[int(i/dimPlot), int(i%dimPlot)].imshow(hyperspectral2RGB(lam,x[:,i,:,:]))
        # plt.tight_layout(pad=0.1)
        # plt.show()
        # plt.close()
        
        dimPlot=int(np.ceil(np.sqrt(zdim)))
        _, axs = plt.subplots(dimPlot,dimPlot)
        for i in range(zdim):
            axs[int(i/dimPlot), int(i%dimPlot)].imshow(slicesRGB[i,:,:,:])
        plt.tight_layout(pad=0.1)
        plt.show()
        plt.close()
    #PLOT IF THE IMAGE IS ONE SLICE (lam,x,y)
    else:
        plt.plot(np.mean(x,axis=(1,2)))
        plt.title('Global spectrum')
        plt.show()

        plt.imshow(np.sum(x,axis=0))
        plt.title('Intensity image')
        plt.show()
        
        plt.plot(x[:,80,88])
        plt.title('Spectrum specific point')
        plt.show()
        
        # imageColorFused = hyperspectral2RGB(lam,x)
        
        # plt.imshow(imageColorFused)
        # plt.title('Colored SPC image')
        # plt.show()
    # ===============================
        
    # slicesRGB[0,:,:,:].save("out.gif", save_all=True, append_images=[slicesRGB[1:,:,:,:]], duration=1000, loop=0)

if __name__ == "__main__":
    main()
# %%
