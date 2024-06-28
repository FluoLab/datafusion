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
from bin_data import bin_data

# def min_max_scale(x):
#     return (x - x.min()) / (x.max() - x.min())

def optimize(x: torch.tensor, spc, cmos) -> torch.tensor:   #x->(time,lam,z,x,y)
    x = torch.swapaxes(x, 0, 1)
    spc = torch.swapaxes(spc, 0, 1) 
    resizer_256 = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)
    
    zw = torch.mean(cmos, dim=(1, 2))
    zw /= zw.max()
    
    # Starting point
    for z in range(x.shape[2]):
        x[:, :, z, :, :] = resizer_256(spc) * zw[z]
    
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
        flattened_x = x.flatten()

        resized = torch.cat([resizer_32(torch.mean(xi, dim=1)).unsqueeze(0) for xi in x]) # for each lambda
        
        # spectral_loss = spectral_fidelity(spc.flatten(), resizer_32(torch.mean(x, dim=1)).flatten())
        spectral_loss = spectral_fidelity(spc.flatten(), resized.flatten())
        spatial_loss = 5 * spatial_fidelity(cmos.flatten(), torch.mean(x, dim=(0,1)).flatten())
        # intensity_loss = intensity_fidelity(torch.mean(cmos,dim=(1,2)).flatten(), torch.mean(x,dim=(0,2,3)).flatten())
        # spectral_slice_loss = spectral_slice_fidelity(spc.repeat(17,1,1,1).transpose(0,1).flatten(), resizer_32(x).flatten())
        # for i in range(cmos.size(0)):
        #     spectral_slice_loss += spectral_slice_fidelity(spc.flatten(), resizer_32(x[:,i,:,:]).flatten())
        # global_lambda_loss =  global_lambda_fidelity(torch.mean(spc, dim=(1, 2)), torch.mean(x, dim=(1,2,3)))
        non_neg_loss = non_neg_fidelity(flattened_x, torch.nn.functional.relu(flattened_x)) 

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

    x = torch.swapaxes(x, 0, 1)
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
    day = '20240612'
    filenamecmos = '3beads_triangle_w4_rec_Hil2D_FOVcorrected.mat'
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


    filenamespc = '480_3beads_triangle_505_500_575_SPC_raw_proc_tlxy.mat'
    mat_fname = pjoin(data_dir,day,filenamespc)
    spc = sp.io.loadmat(mat_fname)["im"]  # (time, lambda, img_dim, img_dim)
    t = np.squeeze(sp.io.loadmat(mat_fname)["t"])
    spc[:,:,0,0] = spc[:,:,1,0]
    # sp.io.savemat('spcOriginal.mat', {'spc': spc})
    filenamelambda = "575_Lambda_L16.mat"
    mat_fname = pjoin(data_dir,"Calibrations",filenamelambda)
    lam = np.squeeze(sp.io.loadmat(mat_fname)["lambda"])
    
    time_dacay = np.sum(spc,axis=(1,2,3))
    plt.plot(t, time_dacay)
    plt.title('Initial global time')
    plt.show()
    
    #data binning
    t,spc,dt = bin_data(spc,t,2)
    
    time_dacay = np.sum(spc,axis=(1,2,3))
    plt.plot(t, time_dacay)
    plt.title('After bin - global spectrum')
    plt.show()
    
    # lam = np.linspace(550,650,16)
    # spc = spc.mean(axis=0)  # (lambda, img_dim, img_dim)
    spc = spc/np.max(spc)
    
    # =========== PLOTS =============
    imageColor = hyperspectral2RGB(lam,np.mean(spc,axis=0))

    plt.plot(lam, np.mean(spc,axis=(0,2,3)))
    plt.title('Initial global spectrum')
    plt.show()

    plt.imshow(np.mean(spc,axis=(0,1)))
    plt.title('Initial SPC image')
    plt.show()
    
    plt.imshow(cmos[5,:,:])
    plt.title('Initial CMOS image')
    plt.show()
    
    plt.imshow(imageColor)
    plt.title('Initial colored SPC image')
    plt.show()
    
    plt.plot(lam,np.mean(spc[:,:,10,11],axis=0))
    plt.plot(lam,np.mean(spc[:,:,16,19],axis=0))
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

    if cmos.ndim>=3:
        x = optimize(x=torch.zeros(len(t), 16, zdim, dimFused, dimFused), spc=spc, cmos=cmos)
    else:
        x = optimize2d(x=torch.zeros(16, dimFused, dimFused), spc=spc, cmos=cmos)
        
    x = x.cpu().detach().numpy()
    np.save("x.npy", x)
    
    # =========== PLOTS =============
    # PLOT IF THE IMAGE IS Z-STACK (time,lam,z,x,y)
    if x.ndim==5:
        zxy = np.sum(x, axis=(0,1))
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
            plt.plot(lam, np.mean(x[:, :, i, :, :],axis=(0,2,3)), label=f"{i}")
        plt.legend()
        plt.tight_layout()
        plt.title('Global spectrum for i-th slice')
        plt.show()
        
        for i in range(1, zdim, 2):
            plt.plot(lam, np.mean(x[:, :, i, 125, 147],axis=0), label=f"{i}")
        plt.legend()
        plt.tight_layout()
        plt.title('Specific point spectrum for i-th slice')
        plt.show()
        
        for i in range(1, zdim, 2):
            plt.plot(lam, np.mean(x[:, :, i, 72, 128],axis=0), label=f"{i}")
        plt.legend()
        plt.tight_layout()
        plt.title('Specific point spectrum for i-th slice - no signal')
        plt.show()
        
        imageColorSlice = hyperspectral2RGB(lam,np.mean(x[:,:,5,:,:],axis=0))
        plt.imshow(imageColorSlice)
        plt.title('Colored SPC image of one slice')
        plt.show()
        
        slicesRGB = hyperspectral2RGBvolume(lam,np.mean(x,axis=0))
        
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
