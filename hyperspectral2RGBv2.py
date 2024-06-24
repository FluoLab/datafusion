import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def Wavelength2color(a):
    lambda_vals = np.array([380, 420, 440, 490, 510, 580, 645, 780])
    r = np.array([97, 106, 0, 0, 0, 255, 255, 97]) / 255
    g = np.array([0, 0, 0, 255, 255, 255, 0, 0]) / 255
    b = np.array([97, 255, 255, 255, 0, 0, 0, 0]) / 255
    
    interp_r = interp1d(lambda_vals, r, kind='linear', bounds_error=False, fill_value=0)
    interp_g = interp1d(lambda_vals, g, kind='linear', bounds_error=False, fill_value=0)
    interp_b = interp1d(lambda_vals, b, kind='linear', bounds_error=False, fill_value=0)
    
    x = interp_r(a)
    y = interp_g(a)
    z = interp_b(a)
    
    return x, y, z

def hyperspectral2RGB(lambda_vals, im):
    if lambda_vals[0] < 380 or lambda_vals[-1] > 780:
        print('Wavelength range out of visible range')
        return None
    
    #values that are less than zero give problems in the spectral visualization
    im[im<0]=0
    
    r, g, b = Wavelength2color(lambda_vals)
    
    S_r = np.zeros((im.shape[-2], im.shape[-1], im.shape[-3]))
    S_g = np.zeros((im.shape[-2], im.shape[-1], im.shape[-3]))
    S_b = np.zeros((im.shape[-2], im.shape[-1], im.shape[-3]))
    
    if im.ndim==4:
        for li in range(im.shape[-3]):
            I = np.sum(im[:, li, :, :], axis=0)
            S_r[:,:,li] = r[li]*I
            S_g[:,:,li] = g[li]*I
            S_b[:,:,li] = b[li]*I
    else:
        for li in range(im.shape[-3]):     
            I = im[li,:,:]
            S_r[:,:,li] = r[li]*I
            S_g[:,:,li] = g[li]*I
            S_b[:,:,li] = b[li]*I
    
    S_r = np.sum(S_r, axis=2)
    S_g = np.sum(S_g, axis=2)
    S_b = np.sum(S_b, axis=2)
    
    max_val = np.max([S_r.max(), S_g.max(), S_b.max()])
    S_r_n = 255 * S_r / max_val
    S_g_n = 255 * S_g / max_val
    S_b_n = 255 * S_b / max_val
    
    S = np.stack((S_r_n, S_g_n, S_b_n), axis=-1).astype(np.uint8)
    
    return S

def hyperspectral2RGBvolume(lambda_vals, im):
    if lambda_vals[0] < 380 or lambda_vals[-1] > 780:
        print('Wavelength range out of visible range')
        return None
    
    r, g, b = Wavelength2color(lambda_vals)
    
    num_layers = im.shape[1]
    height, width = im.shape[2], im.shape[3]
    num_lambda = len(lambda_vals)
    
    S_r = np.zeros((num_lambda, num_layers, height, width))
    S_g = np.zeros((num_lambda, num_layers, height, width))
    S_b = np.zeros((num_lambda, num_layers, height, width))
    
    for li in range(num_lambda):
        for z in range(num_layers):
            I = im[li, z, :, :]
            S_r[li, z, :, :] += r[li] * I
            S_g[li, z, :, :] += g[li] * I
            S_b[li, z, :, :] += b[li] * I
            
    S_r = np.sum(S_r, axis=0)
    S_g = np.sum(S_g, axis=0)
    S_b = np.sum(S_b, axis=0)
    
    max_val = np.max([S_r.max(), S_g.max(), S_b.max()])
    S_r_n = 255 * S_r / max_val
    S_g_n = 255 * S_g / max_val
    S_b_n = 255 * S_b / max_val
    
    S = np.stack((S_r_n, S_g_n, S_b_n), axis=-1).astype(np.uint8)
    
    return S