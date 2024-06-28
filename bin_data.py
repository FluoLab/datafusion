import numpy as np

def bin_data(data_nobin, t_nobin, dt, bin_type='Bin'):
    data = data_nobin
    t = t_nobin
    bin_size = round(len(t) / (dt / (t[1] - t[0])))
    
    if bin_size < len(t):
        N = data.shape[0]
        K = np.arange(1, N + 1)
        D = K[N % K == 0]
        p = np.argmin(np.abs(bin_size - D))
        bins = D[p]
        bin_length = int(N / bins)
        
        bin_edges = np.linspace(0, N, bins)
        data_1 = np.zeros((bins, data.shape[1], data.shape[2], data.shape[3]))
        for li in range(data.shape[1]):
            for xi in range(data.shape[2]):
                for yi in range(data.shape[3]):
                    data_1[:, li, xi, yi] = data[:, li, xi, yi].reshape(-1, bin_length).sum(axis=1) 
        data = data_1
        t = t_nobin.reshape(-1, bin_length).mean(axis=1)
        
    if abs((t[1] - t[0]) - dt) > (dt / 2):
        print('Some problems determining the desired bin size.')
    
    dt = t[1] - t[0]

    return t, data, dt