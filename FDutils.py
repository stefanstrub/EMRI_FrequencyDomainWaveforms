import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
S_git = np.genfromtxt('./LISA_Alloc_Sh.txt')
Sh_X = CubicSpline(S_git[:,0], S_git[:,1])

request_gpu = False
if request_gpu:
    try:
        import cupy as xp
        from cupyx.scipy.signal import convolve
    except:
        from scipy.signal import convolve
        import numpy as xp
else:
    from scipy.signal import convolve
    import numpy as xp


# redefining the LISA sensitivity
def get_sensitivity(f):
    """
    Sensitivity defined from 
    ff = 10**np.linspace(-5.0, 1.0,num=100)
    plt.figure(); plt.loglog(ff, get_sensitivity(ff)); plt.loglog(ff, Sh_X(ff),'--'); plt.show()
    """

    return Sh_X(f)

def get_convolution(a,b):
    # convolve two signals
    return convolve(xp.hstack((a[1:], a)), b, mode='valid')/len(b)

def get_fft_td_windowed(signal, window, dt):
    fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(signal[0] * window )) * dt
    fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(signal[1] * window )) * dt
    return [fft_td_wave_p,fft_td_wave_c]

def get_fd_windowed(signal, window, window_in_fd=False):
    if window is None:
        transf_fd_0 = signal[0]
        transf_fd_1 = signal[1]
    else:
        # # fft convolution
        # transf_fd_0 = xp.fft.fftshift(xp.fft.fft(xp.fft.ifft( xp.fft.ifftshift( signal[0] ) ) * window))
        # transf_fd_1 = xp.fft.fftshift(xp.fft.fft(xp.fft.ifft( xp.fft.ifftshift( signal[1] ) ) * window))
        
        # standard convolution
        if window_in_fd:
            fft_window = window.copy()
        else:
            fft_window = xp.fft.fft(window)
        transf_fd_0 = get_convolution( xp.conj(fft_window) , signal[0] )
        transf_fd_1 = get_convolution( xp.conj(fft_window) , signal[1] )

        # # test check convolution
        # sum_0 = xp.sum(xp.abs(transf_fd_0)**2)
        # yo = get_convolution( xp.conj(fft_window) , signal[0] )
        # sum_yo = xp.sum(xp.abs(yo)**2)
        # xp.dot(xp.conj(yo) , transf_fd_0 ) /xp.sqrt(sum_0 * sum_yo)

    return [transf_fd_0, transf_fd_1]



class get_fd_waveform_fromFD():

    def __init__(self, waveform_generator, positive_frequency_mask, dt, non_zero_mask=None, window=None, window_in_fd=False):
        self.waveform_generator = waveform_generator
        self.positive_frequency_mask = positive_frequency_mask
        self.non_zero_mask = non_zero_mask
        self.window = window
        self.window_in_fd = window_in_fd

    def __call__(self,*args, **kwargs):
        data_channels_td = self.waveform_generator(*args, **kwargs)
        list_p_c = get_fd_windowed(data_channels_td, self.window, window_in_fd=self.window_in_fd)
        ch1 = list_p_c[0][self.positive_frequency_mask]
        ch2 = list_p_c[1][self.positive_frequency_mask]
        if self.non_zero_mask is not None:
            ch1[~self.non_zero_mask] = complex(0.0)
            ch2[~self.non_zero_mask] = complex(0.0)
        return [ch1,ch2]

# conversion
class get_fd_waveform_fromTD():

    def __init__(self, waveform_generator, positive_frequency_mask, dt, non_zero_mask=None, window=None):
        self.waveform_generator = waveform_generator
        self.positive_frequency_mask = positive_frequency_mask
        self.dt = dt
        self.non_zero_mask = non_zero_mask
        if window is None:
            self.window = np.ones_like(self.positive_frequency_mask)
        else:
            self.window = window

    def __call__(self,*args, **kwargs):
        data_channels_td = self.waveform_generator(*args, **kwargs)
        list_p_c = get_fft_td_windowed(data_channels_td, self.window, self.dt)
        fft_td_wave_p = list_p_c[0][self.positive_frequency_mask]
        fft_td_wave_c = list_p_c[1][self.positive_frequency_mask]
        if self.non_zero_mask is not None:
            fft_td_wave_p[~self.non_zero_mask] = complex(0.0)
            fft_td_wave_c[~self.non_zero_mask] = complex(0.0)
        return [fft_td_wave_p,fft_td_wave_c]

def get_colorplot(data, color_value, label):
    colors = color_value
    n_dimensions = data.shape[-1]
    # Plot the corner plot
    figure, axes = plt.subplots(n_dimensions-1, n_dimensions-1, figsize=(10, 10))

    # Custom color map
    cmap = plt.cm.get_cmap('cool')  # Choose a color map of your preference

    for i in range(n_dimensions-1):
        for j in range(n_dimensions-1):
            if j < i:
                axes[j, i].axis('off')
            else:
                axes[j, i].scatter(data[:, i], data[:, j+1], c=colors, cmap=cmap, s=5,alpha=0.2)
                

    [axes[n_dimensions-2, i].set_xlabel(label[i]) for i in range(n_dimensions-1)]
    [axes[n_dimensions-2, i].set_yticklabels([]) for i in range(1,n_dimensions-1)]
    [axes[j, 0].set_ylabel(label[j+1]) for j in range(n_dimensions-1)]
    [axes[j, 0].set_xticklabels([]) for j in range(n_dimensions-2)]

    # Customize color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), anchor=(0.0,11.0), orientation='horizontal')
    cbar.set_label('Color', rotation=0, labelpad=15)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()

# usage of the colorplot
n_samples = 10000
n_dimensions = 5
data = np.random.randn(n_samples, n_dimensions)
colors = np.log(np.exp(-np.sum(data**2,axis=1)/2 ) )
label = ['var '+str(i) for i in range(n_dimensions)]
# get_colorplot(data, colors, label)

