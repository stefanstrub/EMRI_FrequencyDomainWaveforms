import sys
import numpy as np
import scipy as sp
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
import corner
from lisatools.utils.utility import AET

from eryn.moves import StretchMove, GaussianMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *
import multiprocessing as mp


# from lisatools.sensitivity import get_sensitivity
from FDutils import *
from scipy.signal.windows import (
    blackman,
    blackmanharris,
    hamming,
    hann,
    nuttall,
    parzen,
)

from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral
from few.summation.interpolatedmodesum import CubicSplineInterpolant

from eryn.utils import TransformContainer

import time
import matplotlib.pyplot as plt
from few.utils.constants import *
import pickle

SEED = 2601996
np.random.seed(SEED)
dev = 0


# load Power Spectral Density of LISA. This represents how loud the instrumental noise is.
noise = np.genfromtxt('GRAPPA_EMRI_tutorial-main/LPA.txt', names=True)
f, PSD = (
    np.asarray(noise["f"], dtype=np.float64),
    np.asarray(noise["ASD"], dtype=np.float64) ** 2,
)

# here we use a cubic spline to interpolate the PSD
sens_fn = CubicSplineInterpolant(f, PSD, use_gpu=False)

# def inner_product(a,b,dt):
#     a_tilde = np.fft.rfft(a)*dt
#     b_tilde = np.fft.rfft(b)*dt
#     freq = np.fft.rfftfreq(len(a),dt)
#     df = freq[1]-freq[0]
#     psd_f = sens_fn(freq)
#     return 4.0 * np.real ( np.sum( np.conj(a_tilde) * b_tilde * df / psd_f) )


def get_noise(N,dt):
    freq = np.fft.rfftfreq(N,dt)
    noise = np.fft.irfft(np.random.normal(0.0,np.sqrt(sens_fn(freq)))+1j*np.random.normal(0.0,np.sqrt(sens_fn(freq))))/np.sqrt(dt*4/N)
    print(inner_product(noise,noise,dt)/N)
    return noise


request_gpu = True
if request_gpu:
    try:
        import cupy as xp
        # set GPU device
        xp.cuda.runtime.setDevice(dev)
        use_gpu = True
    except (ImportError, ModuleNotFoundError) as e:
        import numpy as xp
        use_gpu = False
else:
    import numpy as xp
    use_gpu = False

print("GPU available", use_gpu)
import warnings

warnings.filterwarnings("ignore")

few_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=use_gpu,
    return_list=False,
    # frame='source',
)


def transform_mass_ratio(logM, logeta):
    return [np.exp(logM), np.exp(logM) * np.exp(logeta)]


few_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=use_gpu,
    return_list=True,
    # frame='source',
)

td_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, odd_len=True),
    use_gpu=use_gpu,
    return_list=True,
    # frame='source',
)

td_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, odd_len=True),
    use_gpu=use_gpu,
    return_list=False,
    # frame='source',
)

parameters = ['M', 'mu', 'a', 'p0', 'e0', 'x0', 'dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']

def transform_parameters_to_01(params, boundaries):
    """
    Transform parameters from the prior space to unit cube
    """
    params01 = (params - boundaries[:, 0]) / (boundaries[:, 1] - boundaries[:, 0])
    return params01

def transform_parameters_from_01(params01, boundaries):
    """
    Transform parameters from the unit cube to the prior space
    """
    params = params01 * (boundaries[:, 1] - boundaries[:, 0]) + boundaries[:, 0]
    return params

def from_01_to_loglikelihood(params01_reduced, *args):
    """
    Transform parameters from the unit cube to the prior space and compute the SNR
    """
    transform_fn, data_stream, boundaries, fd_inner_product_kwargs, emri_kwargs, fd_gen = args
    params = transform_parameters_from_01(params01_reduced, boundaries)
    sample = transform_fn.both_transforms(params[None, :])[0]
    # generate FD waveforms
    data_channels_fd = fd_gen(*sample, **emri_kwargs)
    # compute the likelihood
    return -float(inner_product(data_channels_fd, data_stream, normalize=True, **fd_inner_product_kwargs))


def from_01_to_timefrequency_loglikelihood(params01_reduced, *args):
    """
    Transform parameters from the unit cube to the prior space and compute the SNR
    """
    transform_fn, data_Z, PSD_arr, boundaries, emri_kwargs, fd_gen, x_diff, sum_data_y, max_frequency_index = args
    params = transform_parameters_from_01(params01_reduced, boundaries)
    sample = transform_fn.both_transforms(params[None, :])[0]
    # generate FD waveforms
    data_channels_fd = fd_gen(*sample, **emri_kwargs)
    # compute the likelihood

    f_mesh, t_mesh, sample_Z = sp.signal.stft(xp.fft.irfft(xp.array(data_channels_fd[0])).get(), 1/dt, nperseg=5000)

    y = np.divide(np.abs(data_Z[:max_frequency_index,2:-2]) * np.abs(sample_Z[:max_frequency_index,2:-2]),np.array([PSD_arr]).T) # assumes right summation rule
    sample_y = np.divide(np.abs(sample_Z[:max_frequency_index,2:-2]) * np.abs(sample_Z[:max_frequency_index,2:-2]),np.array([PSD_arr]).T) # assumes right summation rule
    # sample_y_beyond_merger = np.divide(np.abs(sample_Z[:max_frequency_index,-230:-2]) * np.abs(sample_Z[:max_frequency_index,-230:-2]),np.array([PSD_arr]).T) # assumes right summation rule
    # y = np.abs(data_Z[:max_frequency_index,2:-2]) * np.abs(sample_Z[:max_frequency_index,2:-2])
    # sample_y = np.abs(sample_Z[:max_frequency_index,2:-2]) * np.abs(sample_Z[:max_frequency_index,2:-2])

    sum_y = 4 * xp.sum(x_diff * y)
    # sum_data_y = 4 * xp.sum(x_diff * data_y)
    sum_sample_y = 4 * xp.sum(x_diff * sample_y)
    # sum_sample_y_beyond_merger = 4 * xp.sum(x_diff * sample_y_beyond_merger)
    out = sum_y/np.sqrt(sum_data_y*sum_sample_y)#-sum_sample_y_beyond_merger
    return -float(out)


# function call
def run_emri_pe(
    emri_injection_params,
    Tobs,
    dt,
    fp,
    ntemps,
    nwalkers,
    injectFD=1,
    template="fd",
    emri_kwargs={},
    downsample=False,
    window_flag=True,
    # number of MCMC steps
    nsteps = 10_000,
):
    (
        M,
        mu,
        a,  # 2
        p0,
        e0,
        x0,  # 5
        dist,  # 6
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,  # 12
        Phi_r0,
    ) = emri_injection_params
    
    # for transforms
    # this is an example of how you would fill parameters
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
        "ndim_full": 14,
        "fill_values": np.array(
            [dist, qS, phiS, qK, phiK, Phi_theta0] # 0.0, x0
        ),  # spin and inclination and Phi_theta
        # "fill_inds": np.array([i for i, x in enumerate(parameters) if x in ['dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_theta0']]) #'a', 'x0'
        "fill_inds": np.array([i for i, x in enumerate(parameters) if x in []])
        #[2, 5, 6, 7, 8, 9, 10, 12]),
    }
    # sample_inds = np.array([i for i, x in enumerate(parameters) if x not in ['dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_theta0']])
    sample_inds = np.array([i for i, x in enumerate(parameters) if x not in []])

    # mass ratio
    emri_injection_params[1] = np.log(
        emri_injection_params[1] / emri_injection_params[0]
    )
    # log of M mbh
    emri_injection_params[0] = np.log(emri_injection_params[0])

    # remove three we are not sampling from (need to change if you go to adding spin)
    if len(fill_dict["fill_inds"]) > 0:
        emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])
    else:
        emri_injection_params_in = emri_injection_params
        fill_dict = None

    boundaries_dict = {
        # 'M': [np.log(5e5), np.log(1e7)],
        # 'mu': [np.log(1e-6), np.log(1e-4)],
        'M': [np.log(1e6), np.log(5e6)],
        'mu': [np.log(1e-6), np.log(1e-4)],
        'a': [0.0, 0.998],
        'p0': [7.2, 10.0],
        'e0': [0.001, 0.5],
        'x0': [0.0, 10],
        'dist': [1e-2, 1e2],
        'qS': [0, 10.0],
        'phiS': [0.0, 2*np.pi],
        'qK': [0, 10.0],
        'phiK': [0.0, 2*np.pi],
        'Phi_phi0': [0.0, 2*np.pi],
        'Phi_theta0': [0.0, 2*np.pi],
        'Phi_r0': [0.0, 2*np.pi]}
    
    

    boundaries_all = np.array(list(boundaries_dict.values()))
    boundaries = boundaries_all[sample_inds]
    parameters_reduced = np.array(parameters)[sample_inds]

    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(np.log(5e5), np.log(1e7)),  # M
                1: uniform_dist(np.log(1e-6), np.log(1e-4)),  # mass ratio
                2: uniform_dist(0.0, 0.998),  # a
                3: uniform_dist(9.0, 15.0),  # p0
                4: uniform_dist(0.001, 0.7),  # e0
                5: uniform_dist(0.0, 10.0),  # x0
                6: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                7: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        )
    }

    # sampler treats periodic variables by wrapping them properly
    periodic = {"emri": {4: 2 * np.pi, 5: np.pi}}

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    parameter_transforms = {
        (0, 1): transform_mass_ratio,
    }

    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
    )

    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]

    # generate FD waveforms
    data_channels_fd = few_gen(*injection_in, **emri_kwargs)
    
    # timing
    repeat = 1
    tic = time.perf_counter()
    [few_gen(*injection_in, **emri_kwargs) for _ in range(repeat)]
    toc = time.perf_counter()
    fd_time = toc-tic
    print('fd time', fd_time/repeat)

    signal1 = data_channels_fd
    tic = time.perf_counter()
    get_convolution(signal1,signal1)
    toc = time.perf_counter()
    fd_time = toc-tic
    print('get_convolution time', fd_time/repeat, "length of signal", len(signal1))
    
    tic = time.perf_counter()
    get_fft_td_windowed(signal1,signal1,dt)
    toc = time.perf_counter()
    fd_time = toc-tic
    print('get_fft_td_windowed time', fd_time/repeat, "length of signal", len(signal1))

    # frequency goes from -1/dt/2 up to 1/dt/2
    frequency = few_gen.waveform_generator.create_waveform.frequency
    positive_frequency_mask = frequency >= 0.0
    # transform into hp and hc
    emri_kwargs["mask_positive"] = True
    sig_fd = few_gen_list(*injection_in, **emri_kwargs)
    del emri_kwargs["mask_positive"]
    # non zero frequencies
    non_zero_mask = xp.abs(sig_fd[0]) > 1e-50
    # plt.figure(); plt.semilogy(frequency[positive_frequency_mask][non_zero_mask], label='non-zero'  ); plt.semilogy(frequency[positive_frequency_mask][~non_zero_mask], label='zero'  ); plt.legend(); plt.savefig('freq.pdf')
    # breakpoint()

    # generate TD waveform, this will return a list with hp and hc
    data_channels_td = td_gen_list(*injection_in, **emri_kwargs)
    noise = get_noise(len(data_channels_td[0])+2,dt)[:len(data_channels_td[0])]
    data_channels_td_noisy = [data_channels_td[0]+xp.array(noise), data_channels_td[1]+xp.array(noise)]
    data_channels_fd_noisy = [xp.fft.rfft(data_channels_td_noisy[0]), xp.fft.rfft(data_channels_td_noisy[1])]
    # save data stream using pickle
    pickle.dump(data_channels_fd_noisy, open("data_channels_fd_noisy.pkl", "wb"))


    # timing
    tic = time.perf_counter()
    [td_gen(*injection_in, **emri_kwargs) for _ in range(repeat)]
    toc = time.perf_counter()
    fd_time = toc-tic
    print('td time', fd_time/repeat)
    
    # windowing signals
    if window_flag:
        window = xp.asarray(hann(len(data_channels_td[0])))
        fft_td_gen = get_fd_waveform_fromTD(td_gen_list, positive_frequency_mask, dt, window=window)
        fd_gen = get_fd_waveform_fromFD(few_gen_list, positive_frequency_mask, dt, window=window)
    else:
        window = None
        fft_td_gen = get_fd_waveform_fromTD(td_gen_list, positive_frequency_mask, dt, window=window)
        fd_gen = get_fd_waveform_fromFD(few_gen_list, positive_frequency_mask, dt, window=window)

    # injections
    sig_fd = fd_gen(*injection_in, **emri_kwargs)
    sig_td = fft_td_gen(*injection_in, **emri_kwargs)


    # kwargs for computing inner products
    print("shape", sig_td[0].shape, sig_fd[0].shape)
    if use_gpu:
        fd_inner_product_kwargs = dict(
            PSD=xp.asarray(get_sensitivity(frequency[positive_frequency_mask].get())),
            use_gpu=use_gpu,
            f_arr=frequency[positive_frequency_mask],
        )
    else:
        fd_inner_product_kwargs = dict(
            PSD=xp.asarray(get_sensitivity(frequency[positive_frequency_mask])),
            use_gpu=use_gpu,
            f_arr=frequency[positive_frequency_mask],
        )

    print(
        "Overlap total and partial ",
        inner_product(sig_fd, sig_td, normalize=True, **fd_inner_product_kwargs),
        inner_product(sig_fd[0], sig_td[0], normalize=True, **fd_inner_product_kwargs),
        inner_product(sig_fd[1], sig_td[1], normalize=True, **fd_inner_product_kwargs),
    )

    print("frequency len", len(frequency), " make sure that it is odd")
    print("last point in TD", data_channels_td[0][-1])
    check_snr = snr(sig_fd, **fd_inner_product_kwargs)
    print("SNR = ", check_snr)

    # this is a parent likelihood class that manages the parameter transforms
    nchannels = 2
    if template == "fd":
        like_gen = fd_gen
    elif template == "td":
        like_gen = fft_td_gen

    # inject a signal
    if bool(injectFD):
        data_stream = sig_fd
    else:
        data_stream = sig_td



    # # short fourier transform of the signal
    # f, t, Zxx = sp.signal.stft(sig_td[0].get().real, 1/dt, nperseg=5000)
    # plt.figure(figsize=(16,10))
    # cb = plt.pcolormesh(t, f, np.log10(np.abs(Zxx)), shading='gouraud')
    # plt.colorbar(cb,)
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.yscale('log')
    # plt.ylim([1e-4, f[-1]])
    # plt.show()

    plt.figure()
    plt.plot(sig_td[0].get())
    plt.show()
    # plt.savefig(fp[:-3] + "injection.pdf")

    # compute the signal only for 1 week for 1 month prior to the merger
    start_time = 30/365.25
    end_time = 40/365.25
    start_time2 = 80/365.25
    end_time2 = 90/365.25
    year_in_seconds = 365.25*24*3600
    # get the index of the time window
    start_index = int(start_time*year_in_seconds / dt)
    end_index = int(end_time*year_in_seconds / dt)
    start_index2 = int(start_time2*year_in_seconds / dt)
    end_index2 = int(end_time2*year_in_seconds / dt)
    # get the window with zeros outside the time window
    window = xp.zeros(len(data_channels_td[0]))
    hann_window = xp.asarray(hann(end_index-start_index))
    window[start_index:end_index] = 1
    window[start_index2:end_index2] = 1
    # get the windowed signal
    fft_td_gen_windowed = get_fd_waveform_fromTD(td_gen_list, positive_frequency_mask, dt, window=window)
    fd_gen_windowed = get_fd_waveform_fromFD(few_gen_list, positive_frequency_mask, dt, window=window)
    fft_td_gen = get_fd_waveform_fromTD(td_gen_list, positive_frequency_mask, dt)
    fd_gen = get_fd_waveform_fromFD(few_gen_list, positive_frequency_mask, dt)
    # injections
    sig_fd_windowed = fd_gen_windowed(*injection_in, **emri_kwargs)
    sig_td_windowed = fft_td_gen_windowed(*injection_in, **emri_kwargs)

    # check duration of computation
    tic = time.perf_counter()
    [fft_td_gen_windowed(*injection_in, **emri_kwargs) for _ in range(3)]
    toc = time.perf_counter()
    print('fd time windowed', (toc-tic)/3)
    tic = time.perf_counter()
    [fd_gen_windowed(*injection_in, **emri_kwargs) for _ in range(3)]
    toc = time.perf_counter()
    print('fd time', (toc-tic)/3)




    if use_gpu:
        plt.figure()
        plt.loglog(np.abs(data_channels_fd_noisy[0].get()) ** 2)
        plt.loglog(np.abs(data_stream[0].get()) ** 2)
    else:
        plt.figure()
        plt.loglog(np.abs(data_stream[0]) ** 2)


    # check the SNR of the injected signal
    print("SNR = ", snr(data_stream, **fd_inner_product_kwargs))
    print("SNR = ", snr(data_channels_fd_noisy, **fd_inner_product_kwargs))
    print("SNR = ", snr(data_stream, data=data_channels_fd_noisy, **fd_inner_product_kwargs))
    print("SNR windowed = ", snr(sig_td, **fd_inner_product_kwargs))
    print("SNR windowed = ", snr(sig_td_windowed, **fd_inner_product_kwargs))
    
    print(
        "Overlap total and partial ",
        inner_product(sig_td[0], sig_td[0], normalize=False, **fd_inner_product_kwargs),
        inner_product(sig_fd[0], sig_td[0], normalize=True, **fd_inner_product_kwargs),
        inner_product(sig_fd[1], sig_td[1], normalize=True, **fd_inner_product_kwargs),
    )

    # time to compute the likelihood
    tic = time.perf_counter()
    [like_gen(*injection_in, **emri_kwargs) for _ in range(3)]
    toc = time.perf_counter()
    print('fd time', (toc-tic)/3)

    # time to compute the inner product
    tic = time.perf_counter()
    [inner_product(sig_fd, sig_td, normalize=True, **fd_inner_product_kwargs) for _ in range(3)]
    toc = time.perf_counter()
    print('fd time', (toc-tic)/3)

    def tf_product(a,b):
        y = func(temp1.conj() * temp2) / PSD_arr  # assumes right summation rule

        out += 4 * xp.sum(x_vals * y)
        return 

    # get parameters after transformation

    params01_reduced = transform_parameters_to_01(emri_injection_params[sample_inds], boundaries)
    args = (transform_fn, data_channels_fd_noisy, boundaries, fd_inner_product_kwargs, emri_kwargs, fd_gen)
    
    loglikelihood = from_01_to_loglikelihood(params01_reduced, *args)

    print('loglikelihood', loglikelihood)
    print('injected parameters reduced', emri_injection_params[sample_inds])
    print('injected parameters reduced 01', params01_reduced)
    # params01_reduced[0] += 10**-2
    initial_params01_reduced = np.random.uniform(0,1,len(params01_reduced))
    print('initial parameters reduced 01', initial_params01_reduced)
    params = transform_parameters_from_01(initial_params01_reduced, boundaries)
    sample = transform_fn.both_transforms(params[None, :])[0]

    sig_fd_windowed = fft_td_gen_windowed(*injection_in, **emri_kwargs)
    sig_fd = fft_td_gen(*injection_in, **emri_kwargs)
    sample_fd_windowed = fft_td_gen_windowed(*sample, **emri_kwargs)
    sample_fd = fft_td_gen(*sample, **emri_kwargs)
    f_mesh, t_mesh, data_Z = sp.signal.stft(xp.fft.irfft(xp.array(data_channels_fd_noisy[0])).get(), 1/dt, nperseg=5000)
    f_mesh, t_mesh, sig_Z = sp.signal.stft(xp.fft.irfft(xp.array(sig_fd[0])).get(), 1/dt, nperseg=5000)
    f_mesh, t_mesh, sample_Z = sp.signal.stft(xp.fft.irfft(xp.array(sample_fd[0])).get(), 1/dt, nperseg=5000)
    # plt.figure(figsize=(16,10))
    # plt.imshow(np.abs(sig_Z[:400,:640]), aspect='auto')

    PSD_arr = get_sensitivity(f_mesh[:200])
    y = np.real(sig_Z[:200,:640].conj() * sample_Z[:200,:640]) # assumes right summation rule
    y = np.divide(np.abs(sample_Z[:200,:640]) * np.abs(sample_Z[:200,:640]),np.array([PSD_arr]).T) # assumes right summation rule
    # y = np.divide(np.real(sig_Z[:400,:640].conj() * sample_Z[:400,:640]),np.array([PSD_arr]).T) # assumes right summation rule

    # plt.figure()
    # plt.loglog(f_mesh[:200],PSD_arr)
    # plt.show()
    # assumes right summation rule

    x_diff = float(xp.diff(f_mesh[:200])[1])
    out = 4 * xp.sum(x_diff * y)
    print('tf product', out)

    plt.figure()
    plt.imshow(np.abs(sample_Z[40:200,2:-2]), aspect='auto', origin='lower')
    plt.colorbar()


    # print('inner product', inner_product(sig_fd, sig_fd, normalize=True, **fd_inner_product_kwargs))
    # print('inner product', inner_product(sig_fd_windowed, sig_fd_windowed, normalize=True, **fd_inner_product_kwargs))
    # print('inner product', inner_product(sig_fd, sig_fd_windowed, normalize=True, **fd_inner_product_kwargs))
    print('inner product', inner_product(sig_fd, sample_fd, normalize=True, **fd_inner_product_kwargs))
    print('inner product windowed', inner_product(sig_fd_windowed, sample_fd_windowed, normalize=True, **fd_inner_product_kwargs))



    time_series = np.arange(len(data_channels_td[0]))*dt
    plt.figure()
    plt.plot(time_series[:-1],xp.fft.irfft(xp.array(sig_fd[0])).get())
    plt.plot(time_series[:-1],xp.fft.irfft(xp.array(sig_td_windowed[0])).get())
    plt.plot(time_series[:-1],xp.fft.irfft(xp.array(sample_fd[0])).get())
    # plt.plot(time_series[:-1],xp.fft.irfft(xp.array(sample_fd_windowed[0])).get())
    # plt.plot(time_series,window.get()*xp.fft.ifft(xp.array(sig_td[0])).get().max())
    plt.show()


    # short fourier transform of the signal

    # f_mesh, t_mesh, Zxx = sp.signal.stft(xp.fft.irfft(xp.array(sig_fd[0])).get(), 1/dt, nperseg=5000)
    f_mesh, t_mesh, Zxx = sp.signal.stft(xp.fft.irfft(xp.array(data_channels_fd_noisy[0])).get(), 1/dt, nperseg=5000)
    plt.figure(figsize=(16,10))
    cb = plt.pcolormesh(t_mesh, f_mesh, np.log10(np.abs(Zxx)), shading='gouraud')
    plt.colorbar(cb,)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yscale('log')
    plt.ylim([1e-4, f[-1]])
    plt.show()

    loglikelihood = from_01_to_loglikelihood(params01_reduced, *args)
    print('loglikelihood', loglikelihood)

    plt.figure()
    plt.imshow(np.abs(data_Z[40:200,2:-2]), aspect='auto', origin='lower')
    plt.colorbar()

    max_frequency_index = 200
    PSD_arr = get_sensitivity(f_mesh[:max_frequency_index])
    data_y = np.divide(np.abs(data_Z[:max_frequency_index,2:-2]) * np.abs(data_Z[:max_frequency_index,2:-2]),np.array([PSD_arr]).T) # assumes right summation rule
    # data_y = np.abs(data_Z[:max_frequency_index,2:-2]) * np.abs(data_Z[:max_frequency_index,2:-2]) # assumes right summation rule
    sum_data_y = 4 * xp.sum(x_diff * data_y)
    args_tf = (transform_fn, data_Z, PSD_arr, boundaries, emri_kwargs, fd_gen, x_diff, sum_data_y, max_frequency_index)
    tf_loglikelihood = from_01_to_timefrequency_loglikelihood(params01_reduced, *args_tf)
    tic = time.perf_counter()
    [from_01_to_timefrequency_loglikelihood(params01_reduced, *args_tf) for _ in range(3)]
    toc = time.perf_counter()
    print('tf time', (toc-tic)/3)
    print('tf loglikelihood', tf_loglikelihood)

    tic = time.perf_counter()
    found_parameters01 = []
    number_of_runs = 5
    for i in range(number_of_runs):
        result01 = sp.optimize.differential_evolution(
            from_01_to_timefrequency_loglikelihood,
            [(0.0, 1.0)]*len(initial_params01_reduced),
            args=args_tf,
            maxiter=100,
            strategy='best1bin',
            tol=1e-5,
            disp=True,
            polish=True,
            popsize=10,
            recombination=0.7,
            mutation=(0.5,1),
            x0=initial_params01_reduced,
            seed=47
        )
        print(result01)
        initial_params01_reduced = result01.x
        loglikelihood = from_01_to_loglikelihood(result01.x, *args)
        print('loglikelihood', loglikelihood)
        print('found parameters', transform_parameters_from_01(result01.x, boundaries))
        print('injected parameters', emri_injection_params[sample_inds])
        found_parameters01.append(result01.x)
        initial_params01_reduced = np.random.uniform(0,1,len(params01_reduced))
        print('time', (time.perf_counter()-tic)/(i+1))
    toc = time.perf_counter()
    print('time', (toc-tic)/number_of_runs)

    for result01 in found_parameters01:
        # print('found parameters', transform_parameters_from_01(result01, boundaries))
        print('tf loglikelihood', from_01_to_timefrequency_loglikelihood(result01, *args_tf))

    result01 = found_parameters01[0]
    initial_params01_reduced = result01
    adjusted_sample = np.copy(params01_reduced)
    adjusted_sample[0] = result01[0]
    adjusted_sample[1] = result01[1]
    adjusted_sample[3] = result01[3]
    adjusted_sample[4] = result01[4]

    print('loglikelihood', from_01_to_loglikelihood(params01_reduced, *args))
    print('loglikelihood found', from_01_to_loglikelihood(adjusted_sample, *args))
    
    params = transform_parameters_from_01(result01, boundaries)
    sample = transform_fn.both_transforms(params[None, :])[0]
    print(parameters)
    injection_params_linear = transform_fn.both_transforms(emri_injection_params[None, :])[0]
    print('found parameters', sample)
    print('injected parameters', transform_fn.both_transforms(emri_injection_params[None, :])[0])
    print('parameter', 'recovered', 'injected')
    for parameter in parameters:
        print(parameter, np.round(sample[parameters.index(parameter)],4), np.round(injection_params_linear[parameters.index(parameter)],4))
    sample_fd = fft_td_gen(*sample, **emri_kwargs)
    f_mesh, t_mesh, sig_Z = sp.signal.stft(xp.fft.irfft(xp.array(sig_fd[0])).get(), 1/dt, nperseg=5000)
    f_mesh, t_mesh, sample_Z = sp.signal.stft(xp.fft.irfft(xp.array(sample_fd[0])).get(), 1/dt, nperseg=5000)
    plt.figure()
    plt.imshow(np.abs(sample_Z[40:200,2:-2]), aspect='auto', origin='lower')
    plt.colorbar()


    # check which parameters influence the likelihood
    for i in range(len(params01_reduced)+1):
        params01_reduced_test = params01_reduced.copy()
        if i < len(params01_reduced_test):
            print('parameter', parameters_reduced[i])
            params01_reduced_test[i] -= 1e-1
        tf_loglikelihood = from_01_to_timefrequency_loglikelihood(params01_reduced_test, *args_tf)
        print('tf loglikelihood', tf_loglikelihood)
        loglikelihood = from_01_to_loglikelihood(params01_reduced_test, *args)
        print('loglikelihood', loglikelihood)
    # indifferent to the parameters:
    # loglikelihood a, x0, Phi_theta0
    # tf loglikelihood a, x0, dist, Phi_phi0, Phi_theta0, Phi_r0
    # tf loglikelihood small qS, phiS, qK, phiK

    # generate FD waveforms
    data_channels_fd = fd_gen(*sample, **emri_kwargs)
    # short fourier transform of the signal
    f_mesh, t_mesh, Zxx = sp.signal.stft(xp.fft.irfft(xp.array(sig_td[0])).get(), 1/dt, nperseg=5000)
    plt.figure(figsize=(16,10))
    cb = plt.pcolormesh(t_mesh, f_mesh, np.log10(np.abs(Zxx)), shading='gouraud')
    plt.colorbar(cb,)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yscale('log')
    plt.ylim([1e-4, f[-1]])
    plt.show()



def func(parameters, *data):

    #we have 3 parameters which will be passed as parameters and
    #"experimental" x,y which will be passed as data

    a,b,c = parameters
    x,y = data

    result = 0

    for i in range(len(x)):
        result += (a*x[i]**2 + b*x[i]+ c - y[i])**2

    return result**0.5
bounds = [(0.5, 1.5), (-0.3, 0.3), (-0.1, 0.1)]

#producing "experimental" data 
x = [i for i in range(6)]
y = [x**2 for x in x]

#packing "experimental" data into args
args = (x,y)

result = sp.optimize.differential_evolution(func, bounds=bounds, args=args)
print(result.x)



window_flag = bool(0)
downsample = int(0)
Tobs = 0.2  # years
dt = 10.0  # seconds
eps = 1e-2  # threshold mode content
injectFD = 0  # 0 = inject TD
template = 'td'  #'fd'

# set parameters
M = 2000000.0  # 1e6
a = 0.1  # will be ignored in Schwarzschild waveform
mu = 20.0  # 10.0
p0 = 13.709101864726545  # 12.0
p0 = 8.0  # 12.0
e0 = 0.35 #0.5794130830706371  # 0.35
x0 = 1.0  # will be ignored in Schwarzschild waveform
qK = np.pi / 3  # polar spin angle
phiK = np.pi / 3  # azimuthal viewing angle
qS = np.pi / 3  # polar sky angle
phiS = np.pi / 3  # azimuthal viewing angle
# the next lines normalize the distance to the SNR for the source analyze in the paper
# if window_flag:
#     dist = 1
# else:
#     dist = 2.4539054256
dist = .1
Phi_phi0 = np.pi / 3
Phi_theta0 = 0.0
Phi_r0 = np.pi / 3

m1 = M / (1 + mu)
m2 = M - m1

ntemps = 1
nwalkers = 16
nsteps = 1000

traj = EMRIInspiral(func="SchwarzEccFlux")

# fix p0 given T
tic = time.time()
[get_p_at_t(traj, Tobs * 0.9, [M, mu, a, e0, x0], index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5) for _ in range(10)]
toc = time.time()
print("time to get p0", (toc - tic) / 10)
p0 = get_p_at_t(
    traj,
    Tobs * 0.9,
    [M, mu, 0.0, e0, 1.0],
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    traj_kwargs={},
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=None,
)


print("new p0 fixed by Tobs", p0)

# name output
fp = f"./test_MCMC_M{M:.2}_mu{mu:.2}_p{p0:.2}_e{e0:.2}_T{Tobs}_eps{eps}_seed{SEED}_nw{nwalkers}_nt{ntemps}_downsample{int(downsample)}_injectFD{injectFD}_usegpu{str(use_gpu)}_template{template}_window_flag{window_flag}.h5"

emri_injection_params = np.array([
    M,  
    mu, 
    a,
    p0, 
    e0, 
    x0, 
    dist, 
    qS, 
    phiS, 
    qK, 
    phiK, 
    Phi_phi0, 
    Phi_theta0, 
    Phi_r0
])


waveform_kwargs = {
    "T": Tobs,
    "dt": dt,
    "eps": eps
}

run_emri_pe(
    emri_injection_params,
    Tobs,
    dt,
    fp,
    ntemps,
    nwalkers,
    emri_kwargs=waveform_kwargs,
    template=template,
    downsample=downsample,
    injectFD=injectFD,
    window_flag=window_flag,
    nsteps=nsteps,
)