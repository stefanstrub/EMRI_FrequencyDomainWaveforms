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
from LISAanalysistools.lisatools.diagnostic import *
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
    Transform parameters from the unit cube to the prior space and compute the loglikelihood
    """
    transform_fn, data_stream, boundaries, fd_inner_product_kwargs, emri_kwargs, fd_gen, normalize = args
    params = transform_parameters_from_01(params01_reduced, boundaries)
    sample = transform_fn.both_transforms(params[None, :])[0]
    # generate FD waveforms
    sample_channels_fd = fd_gen(*sample, **emri_kwargs)
    # compute the likelihood
    return -float(inner_product(sample_channels_fd, data_stream, normalize=normalize, **fd_inner_product_kwargs))

def from_01_to_SNR(params01_reduced, *args):
    """
    Transform parameters from the unit cube to the prior space and compute the SNR
    """
    transform_fn, data_stream, PSD_arr, boundaries, emri_kwargs, fd_gen, x_diff, sum_data_y = args
    params = transform_parameters_from_01(params01_reduced, boundaries)
    sample = transform_fn.both_transforms(params[None, :])[0]
    # generate FD waveforms
    sample_channels_fd = fd_gen(*sample, **emri_kwargs)
    # compute the SNR

    A = xp.real(xp.divide(data_stream[0] * xp.conj(sample_channels_fd[0]),xp.array([PSD_arr]).T)) # assumes right summation rule
    sample_A = xp.real(xp.divide(sample_channels_fd[0] * xp.conj(sample_channels_fd[0]),xp.array([PSD_arr]).T)) # assumes right summation rule
    E = xp.real(xp.divide(data_stream[1] * xp.conj(sample_channels_fd[1]),xp.array([PSD_arr]).T)) # assumes right summation rule
    sample_E = xp.real(xp.divide(sample_channels_fd[1] * xp.conj(sample_channels_fd[1]),xp.array([PSD_arr]).T)) # assumes right summation rule

    sum_AE = 4 * xp.sum(x_diff * A) + 4 * xp.sum(x_diff * E)
    # sum_data_y = 4 * xp.sum(x_diff * data_y)
    sum_sample_AE = 4 * xp.sum(x_diff * sample_A) + 4 * xp.sum(x_diff * sample_E)
    # sum_sample_y_beyond_merger = 4 * xp.sum(x_diff * sample_y_beyond_merger)
    out = sum_AE/np.sqrt(sum_sample_AE)#-sum_sample_y_beyond_merger
    return -float(out)


def from_01_to_timefrequency_loglikelihood(params01_reduced, *args):
    """
    Transform parameters from the unit cube to the prior space and compute the SNR
    """
    transform_fn, data_Z, PSD_arr, boundaries, emri_kwargs, fd_gen, x_diff, sum_data_y, max_frequency_index, dt = args
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
def search_emri_pe(data_stream,boundaries_all, parameters,
    template="fd",
    emri_kwargs={}, function_to_optimize='timefrequency', number_of_runs=1,maxiter=100, initial_params=None, injected_params=None):

    dt = emri_kwargs["dt"]
    # for transforms
    # this is an example of how you would fill parameters
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
        "ndim_full": 14,
        "fill_values": np.array([]),#[dist, qS, phiS, qK, phiK, Phi_theta0] # 0.0, x0),  # spin and inclination and Phi_theta
        # "fill_inds": np.array([i for i, x in enumerate(parameters) if x in ['dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_theta0']]) #'a', 'x0'
        "fill_inds": np.array([i for i, x in enumerate(parameters) if x in []])
        #[2, 5, 6, 7, 8, 9, 10, 12]),
    }
    # sample_inds = np.array([i for i, x in enumerate(parameters) if x not in ['dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_theta0']])
    sample_inds = np.array([i for i, x in enumerate(parameters) if x not in []])

    boundaries = boundaries_all[sample_inds]
    parameters_reduced = np.array(parameters)[sample_inds]

    # remove three we are not sampling from (need to change if you go to adding spin)
    if len(fill_dict["fill_inds"]) == 0:
        fill_dict = None
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
    if initial_params is not None:
        initial_params[1] = np.log(initial_params[1] / initial_params[0])# mass ratio
        initial_params[0] = np.log(initial_params[0]) # log of M mbh
        initial_params01_reduced = transform_parameters_to_01(initial_params, boundaries)
    else:
        initial_params01_reduced = np.random.uniform(0,1,len(sample_inds))
    print('initial parameters reduced 01', initial_params01_reduced)
    params = transform_parameters_from_01(initial_params01_reduced, boundaries)
    sample = transform_fn.both_transforms(params[None, :])[0]

    # generate FD waveforms
    initial_sample = few_gen(*sample, **emri_kwargs)
    # frequency goes from -1/dt/2 up to 1/dt/2
    frequency = few_gen.waveform_generator.create_waveform.frequency
    positive_frequency_mask = frequency >= 0.0

    # kwargs for computing inner products
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


    if use_gpu:
        plt.figure()
        plt.loglog(np.abs(data_stream[0].get()) ** 2)
    else:
        plt.figure()
        plt.loglog(np.abs(data_stream[0]) ** 2)


    fd_gen = get_fd_waveform_fromFD(few_gen_list, positive_frequency_mask, dt)
    args = (transform_fn, data_stream, boundaries, fd_inner_product_kwargs, emri_kwargs, fd_gen)
    


    f_mesh, t_mesh, data_Z = sp.signal.stft(xp.fft.irfft(xp.array(data_stream[0])).get(), 1/dt, nperseg=5000)

    # f_mesh, t_mesh, Zxx = sp.signal.stft(xp.fft.irfft(xp.array(sig_fd[0])).get(), 1/dt, nperseg=5000)
    f_mesh, t_mesh, Zxx = sp.signal.stft(xp.fft.irfft(xp.array(data_stream[0])).get(), 1/dt, nperseg=5000)
    plt.figure(figsize=(16,10))
    cb = plt.pcolormesh(t_mesh[2:-2], f_mesh[40:200], np.log10(np.abs(Zxx[40:200,2:-2])), shading='gouraud')
    plt.colorbar(cb,)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yscale('log')
    plt.ylim([1e-4, f[-1]])
    plt.show()

    plt.figure()
    plt.imshow(np.abs(data_Z[40:200,2:-2]), aspect='auto', origin='lower')
    plt.colorbar()

    x_diff = float(xp.diff(f_mesh[:200])[1])
    max_frequency_index = 200
    PSD_arr = get_sensitivity(f_mesh[:max_frequency_index])
    data_y = np.divide(np.abs(data_Z[:max_frequency_index,2:-2]) * np.abs(data_Z[:max_frequency_index,2:-2]),np.array([PSD_arr]).T) # assumes right summation rule
    # data_y = np.abs(data_Z[:max_frequency_index,2:-2]) * np.abs(data_Z[:max_frequency_index,2:-2]) # assumes right summation rule
    sum_data_y = 4 * xp.sum(x_diff * data_y)
    args_tf = (transform_fn, data_Z, PSD_arr, boundaries, emri_kwargs, fd_gen, x_diff, sum_data_y, max_frequency_index, dt)
    args_SNR = (transform_fn, data_stream, PSD_arr, boundaries, emri_kwargs, fd_gen, x_diff, sum_data_y)
    args = (transform_fn, data_stream, boundaries, fd_inner_product_kwargs, emri_kwargs, fd_gen, 'sig1')

    if function_to_optimize == 'timefrequency':
        function = from_01_to_timefrequency_loglikelihood
        arguments = args_tf
    elif function_to_optimize == 'loglikelihood':
        function = from_01_to_loglikelihood
        arguments = args
    elif function_to_optimize == 'SNR':
        function = from_01_to_SNR
        arguments = args_SNR


    if injected_params is not None:
        injected_params_in = np.copy(injected_params)
        injected_params_in[1] = np.log(injected_params[1] / injected_params[0])
        injected_params_in[0] = np.log(injected_params[0])
        injected_params01_reduced = transform_parameters_to_01(injected_params_in, boundaries)
        loglikelihood = from_01_to_loglikelihood(injected_params01_reduced, *args)
        print('loglikelihood', loglikelihood)
        SNR = from_01_to_SNR(injected_params01_reduced, *args_SNR)
        print('SNR', SNR)
        tic = time.perf_counter()
        SNR = from_01_to_SNR(injected_params01_reduced, *args_SNR)
        toc = time.perf_counter()
        print('time', (toc-tic))
        tic = time.perf_counter()
        loglikelihood = from_01_to_loglikelihood(injected_params01_reduced, *args)
        toc = time.perf_counter()
        print('time', (toc-tic))

    tic = time.perf_counter()
    found_parameters01 = []
    values = []
    for i in range(number_of_runs):
        result01 = sp.optimize.differential_evolution(
            function,
            [(0.0, 1.0)]*len(initial_params01_reduced),
            args=arguments,
            maxiter=maxiter,
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
        found_parameters01.append(result01.x)
        values.append(result01.fun)
        initial_params01_reduced = np.random.uniform(0,1,len(initial_params01_reduced))
        print('time', (time.perf_counter()-tic)/(i+1))
    toc = time.perf_counter()
    print('average time', (toc-tic)/number_of_runs)
    found_parameters = [transform_parameters_from_01(params01, boundaries) for params01 in found_parameters01]
    found_parameters = [transform_fn.both_transforms(params[None, :])[0] for params in found_parameters]
    return found_parameters, values


