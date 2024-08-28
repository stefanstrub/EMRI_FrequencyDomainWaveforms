import os
os.environ["OMP_NUM_THREADS"] = str(1)
os.system("OMP_NUM_THREADS=1")
import sys

import numpy as np
from lisatools.utils.utility import AET
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *
import matplotlib.pyplot as plt
from lisatools.sensitivity import get_sensitivity

from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from few.summation.interpolatedmodesum import CubicSplineInterpolant

from few.waveform import GenerateEMRIWaveform
from eryn.utils import TransformContainer

from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import *
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

func = "SchwarzEccFlux"
traj = EMRIInspiral(func=func)

from fastlisaresponse import ResponseWrapper

from few.utils.constants import *

try:
    import cupy as xp
    gpu_available = True
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# from few.utils.utility import omp_set_num_threads
# omp_set_num_threads(4)

import warnings

warnings.filterwarnings("ignore")

# whether you are using 
use_gpu = True

def get_check_fft_plots(tmp_in, true_in, wave_gen, emri_kwargs, innkw, name='plotfft', window=None):
    if window is None:
        window = 1.0
    temp_data_channels = wave_gen(*tmp_in, **emri_kwargs)
    freq = xp.fft.rfftfreq(len(temp_data_channels[0]), innkw["dt"]).get()
    # get the expectation value of the loglike
    hh_here = inner_product(temp_data_channels,temp_data_channels, normalize=True,**innkw)
    # now do the same for the truth
    true_wave_channels = wave_gen(*true_in, **emri_kwargs)
    hh_truth = inner_product(true_wave_channels,temp_data_channels, normalize=True,**innkw)
    
    plt.figure(); 
    plt.title(f'h_in h_true {hh_truth}')
    plt.loglog(freq, xp.abs(xp.fft.rfft(true_wave_channels[0]*window)).get(),label='true')
    plt.loglog(freq, xp.abs(xp.fft.rfft(temp_data_channels[0]*window)).get(),label='best fit',alpha=0.8)
    plt.legend()
    plt.savefig(name)

    print("best point: h_in h_in", hh_here)
    print("truth point: h_in h_true", hh_truth)
    print("injected parameters",true_in)
    print("recovered parameters",tmp_in)

def create_response_EMRI(few_gen, Tobs, dt, use_gpu=True):

    orbit_file_esa = "./esa-trailing-orbits.h5"
    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

    tdi_gen = "1st generation"

    order = 25  # interpolation order (should not change the result too much)
    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AE",
    )  # could do "AET"

    index_lambda = 8
    index_beta = 7

    # with longer signals we care less about this
    t0 = 20000.0  # throw away on both ends when our orbital information is weird
    remove_garbage = "zero"
    wave_gen = ResponseWrapper(
        few_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage=remove_garbage,  # removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )
    return wave_gen


def create_response_GB(gb, Tobs, dt, use_gpu=True):

    orbit_file_esa = "./esa-trailing-orbits.h5"
    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

    tdi_gen = "1st generation"

    order = 25  # interpolation order (should not change the result too much)
    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AE",
    )  # could do "AET"

    
    index_lambda = 7
    index_beta = 8

    # with longer signals we care less about this
    t0 = 20000.0  # throw away on both ends when our orbital information is weird
    remove_garbage = "zero"
    gb_lisa_esa = ResponseWrapper(
    gb,
    Tobs,
    dt,
    index_lambda,
    index_beta,
    t0=t0,
    flip_hx=False,  # set to True if waveform is h+ - ihx
    use_gpu=use_gpu,
    remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
    is_ecliptic_latitude=True,  # False if using polar angle (theta)
    remove_garbage=remove_garbage,  # removes the beginning of the signal that has bad information
    **tdi_kwargs_esa,
    )
    return gb_lisa_esa


traj = EMRIInspiral(func="SchwarzEccFlux")
from few.utils.utility import get_p_at_t, get_mu_at_t

def get_p0(Tobs, M, mu, e0):
    # fix p0 given T
    try:
        p0 = get_p_at_t(
        traj,
        Tobs,
        [M, mu, 0.0, e0, 1.0],
        index_of_p=3,
        index_of_a=2,
        index_of_e=4,
        index_of_x=5,
        traj_kwargs={},
        xtol=2e-12,
        rtol=8.881784197001252e-16,
        bounds=[6+2*e0+0.1, 16.0],)
    except:
        # print("Tpl not found")
        return 16.0

    return p0

def get_Tplunge(M, mu, p0, e0):
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0, T=100.0, err=1e-6, use_rk4=False, dt=3600)
    return t[-1]/YRSID_SI

def get_mu(Tobs, M, p0, e0):
    # fix p0 given T
    try:
        mu = get_mu_at_t(
        traj,
        Tobs,
        [M, 0.0, p0, e0, 1.0],
        index_of_mu=1,
        traj_kwargs={},
        xtol=2e-10,
        rtol=8.881784197001252e-10,
        bounds=[1.0, 100.0],)
    except:
        return 0.1

    return mu

def objective_function(inputs, true_freqs):
    p, e = inputs
    if p < 6 + 2*e:
        return np.inf
    else:
        frs = get_fundamental_frequencies(0.0, p, e, 1.0)
        return (frs[0] - true_freqs[0])**2 + (frs[2] - true_freqs[1])**2

def get_p_e(true_freqs):
    result = minimize(
        objective_function, 
        x0 = [10., 0.3], 
        bounds=([9.5, 16.0],[0.01, 0.6]), 
        args = (true_freqs, ),
        method="Nelder-Mead", 
        options=dict(xatol=1e-8),  # xatol specifies tolerance on p,e
    ) 
    return result.x

def objOmphi(p, omegaPhi_e):
    omphi, e = omegaPhi_e
    if p < 6 + 2*e:
        return np.inf
    else:
        frs = get_fundamental_frequencies(0.0, p, e, 1.0)[0]
        return (frs - omphi)**2

def get_p_from_omegaPhi(omegaPhi_e):
    result = minimize(
        objOmphi, 
        x0 =10., 
        bounds=[[9.5, 16.0]], 
        args = (omegaPhi_e, ),
        method="Nelder-Mead", 
        options=dict(xatol=1e-8),  # xatol specifies tolerance on p,e
    ) 
    return result.x

def get_initial_cond(size, freq_bounds=None):
    Mhere = 10**np.random.uniform(5,7, size=size)
    e0here = np.sqrt(1-np.random.uniform(0.75,1.0, size=size)) # 
    Tpl = np.random.uniform(7/365,4.0,size=size)
    p0here = np.random.uniform(9.5,16.00, size=size) #  
    epshere = np.asarray([get_mu(tt, MM, mass, ecc)/MM for tt,MM,mass,ecc in zip(Tpl, Mhere, p0here, e0here)])
    f_phi = np.asarray([get_omegaPhi(np.log10(MM), pp, ee) for MM,pp,ecc in zip(Mhere, p0here, e0here)])
    if freq_bounds is not None:
        mask = (f_phi>freq_bounds[0])*(f_phi<freq_bounds[1])
        print(np.sum(mask)/size)
        return Mhere[mask],epshere[mask],p0here[mask],e0here[mask]
    else:
        return Mhere,epshere,p0here,e0here

def get_omegaPhi(log10M, p, e):
    return get_fundamental_frequencies(0.0, p, e, 1.0)[0]/(M*MTSUN_SI*np.pi*2.0)

def objective_function_Mpe(inputs, true_freqs):
    log10M, p, e = inputs
    M = 10**log10M
    if p < 6 + 2*e:
        return np.inf
    else:
        frs = get_fundamental_frequencies(0.0, p, e, 1.0)
        return (frs[0]/(M*MTSUN_SI*np.pi*2.0) - true_freqs)**2

def get_log10M_p_e(true_freqs):
    result = minimize(
        objective_function_Mpe, 
        x0 = [6.0, 10., 0.3], 
        bounds=([5, 7], [9.5, 16.0],[0.01, 0.6]), 
        args = (true_freqs, ),
        method="Nelder-Mead", 
        options=dict(xatol=1e-8),  # xatol specifies tolerance on p,e
    ) 
    return result.x

def get_newtonian_SymMassRatio(fdot_nodim, freq_phi_nodim):
    return (2*np.pi*freq_phi_nodim)**(-11/3) * fdot_nodim * 5 * np.pi/96

def get_newtownian_fdot(freq_phi_nodim,SymMassRatio):
    return 96/(5*np.pi) * (2*np.pi*freq_phi_nodim)**(11/3) * SymMassRatio


def get_log10fdot(logM, logeps, p0, e0, nu=1):
    M = np.exp(logM)
    mu = np.exp(logeps) * M

    # generate trajectory
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0, T=1/365/24, err=1e-6, use_rk4=False, dt=10.0)
    omega_phi,omega_theta,omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)
    out = []

    if len(t)<2:
        return -5.0
    
    fact = 1/(M*MTSUN_SI*np.pi*2)
    cs = CubicSpline(t, omega_phi * fact)

    return np.log10(cs.derivative(nu=nu)(0.0))

def obj_fun(logeps, fdot_p_e):
    
    log10fdot, logM, p, e = fdot_p_e
    
    if p < 6 + 2*e:
        return np.inf
    else:
        frs = get_log10fdot(logM, logeps, p, e)
        return np.log10((1-frs/log10fdot)**2)

def get_logeps(log10fdot, logM, p, e, tol=2e-4):
    
    bounds = [[np.log(1e-7),np.log(1e-3)]]
    result = minimize(
        obj_fun, 
        x0 = np.log(1e-5), 
        bounds = bounds, 
        args = ([log10fdot, logM, p, e],),
        method="COBYLA",
        tol=tol,
        ) 
    return result.x

from scipy.signal import butter, lfilter
# from cupyx.scipy.signal import butter
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def trace_track_to_plot(M, eps, p0, e0, num=200, Tmax=1.0, m=2, n=0):
    mu = eps * M

    # generate trajectory
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0, T=10.0)#, upsample=True)
    # time vector
    time_vec = np.linspace(0.0, YRSID_SI*Tmax, num=num)
    out = []
    # only inspiral within 10 years plung
    if (t[-1]<= 10*YRSID_SI):# and (t[-1]> 3.14e7):
        omega_phi,omega_theta,omega_r = get_fundamental_frequencies(0.0, p, e, 1.0)
        for el in [omega_phi,omega_r]:
            cs = CubicSpline(t, el)
            mask = (time_vec<t[-1])
            omega_phi_out = np.ones_like(time_vec)*0.0
            omega_phi_out[mask] = cs(time_vec[mask])
            out.append(omega_phi_out)
            return omega_phi_out[mask]/(M*MTSUN_SI*np.pi*2), time_vec[mask]
    else:
        return 0.0, 0.0

def plot_tracks(Mhere,epshere,p0here,e0here, name=None, truths=None):
    size = Mhere.shape[0]
    plt.figure()
    if truths is not None:
        om,t = trace_track_to_plot(*truths,Tmax=4.0)
        plt.semilogy(t,om,color='k')
    
    for i in np.arange(size):
        try:
            om,t = trace_track_to_plot(Mhere[i], epshere[i], p0here[i], e0here[i],Tmax=4.0)
            plt.semilogy(t,om,color='r',alpha=0.1)
        except:
            pass
    plt.xlim(0.0,3.14e7)
    if name is None:
        plt.savefig('tracks.png')
    else:
        plt.savefig(name)

# import time
# it = 0
# timing = 0.0
# error_avg = []
# fdot_val = []
# pars = []
# num=1000
# for _ in range(num):
#     print('----------------------')
#     M = 10**np.random.uniform(5,7)
#     mu = np.random.uniform(1,100)
#     e0 = np.random.uniform(0.1,0.5)
#     p0 = np.random.uniform(9.5,15.9)
#     eps = mu/M
#     pars.append([np.log(M), np.log(eps), p0, e0])
#     fdot = get_log10fdot(np.log(M), np.log(eps), p0, e0)
#     fdot_val.append(fdot)
#     tic = time.time()
#     out = get_logeps(fdot, np.log(M), p0, e0)
#     toc = time.time()
#     newfdot = get_log10fdot(np.log(M), out, p0, e0)
#     err = np.abs(1-out/np.log(eps))
#     timing += toc-tic 
#     error_avg.append(err)
#     if err > 1e-4:
#         it += 1
#         print(p0,e0)
#         print(err,1-np.log(eps)/out,timing)
# pars = np.asarray(pars)
# plt.figure(); plt.hist(np.log10(error_avg)); plt.show()
# print(it/num, timing/num,np.mean(error_avg))

from eryn.backends import HDFBackend,Backend

def get_samples_from_file(size,filename,priors):
    tmp = priors["emri"].rvs(size=size)

    run = HDFBackend(filename)
    samples = run.get_chain()['emri'][run.get_inds()['emri']]
    z = run.get_log_like()[run.get_inds()['emri'][...,0]]
    mask = (z > 4.0)
    
    ind = np.arange(samples.shape[0])[mask]
    np.random.shuffle(ind)
    
    return samples[ind[:size]]