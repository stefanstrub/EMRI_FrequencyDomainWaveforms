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
from search_utils import *

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

few_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=use_gpu,
    return_list=False,
    # frame='source',
)

# def inner_product(a,b,dt):
#     a_tilde = np.fft.rfft(a)*dt
#     b_tilde = np.fft.rfft(b)*dt
#     freq = np.fft.rfftfreq(len(a),dt)
#     df = freq[1]-freq[0]
#     psd_f = sens_fn(freq)
#     return 4.0 * np.real ( np.sum( np.conj(a_tilde) * b_tilde * df / psd_f) )


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

data_stream = pickle.load(open("data_channels_fd_noisy.pkl", 'rb'))
boundaries_all = pickle.load(open('boundaries.pkl', 'rb'))
boundaries_all = np.array(list(boundaries_all.values()))

maxiter = 200
# found_parameters_list, values = search_emri_pe(data_stream,boundaries_all, parameters, function_to_optimize='timefrequency', maxiter=maxiter, template="fd",emri_kwargs=waveform_kwargs)
# found_parameters = found_parameters_list[np.argmax(values)]
# pickle.dump(found_parameters, open('found_parameters_maxiter'+str(maxiter)+'.pkl', 'wb'))
found_parameters = pickle.load(open('found_parameters_maxiter'+str(maxiter)+'.pkl', 'rb'))

print('found parameters', found_parameters)
print('injected parameters', emri_injection_params)

found_parameters_in = np.copy(found_parameters)
found_parameters_in[1] = np.log(found_parameters[1] / found_parameters[0])# mass ratio
found_parameters_in[0] = np.log(found_parameters[0]) # log of M mbh
emri_injection_params_in = np.copy(emri_injection_params)
emri_injection_params_in[1] = np.log(emri_injection_params[1] / emri_injection_params[0])# mass ratio
emri_injection_params_in[0] = np.log(emri_injection_params[0]) # log of M mbh
print('found parameters M q', found_parameters_in)
print('injected parameters M q', emri_injection_params_in)


boundaries_reduced = np.copy(boundaries_all)
boundaries_reduced[0] = [14.506,14.509]
boundaries_reduced[1] = [-11.51,-11.525]
boundaries_reduced[3] = [7.535,7.536]
boundaries_reduced[4] = [0.345,0.355]


found_parameters_list, values = search_emri_pe(data_stream,boundaries_reduced, parameters, function_to_optimize='loglikelihood', maxiter=500, initial_params=found_parameters, template="fd",emri_kwargs=waveform_kwargs, injected_params=emri_injection_params)
found_parameters = found_parameters_list[np.argmax(values)]

for parameter in parameters:
    print(parameter, np.round(found_parameters[parameters.index(parameter)],4), np.round(emri_injection_params[parameters.index(parameter)],4))
print('found parameters', found_parameters)
print('injected parameters', emri_injection_params)
print('parameter', 'recovered', 'injected')
for parameter in parameters:
    print(parameter, np.round(sample[parameters.index(parameter)],4), np.round(injection_params_linear[parameters.index(parameter)],4))
sample_fd = fft_td_gen(*sample, **emri_kwargs)
f_mesh, t_mesh, sig_Z = sp.signal.stft(xp.fft.irfft(xp.array(sig_fd[0])).get(), 1/dt, nperseg=5000)
f_mesh, t_mesh, sample_Z = sp.signal.stft(xp.fft.irfft(xp.array(sample_fd[0])).get(), 1/dt, nperseg=5000)
plt.figure()
plt.imshow(np.abs(sample_Z[40:200,2:-2]), aspect='auto', origin='lower')
plt.colorbar()
