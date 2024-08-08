import numpy as np
import pickle
boundaries_dict = {
    # 'M': [np.log(5e5), np.log(1e7)],
    # 'mu': [np.log(1e-6), np.log(1e-4)],
    'M': [np.log(1e6), np.log(5e6)],
    'q': [np.log(1e-6), np.log(1e-4)],
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
pickle.dump(boundaries_dict, open('boundaries.pkl', 'wb'))