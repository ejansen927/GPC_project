import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import findpeaks as FP
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from scipy.stats import qmc

pi = np.pi

@jit
def NormalizeSpins(SpinConf):
    """
    Normalizes each spin in the 3D array SpinConf so that |spin| = 1.
    SpinConf shape: (3, Nx, Ny).
    """
    multiply = np.conjugate(SpinConf) * SpinConf
    Norm = multiply[0, :, :] + multiply[1, :, :] + multiply[2, :, :]
    return SpinConf / np.sqrt(Norm)

@jit
def InitializeSpins(Nx, Ny, Ferro=True):
    """
    Creates an initial spin configuration on an Nx x Ny lattice.
    Ferro=True sets all spins along +z. Otherwise, random directions.
    """
    SpinConf = np.zeros((3, Nx, Ny))
    if Ferro:
        SpinConf[2, :, :] = 1.0  # +z direction
    else:
        # Random spin directions
        u = 1 - 2 * np.random.rand(Nx, Ny)
        phi = np.random.rand(Nx, Ny) * 2 * pi
        x = np.sqrt(1 - u**2) * np.cos(phi)
        y = np.sqrt(1 - u**2) * np.sin(phi)
        z = u
        SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = x, y, z
    return SpinConf

@jit
def PBC(site, size):
    """
    Periodic boundary condition. Returns (site + size) % size.
    """
    return (site + size) % size

# ---------------------- Various Spin Ansätze ---------------------- #

def PiPi(Nx, Ny, theta, phi, SpinConf):
    """
    (π, π) spiral: one variational parameter theta for the spin inclination,
    and phi offsets the global phase in the spiral.
    """
    theta = 2 * pi * theta
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    Qr = 2 * pi * (0.5 * i + 0.5 * j + phi)
    Sx = np.sin(theta) * np.cos(Qr)
    Sy = np.sin(theta) * np.sin(Qr)
    Sz = np.cos(theta)
    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def Stripe(Nx, Ny, theta, SpinConf):
    """
    (π, 0) or (0, π) stripe phase. One variational parameter theta for inclination.
    """
    theta = 2 * pi * theta
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    Qr = 2 * pi * (0.5 * i)
    Sx = np.sin(theta) * np.cos(Qr)
    Sy = np.sin(theta) * np.sin(Qr)
    Sz = np.cos(theta)
    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def ConicalSpiral(Nx, Ny, Qvec, theta, SpinConf):
    """
    Single-Q conical spiral, which can reduce to FM if theta=0 or Qvec=0.
    """
    theta = 2 * pi * theta
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    Qr = 2 * pi * (Qvec[0] * i + Qvec[1] * j)
    Sx = np.sin(theta) * np.cos(Qr)
    Sy = np.sin(theta) * np.sin(Qr)
    Sz = np.cos(theta)
    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def ConicalSpiral_FM(Nx, Ny, Qvec, phi, theta, SpinConf):
    """
    Single-Q conical spiral with an additional phase shift phi.
    """
    theta = 2 * pi * theta
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    Qr = 2 * pi * (Qvec[0] * i + Qvec[1] * j + phi)
    Sx = np.sin(theta) * np.cos(Qr)
    Sy = np.sin(theta) * np.sin(Qr)
    Sz = np.cos(theta)
    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def VerticalSpiral(Nx, Ny, a1, a2, qv, m, SpinConf):
    """
    Vertical spiral with amplitude parameters a1, a2, wavevector qv, and offset m.
    """
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    Sx = np.zeros((Nx, Ny))
    Sy = a1 * np.sin(2 * pi * qv * i)
    Sz = a2 * np.cos(2 * pi * qv * i) + m
    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def TriangularLatticeSkyrmions(Nx, Ny, a1, a2, m, qv, phi, alpha, SpinConf):
    """
    Triangular-lattice skyrmion ansatz from Batista's paper.
    """
    phi = 2 * pi * phi
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

    q1r = 2 * pi * qv * (np.cos(2*pi*alpha)*i - np.sin(2*pi*alpha)*j)
    q2r = 2 * pi * qv * j
    q3r = -2 * pi * qv * (np.cos(2*pi*alpha)*i + np.sin(2*pi*alpha)*j)

    Sx = a1 * (
        np.cos(phi) * np.sin(q1r)
        + np.cos(phi + 2*pi/3) * np.sin(q2r)
        + np.cos(phi + 4*pi/3) * np.sin(q3r)
    )
    Sy = a1 * (
        np.sin(phi) * np.sin(q1r)
        + np.sin(phi + 2*pi/3) * np.sin(q2r)
        + np.sin(phi + 4*pi/3) * np.sin(q3r)
    )
    Sz = -a2 * (np.cos(q1r) + np.cos(q2r) + np.cos(q3r)) + m

    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def MultipleConicalSpiral(Nx, Ny, a1, a2, a3, qv, m, alpha, SpinConf):
    """
    Multiple-Q conical spiral on a triangular lattice from Batista's approach.
    """
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    q1r = 2 * pi * qv * (np.cos(2*pi*alpha)*i - np.sin(2*pi*alpha)*j)
    q2r = 2 * pi * qv * j
    q3r = -2 * pi * qv * (np.cos(2*pi*alpha)*i + np.sin(2*pi*alpha)*j)

    Sx = a1 * np.sin(q1r) + a2 * np.sin(q3r)
    Sy = a1 * np.cos(q1r) - a2 * np.cos(q3r)
    Sz = a3 * np.cos(q2r) + m

    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def MultipleVerticalSpiral(Nx, Ny, a1, a2, qv, m, phi, alpha, SpinConf):
    """
    Multiple-Q vertical spiral from Batista's approach.
    """
    phi = 2 * pi * phi
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

    q1r = 2 * pi * qv * (np.cos(2*pi*alpha)*i - np.sin(2*pi*alpha)*j)
    q2r = 2 * pi * qv * j
    q3r = -2 * pi * qv * (np.cos(2*pi*alpha)*i + np.sin(2*pi*alpha)*j)

    Sx = (
        a1*np.cos(phi) * np.sin(q2r)
        + a2*np.sin(phi) * (np.cos(q1r) + np.cos(q3r))
    )
    Sy = (
        a1*np.sin(phi) * np.sin(q2r)
        - a2*np.cos(phi) * (np.cos(q1r) + np.cos(q3r))
    )
    Sz = m - a2 * np.cos(q2r)

    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def SquareLatticeSkyrmions(Nx, Ny, Q1vec, Q2vec, theta1, theta2, az, mz, SpinConf):
    """
    Square-lattice skyrmions. Two wavevectors Q1vec, Q2vec, and parameters
    theta1, theta2, az, mz control the configuration.
    """
    j, i = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    Q1 = 2 * pi * (Q1vec[0] * i + Q1vec[1] * j + theta1)
    Q2 = 2 * pi * (Q2vec[0] * i + Q2vec[1] * j + theta2)
    Sx = -np.cos(Q1) + np.cos(Q2)
    Sy = -np.cos(Q1) - np.cos(Q2)
    Sz = -az * (np.sin(Q1) + np.sin(Q2)) + mz
    SpinConf[0, :, :], SpinConf[1, :, :], SpinConf[2, :, :] = Sx, Sy, Sz
    return NormalizeSpins(SpinConf)

def InitializeFerroZ(Nx, Ny):
    """
    All spins pointing in the +z direction.
    """
    SpinConf = np.zeros((3, Nx, Ny))
    SpinConf[2, :, :] = 1.0
    return SpinConf

# ---------------------- Build Spin Config from Params ---------------------- #

def SpinConfFromParams(params, Nx, Ny, ansatz):
    """
    Constructs a spin configuration from 'params' for a given 'ansatz'.
    Nx, Ny are the local lattice sizes.
    """
    SpinConf = np.zeros((3, Nx, Ny))

    if ansatz == 'PiPi':
        SpinConf = PiPi(Nx, Ny, params[0], params[1], SpinConf)
    elif ansatz == 'Stripe':
        SpinConf = Stripe(Nx, Ny, params[0], SpinConf)
    elif ansatz == 'CS':
        Qvec = [params[0], params[1]]
        SpinConf = ConicalSpiral(Nx, Ny, Qvec, params[2], SpinConf)
    elif ansatz == 'CS-FM':
        Qvec = [params[0], params[1]]
        SpinConf = ConicalSpiral_FM(Nx, Ny, Qvec, params[2], params[3], SpinConf)
    elif ansatz == 'VS':
        SpinConf = VerticalSpiral(Nx, Ny, params[0], params[1], params[2], params[3], SpinConf)
    elif ansatz == 'TLS':
        SpinConf = TriangularLatticeSkyrmions(
            Nx, Ny, params[0], params[1], params[2],
            params[3], params[4], params[5], SpinConf
        )
    elif ansatz == 'MCS':
        SpinConf = MultipleConicalSpiral(
            Nx, Ny, params[0], params[1], params[2],
            params[3], params[4], params[5], SpinConf
        )
    elif ansatz == 'MVS':
        SpinConf = MultipleVerticalSpiral(
            Nx, Ny, params[0], params[1], params[2],
            params[3], params[4], params[5], SpinConf
        )
    elif ansatz == 'SLS':
        Q1vec = [params[0], params[1]]
        Q2vec = [params[2], params[3]]
        SpinConf = SquareLatticeSkyrmions(
            Nx, Ny, Q1vec, Q2vec,
            params[4], params[5], params[6], params[7],
            SpinConf
        )
    else:
        sys.exit(
            "Unsupported ansatz. Supported: PiPi, Stripe, CS, VS, TLS, "
            "SLS, MCS, MVS, CS-FM."
        )
    return SpinConf

# ---------------------- Energy Calculation ---------------------- #

##@jit
def CalculateSpinEnergy(SpinConf, Nx, Ny, J_params):
    """
    Calculates lattice spin energy with up to 3rd-neighbor couplings and
    an external field / anisotropy.
    J_params = [J1, J2, J3, H_a, A].
    """
    J1, J2, J3, H_a, A = J_params

    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    l_n = PBC(i - 1, Nx)
    r_n = PBC(i + 1, Nx)
    u_n = PBC(j + 1, Ny)
    d_n = PBC(j - 1, Ny)
    l2_n = PBC(i - 2, Nx)
    r2_n = PBC(i + 2, Nx)
    u2_n = PBC(j + 2, Ny)
    d2_n = PBC(j - 2, Ny)

    # 1st-neighbor dot products
    NN = np.sum(
        SpinConf[:, i, j]
        * (
            SpinConf[:, l_n, j]
            + SpinConf[:, r_n, j]
            + SpinConf[:, i, u_n]
            + SpinConf[:, i, d_n]
        ),
        axis=0
    )

    # 2nd-neighbor (diagonal) dot products
    NNN = np.sum(
        SpinConf[:, i, j]
        * (
            SpinConf[:, l_n, u_n]
            + SpinConf[:, r_n, u_n]
            + SpinConf[:, l_n, d_n]
            + SpinConf[:, r_n, d_n]
        ),
        axis=0
    )

    # 3rd-neighbor (next-nearest in same row/column)
    NNNN = np.sum(
        SpinConf[:, i, j]
        * (
            SpinConf[:, l2_n, j]
            + SpinConf[:, r2_n, j]
            + SpinConf[:, i, u2_n]
            + SpinConf[:, i, d2_n]
        ),
        axis=0
    )

    Energy = (
        -0.5 * (J1 * NN + J2 * NNN + J3 * NNNN)
        - H_a * SpinConf[2, i, j]
        + A * (SpinConf[2, i, j]) ** 2
    )

    return np.sum(Energy) / (Nx * Ny)

# ---------------------- Scipy Minimization Wrappers ---------------------- #

def function_to_minimize_for_scipy(X, other_args):
    """
    Scipy-compatible objective. X is the array of variational parameters.
    other_args = (Nx, Ny, J_params, ansatz).
    """
    Nx_, Ny_, J_params, ansatz = other_args
    SpinConf = SpinConfFromParams(X, Nx_, Ny_, ansatz)
    return CalculateSpinEnergy(SpinConf, Nx_, Ny_, J_params)

def minimize_wrapper(params, other_args):
    """
    Local minimization from a given starting 'params' using BFGS.
    """
    
    bounds=[(-1,1)]*len(params)
    options = {'disp': False, 'maxiter': 500}
    """
    result = minimize(
        function_to_minimize_for_scipy,
        params,
        args=other_args,
        method='BFGS',
        tol=1e-9,
        options=options
    )
    """
    result = minimize(
        function_to_minimize_for_scipy,
        params,
        args=other_args,
        method='L-BFGS-B',
        tol=1e-9,
        bounds=bounds,
        options=options
    )
    return result.x, result.fun

def parallel_minimize(data, *other_params):
    """
    Parallelize multiple local minimizations from different initial guesses.
    Returns the best parameters found and the corresponding energy.
    """
    best_x = None
    best_value = float('inf')

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(minimize_wrapper, x, other_params) for x in data]
        for future in futures:
            result_x, result_value = future.result()
            if result_value < best_value:
                best_x = result_x
                best_value = result_value

    return best_x, best_value

# ---------------------- Main Black-Box Function ---------------------- #

def blackboxfunc(X, Nx, Ny):
    """
    Given a set of coupling constants / parameters X = [J1, J2, J3, H_a, A],
    tries multiple ansätze in parallel to find the minimum-energy state.
    Returns the phase name (rough guess) and the corresponding spin config.
    Nx, Ny are the local lattice sizes.
    """
    J1, J2, J3, H_a, A = X
    J_params = [J1, J2, J3, H_a, A]

    # Ferro spin reference
    FerroSpin = InitializeSpins(Nx, Ny, Ferro=True)
    E_ferro = CalculateSpinEnergy(FerroSpin, Nx, Ny, J_params)

    # Candidate phases to explore
    #phases = ['PiPi', 'Stripe', 'CS-FM', 'VS', 'MCS', 'MVS', 'TLS']
    phases = [ 'CS-FM', 'VS', 'MCS', 'MVS', 'TLS']
    Energy = 100 * np.ones(len(phases))
    Energy[0] = E_ferro
    phase_params = [[] for _ in phases]
    phase_params[0] = ["FM"]  # placeholder to track Ferro

    Ntrials = 3
    for z, phase in enumerate(phases):
        # Decide how many parameters
        if phase == 'Stripe':
            nparams = 1
        elif phase == 'PiPi':
            nparams = 2
        elif phase == 'CS':
            nparams = 3
        elif phase in ['CS-FM', 'VS']:
            nparams = 4
        elif phase in ['TLS', 'MCS', 'MVS']:
            nparams = 6
        elif phase == 'SLS':
            nparams = 8
        else:
            continue

        other_args = (Nx, Ny, J_params, phase)

        # Choose Sobol sample size (m=MM)
        if phase in ['TLS', 'MCS', 'MVS']:
            MM = 9
        elif phase in ['PiPi', 'Stripe']:
            MM = 4
        else:
            MM = 4 #5

        for _ in range(Ntrials):
            sampler = qmc.Sobol(d=nparams, scramble=True)
            #see to get reproducible result
            #sampler = qmc.Sobol(d=nparams, scramble=True, seed=12345)

            data = sampler.random_base2(m=MM)

            # Slight random shift for alpha in triangular-based phases
            if phase in ['TLS', 'MCS', 'MVS']:
                for i in range(len(data)):
                    data[i, -1] = 1./12 + (np.random.rand() - 0.5) * 0.05

            params_best, E_best = parallel_minimize(data, other_args)
            if E_best < Energy[z]:
                Energy[z] = E_best
                phase_params[z] = params_best

    print(f"All energy {Energy}")
    E_min = np.min(Energy)
    minphase_index = np.argmin(Energy)

    print(f"Min E {E_min}, and E_ferro {E_ferro}")
    if abs(E_min - E_ferro) < 1e-4:
        final_phase = 'FM'
        FinalSpinConf = InitializeFerroZ(Nx, Ny)
    else:
        phase_temp = phases[minphase_index]
        print(f"Phase detected from energy minimum {phase_temp}")
        chosen_params = phase_params[minphase_index]
        FinalSpinConf = SpinConfFromParams(chosen_params, Nx, Ny, phase_temp)
        final_phase = FP.DeterminePhase(FinalSpinConf,phase_temp)

    PHASE_MAP = {
        'FM': 0,
        'SKYRMION': 1,
        'UNKNOWN': 2,
        'STRIPE': 3,
        'PI_PI': 4,
        'SINGLE_Q': 5,
        'DOUBLE_Q': 6,
        'COMPLEX': 7,
        'FM + STRIPE': 8,
        'FM + PI_PI': 9,
        'FM + SINGLE_Q': 10,
        'FM + DOUBLE_Q': 11,
        'FM + COMPLEX': 12
    }


    #return final_phase, FinalSpinConf
    final_class = PHASE_MAP[final_phase]
    return final_class, FinalSpinConf

if __name__ == '__main__':
    # Example usage: we specify Nx, Ny here (locally) instead of globally.
    Nx_local = 12 #256
    Ny_local = 12 #256

    # Example coupling constants
    #J1 = float(sys.argv[1])
    J1 = float(np.random.rand()*2-1)
    J2 = float(np.random.rand()*2-1)
    J3 = float(np.random.rand()*2-1)
    Ha = float(np.random.rand()*2-1)
    A = float(np.random.rand()*2-1)
    #J3 = float(sys.argv[2])
    #J2 = float(sys.argv[3])
    #Hs = 0.14
    #A = -a * Hs
    #a=float(sys.argv[4])
    #ha=float(sys.argv[5])

    nn=1
    for e in range(nn):
        start_t = time.time()
        #Ha = Hs * ha
        #A=-Hs*a #*e/nn
        X = [J1, J2, J3, Ha, A]
        #X=np.random.rand(5)
        print(f"Parameters {X}")

        phase, spin_conf = blackboxfunc(X, Nx_local, Ny_local)
        #print(f"Min E phase {phase} for parameters {X}")
        print(f"<<<<<< FINAL PHASE =  {phase}")

        fname = f"FinalSpin_{a}_{ha}.npy"
        #fname=f"FinalSpin.npy"
        np.save(fname, spin_conf)
        

        elapsed = time.time() - start_t
        print(f"Time taken: {elapsed:.3f} seconds")


