import numpy as np
import sys
from scipy.signal import find_peaks
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import findtopologicalcharge as Topo 
from skimage.feature import peak_local_max

def Sq(spinconf):
    """
    Computes the structure factor S(q) for spinconf.
    spinconf has shape: (3, Nx, Ny).
    
    Returns:
       SF_norm: The normalized structure factor array with shape (Nx, Ny).
    """
    # spinconf is (3, Nx, Ny)
    _, Nx, Ny = spinconf.shape
    
    # Accumulate sum of squares of individual Fourier components for each spin
    # c_ij is the Fourier transform of spinconf[component].
    # We'll store the total power in SF.
    SF = np.zeros((Nx, Ny), dtype=np.float64)
    
    # 2D FFT on each spin component and sum of squares
    for comp in range(3):
        c_ij = np.fft.fft2(spinconf[comp])         # shape (Nx, Ny)
        SF += np.abs(c_ij)**2                     # sum of squares
    
    # Optionally do a shift for better visualization (comment out if not needed)
    # SF = np.fft.fftshift(SF)
    
    # sqrt is optional if you want the amplitude rather than power
    #SF = np.sqrt(SF)
    
    # Normalize to [0,1] range for display
    SF_norm = SF / SF.max()

    # Quick visualization
    fig, ax = plt.subplots()
    im = ax.imshow(SF_norm, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title('S(q)')
    im.set_clim(0, 1)
    fig.savefig("Sq.png", dpi=150, bbox_inches='tight')
    
    return SF_norm


def FindPeaks(spinconf):
    """
    Finds local maxima in the structure factor of spinconf.
    spinconf: shape (3, Nx, Ny)
    
    Returns:
       new_coordinates: array of peak coordinates in fractional units of 2*pi/N.
    """
    _, Nx, Ny = spinconf.shape
    sq = Sq(spinconf)
    
    #print(f"Peak intensity at origin: {sq[0,0]:.4f}, global max: {sq.max():.4f}")
    
    # (Optional) thresholding below 0.1 for clarity
    sq_threshold = sq.copy()
    sq_threshold[sq_threshold < 0.1] = 0
    
    # Find local maxima
    # We apply a maximum filter with size=3 (or bigger) to find local peaks
    image_max = ndi.maximum_filter(sq_threshold, size=1, mode='constant')
    coords = peak_local_max(image_max, min_distance=1, exclude_border=False)
    
    # Visualization of peaks
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
    ax0, ax1, ax2 = axes
    
    ax0.imshow(sq_threshold, origin='lower', cmap='gray')
    ax0.axis('off')
    ax0.set_title('Thresholded SF')

    ax1.imshow(image_max, origin='lower', cmap='gray')
    ax1.axis('off')
    ax1.set_title('Maximum Filter')

    ax2.imshow(sq_threshold, origin='lower', cmap='gray')
    ax2.plot(coords[:, 1], coords[:, 0], 'r.', markersize=4)
    ax2.axis('off')
    ax2.set_title('Peak Local Max')

    fig.tight_layout()
    fig.savefig("peaks.png", dpi=150, bbox_inches='tight')

    # Convert coordinates to fraction of the Brillouin zone, i.e. range ~ [-1, 1]
    new_coordinates = []
    for (row, col) in coords:
        # row ~ y-index, col ~ x-index
        # multiply by 2/Nx to get a fraction of 2*pi
        fx = col * 2.0 / Nx
        fy = row * 2.0 / Ny
        
        # Shift from [0,2) to [-1,1)
        if fx > 1.0:
            fx -= 2.0
        if fy > 1.0:
            fy -= 2.0
        
        new_coordinates.append([round(fx, 5), round(fy, 5)])
    
    return np.array(new_coordinates)


def CheckImage_ver0(spinconf):
    """
    Classifies the phase by analyzing peaks in the structure factor.
    spinconf: shape (3, Nx, Ny).

    Returns:
        phase (str): A guess at the phase based on the number/positions of peaks.
    """
    _, Nx, Ny = spinconf.shape
    peak_coords = FindPeaks(spinconf)  # returns array [[fx, fy], [fx, fy], ...] in [-1, 1)
    
    print("Peak positions:")
    print( peak_coords)
    if peak_coords.size == 0:
        print("No peaks found in S(q).")
        return "NO_PEAKS"
    
    # Sort peaks by distance from origin in reciprocal space
    dist_to_origin = np.linalg.norm(peak_coords, axis=1)
    sort_idx = np.argsort(dist_to_origin)
    peak_coords = peak_coords[sort_idx]
    
    # Check if the first peak is basically (0,0), indicating an FM component
    fm_peak_thresh = 1e-3
    fm_present = np.allclose(peak_coords[0], [0, 0], atol=fm_peak_thresh)

    # If FM peak is found, remove it from the analysis
    if fm_present:
        red_peaks = peak_coords[1:]
    else:
        red_peaks = peak_coords

    # Basic classification logic:
    num_peaks = len(red_peaks)

    if num_peaks == 0:
        # Then everything is basically at the origin
        phase = "FM" if fm_present else "UNKNOWN"

    #USE MY analytical solution here to determine single q phase
    elif num_peaks == 1:
        # Single-Q. Distinguish between known stripes or pi-pi if you like:
        qx, qy = red_peaks[0]
        # Example numeric checks:
        if np.allclose([abs(qx), abs(qy)], [1.0, 0.0], atol=0.05) or \
           np.allclose([abs(qx), abs(qy)], [0.0, 1.0], atol=0.05):
            phase = "STRIPE"
        elif np.allclose([abs(qx), abs(qy)], [1.0, 1.0], atol=0.05):
            phase = "PI_PI"
        else:
            phase = "SINGLE_Q"
    elif num_peaks == 2:
        # Possibly ±q pairs for single-Q spirals or stripe
        # e.g., if q1 + q2 ~ 0 => symmetrical about origin
        sum_vec = red_peaks[0] + red_peaks[1]
        if np.allclose(sum_vec, [0, 0], atol=0.1):
            #phase = "SINGLE_Q (±q) or STRIPE/PI-PI"
            phase = 'SINGLE_Q'
        else:
            phase = "MULTI_Q"
    elif num_peaks == 4:
        # Often double-Q or vortex/skyrmion
        phase = "DOUBLE_Q"
    else:
        # Could be more complicated multi-Q or partial skyrmion lattices
        phase = "COMPLEX"

    if fm_present and phase != "FM":
        phase = f"FM + {phase}"  # e.g., canted FM with additional peaks?

    print(f"Classified phase (based on {len(peak_coords)} peaks): {phase}")
    return phase


def CheckImage(spinconf):
    """
    Classifies the phase by analyzing peaks in the structure factor.
    spinconf: shape (3, Nx, Ny).

    Returns:
        phase (str): A guess at the phase based on the number/positions of peaks.
    """
    _, Nx, Ny = spinconf.shape
    peak_coords = FindPeaks(spinconf)  # returns array [[fx, fy], [fx, fy], ...] in [-1, 1)
    
    print("Peak positions:")
    print( peak_coords)
    if peak_coords.size == 0:
        print("No peaks found in S(q).")
        return "NO_PEAKS"
    
    # Sort peaks by distance from origin in reciprocal space
    dist_to_origin = np.linalg.norm(peak_coords, axis=1)
    sort_idx = np.argsort(dist_to_origin)
    peak_coords = peak_coords[sort_idx]
    
    # Check if the first peak is basically (0,0), indicating an FM component
    fm_peak_thresh = 1e-3
    fm_present = np.allclose(peak_coords[0], [0, 0], atol=fm_peak_thresh)

    # If FM peak is found, remove it from the analysis
    if fm_present:
        red_peaks = peak_coords[1:]
    else:
        red_peaks = peak_coords

    # Basic classification logic:
    num_peaks = len(red_peaks)

    if num_peaks == 0:
        # Then everything is basically at the origin
        phase = "FM" if fm_present else "UNKNOWN"

    #USE MY analytical solution here to determine single q phase
    elif num_peaks == 1:
        # Single-Q. Distinguish between known stripes or pi-pi if you like:
        qx, qy = red_peaks[0]
        # Example numeric checks:
        if np.allclose([abs(qx), abs(qy)], [1.0, 0.0], atol=0.01) or \
           np.allclose([abs(qx), abs(qy)], [0.0, 1.0], atol=0.01):
            phase = "STRIPE"
        elif np.allclose([abs(qx), abs(qy)], [1.0, 1.0], atol=0.01):
            phase = "PI_PI"
        else:
            phase = "SINGLE_Q"
    elif num_peaks == 2:
        # Possibly ±q pairs for single-Q spirals or stripe
        #sometimes pi-pi or pi-0 phases also do not converge and hence give 2 peaks close to the boundary. 
        #at any rate, there is +- symmetry, so it suffices to just look at one peak.
        # e.g., if q1 + q2 ~ 0 => symmetrical about origin

        qx, qy = red_peaks[0]

        if np.allclose([abs(qx), abs(qy)], [1.0, 0.0], atol=0.05) or \
           np.allclose([abs(qx), abs(qy)], [0.0, 1.0], atol=0.05):
            phase = "STRIPE"
        elif np.allclose([abs(qx), abs(qy)], [1.0, 1.0], atol=0.05):
            phase = "PI_PI"
        else:
            phase="SINGLE_Q"

    elif num_peaks == 4:
        # Often double-Q or vortex/skyrmion
        #there are 4 equal intensity peaks for double Q phase: 
        #the case of Multiple Vertical spiral phase is similar to skyrmion, multiple peaks may occur but the net topological charge is zero. Skyrmion phases can also have many peaks (but not euqla intensity). So need a check for skyrmion phase
        #A confusion is sometimes we get 4 very close peaks but they still belong to single Q phase (due to nono-convergence).
        #This happens when x coorindates are almost equal and all have same y-coorindate.. (comes from unconverged MVS phase)
        #check if the x-coordinates are same and y-coorindates are equal for all 4 peaks
            #temp1,temp2=0,0
            #for i in range(len(red_peak_cor)):
            #    temp1+=red_peak_cor[i,0]
            #    temp2+=red_peak_cor[i,1]
    ### Replacing this red_peak_cor with red_peaks
        ####m=0.5*(np.max(red_peak_cor[:,0]) - np.min(red_peak_cor[:,0]))
        #n=0.5*(np.max(red_peak_cor[:,1]) - np.min(red_peak_cor[:,1]))
        #print(m,n)
        m = 0.5 * (np.max(red_peaks[:, 0]) - np.min(red_peaks[:, 0]))
        n = 0.5 * (np.max(red_peaks[:, 1]) - np.min(red_peaks[:, 1]))

        if(abs(m) <0.05 or abs(n) <0.05):   #if the peaks are very close then it is just one peak
            phase="SINGLE_Q"
        else:
            phase = "DOUBLE_Q"

    else:
        # Could be more complicated multi-Q or partial skyrmion lattices
        phase = "COMPLEX"

    print(f"Classified phase (based on {len(peak_coords)} peaks): {phase}")
    return phase





def DeterminePhase(spinconf, possible_phase='TLS'):
    """
    This function checks the topological charge if relevant,
    then calls CheckImage(spinconf) to identify the phase from peaks.
    """
    # If you have a separate topological charge function, call it here:
    # topo_charge = Topo.SkyrmionNumber(spinconf, Nx)
    # ...

    if (possible_phase=='MVS' or possible_phase=='TLS'):
        topo_charge=np.sum(Topo.SkyrmionNumber(spinconf))
        #topo_charge=np.sum(Topo.SkyrmionNumber(spinconf,Nx))
        #chirality=np.sum(Topo.Chirality(spinconf,Nx))
        print(f"The topological charge is {topo_charge}")
        #print(f"The chirality number is {chirality}")
        if(abs(topo_charge)+0.2>=1):
            phase="SKYRMION"
            #phase=CheckImage(spinconf,Nx)
        else:
            phase=CheckImage(spinconf)
            #print("Possible Skyrmion but not quite??")
            #Just call this function: I want to see what i get for skyrmion phase
            #print(CheckImage(spinconf))

    else:
        phase=CheckImage(spinconf)
    
    #print(f"<<<<< FINAL PHASE = {phase}")
    return phase

    #phase = CheckImage(spinconf)
    #print(f"<<<<< FINAL PHASE = {phase}")
    #return phase


if __name__ == "__main__":
    # Example usage:
    f = sys.argv[1]  # e.g., 'FinalSpin_1.0_1.0.npy'
    spinconf = np.load(f)
    # spinconf should have shape (3, Nx, Ny)
    Nx = spinconf.shape[1]
    # possible_phase is for your reference
    possible_phase = 'TLS'
    
    DeterminePhase(spinconf, possible_phase)

