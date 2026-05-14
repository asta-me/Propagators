import numpy as np

def propagate_field(u1,dx,lam,z):
    """
    Propagates a field using Fresnel approaximation
        u1: np.array, source plane field
        dx: float, pixel size [m]
        lam: float, wavelength in [m]
        z - propagation distance [m]
    Returns:
        u2: observation plane field
    """
    from numpy.fft import fft2,ifft2, ifftshift, fftshift
    (M,N) = u1.shape
    L = M*dx
    k = 2*np.pi/lam
    fx = fy = np.arange(-1/(2*dx), 1/(2*dx), 1/L)
    FX, FY = np.meshgrid(fx,fy)
    H = np.exp(-1.j*np.pi*lam*z*(FX**2+FY**2))*np.exp(1.j*k*z)
    H = fftshift(H)
    U1 = fft2(fftshift(u1))
    U2 = H*U1
    u2 = ifftshift(ifft2(U2))    
    return u2

def propagate_hologram(phase, d, lam, res, z):
    """
    Propagates a field starting from a wavefront phase, i.e. the SLM plane,
    adding a parabolic (lens) phase for efficient propagation.
        phase: np.array, phase of the hologram on the SLM
        d: pixel size of the SLM [m]
        lam: wavelenght [m]
        res: number of pixels in SLM in lateral direction
        z - propagation distance [m]
    Returns:
        u_0 : np.array, complex field to be propagated (w padding and lens) 
        extent_0: (l, r, bot, top) in data coordinates, for the field u_0
        u_1: np.array, complex field after propagation  
        extent_0: (l, r, bot, top) in data coordinates, for the field u_0
    """
    phase_2 = np.kron(phase, np.ones((2, 2))) # nearest interpolation with size 2
    xcoords = np.linspace( -1.0, 1.0, 2 * res ) * d * res / 2
    x,y = np.meshgrid(xcoords,xcoords)
    zmin = (d*res)/(2*np.tan( np.arcsin( lam / (2*d) ) )); # SLM zmin
    lens_phase = - np.pi*(x**2+y**2)/(lam*zmin); # lens phase profile
    phase_3 = ((phase_2+lens_phase) % (2*np.pi)) - np.pi; # phase with added lens profile
    u_0 = 1 * np.exp( 1j * phase_3 ); # field to propagate
    u_0 = np.pad( u_0 , pad_width = (phase.shape, phase.shape) )
    extent_0 = np.asarray([-1,1,-1,1])  * d * res 
    #Proapagate at a distance zconj
    zconj = 1 / ((1 / zmin) - (1 / z))
    scale_fact = zconj / z
    u_1 = propagate_field(u_0, d/2 , lam , zconj)
    extent_1 = extent_0 / scale_fact ;
    extent_1 = [x / scale_fact for x in extent_0]
 
    return u_0, u_1, extent_0, extent_1

def propagate_hologram_pupil(phase, d, lam, res, z):
    """
    Propagates a field starting from a wavefront phase, i.e. the SLM plane,
    adding a parabolic (lens) phase for efficient propagation.
        phase: np.array, phase of the hologram on the SLM
        d: pixel size of the SLM [m]
        lam: wavelength [m]
        res: number of pixels in SLM in lateral direction
        z: propagation distance [m]
    Returns:
        u_0 : np.array, complex field to be propagated (with padding and lens)
        extent_0: (l, r, bot, top) in data coordinates, for the field u_0
        u_1: np.array, complex field after propagation  
        extent_1: (l, r, bot, top) in data coordinates, for the field u_0
    """
    # Nearest interpolation with size 2
    phase_2 = np.kron(phase, np.ones((2, 2))) 
    
    # Coordinate grid
    xcoords = np.linspace(-1.0, 1.0, 2 * res) * d * res / 2
    x, y = np.meshgrid(xcoords, xcoords)
    
    # Lens phase
    zmin = (d * res) / (2 * np.tan(np.arcsin(lam / (2 * d))))  # SLM zmin
    lens_phase = -np.pi * (x**2 + y**2) / (lam * zmin)  # lens phase profile
    
    # Phase with added lens profile
    phase_3 = ((phase_2 + lens_phase) % (2 * np.pi)) - np.pi  
    
    # Circular mask
    circular_mask = (np.sqrt(x**2 + y**2) <= d * res / 2).astype(float)
    
    # Apply circular mask to the field
    u_0 = circular_mask * np.exp(1j * phase_3)  
    
    # Pad the field
    pad_width = (phase_2.shape[0] // 2, phase_2.shape[1] // 2)
    u_0 = np.pad(u_0, pad_width=(pad_width, pad_width), mode='constant', constant_values=0)
    
    extent_0 = np.asarray([-1, 1, -1, 1]) * d * res 
    
    # Propagate at a distance zconj
    zconj = 1 / ((1 / zmin) - (1 / z))
    scale_fact = zconj / z
    u_1 = propagate_field(u_0, d / 2, lam, zconj)
    extent_1 = extent_0 / scale_fact
    extent_1 = [x / scale_fact for x in extent_0]
    
    return u_0, u_1, extent_0, extent_1
