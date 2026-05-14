# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:54:30 2023

Classic Recontruction on multiple planes 

@author: astam
"""

#%% ====== Field Reconstruction  ===

"""Field Reconstruction in Different Propagation Planes"""

# 2 Possible propagators
def propTF(u1,dx,wvl,z):
    """
    u1 - source plane field
    L - source and observation plane side length
    wvl - wavelength
    z - propagation distance
    u2 - observation plane field
    """
    (M,N)=u1.shape
    #dx=L/M
    L=M*dx
    k=2*numpy.pi/wvl
    
    fx= fy = numpy.arange(-1/(2*dx), 1/(2*dx), 1/L)
    FX, FY = numpy.meshgrid(fx,fy)
    
    H=numpy.exp(-1.j*numpy.pi*wvl*z*(FX**2+FY**2))*numpy.exp(1.j*k*z)
    H=fftshift(H)
    
    U1=fft2(fftshift(u1))
    U2=H*U1

    u2=ifftshift(ifft2(U2))    
    
    return u2

def propIR(u1,dx,wvl,z):
    """
    u1 - source plane field
    L - source and observation plane side length
    wvl - wavelength
    z - propagation distance
    u2 - observation plane field
    """
    (M,N)=u1.shape
    L=M*dx
    k=2*numpy.pi/wvl
    
    x = y = numpy.arange(-L/2, L/2, dx)
    X, Y = numpy.meshgrid(x,y)
    
    h=1/(1.j*wvl*z)*numpy.exp(1.j*k/(2*z)*(X**2+Y**2))*numpy.exp(1.j*k*z)
    H=fft2(fftshift(h))*dx**2
        
    U1=fft2(fftshift(u1))
    U2=H*U1

    u2=ifftshift(ifft2(U2))    
    
    return u2


#Main 
u1=1*numpy.exp(1j*rs_phase);
u1=numpy.pad(u1, pad_width=((540, 540), (540, 540)));

zmin=-1.2; zmax=-0.9; steps=4;
zs=numpy.linspace(zmin,zmax,steps);
u_prop=numpy.zeros(u1.shape+(steps,));
for i in range(steps):
    u_prop[:,:,i]=numpy.abs(propTF(u1,d,lam,zs[i]));
        
fig, ax=plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.35);
xxx =plt.imshow(numpy.abs(propTF(u1,d,lam,zmin)),extent=[-d*res, d*res, -d*res, d*res])
# plt.axis([0, 10, 0, 100])

axSlider1=plt.axes([0.1, 0.2, 0.8, 0.05])
slider1=Slider(axSlider1,
               'Z propagation',
               valmin=zs[0],
               valmax=zs[steps-1],
               valinit=zs[0],
               valstep=numpy.abs(zs[steps-1]-zs[0])/(steps-1))

def val_update(val):
    i=numpy.where(zs==slider1.val)[0][0]
    xxx.set_data(u_prop[:,:,i])
    plt.draw()

slider1.on_changed(val_update)
plt.show()    
