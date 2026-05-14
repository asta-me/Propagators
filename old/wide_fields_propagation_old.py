# -*- coding: utf-8 -*-
"""
Ricostruzione di campi ampi tramite ri-scalamento.


@author: astam
"""

import numpy as np
from numpy.fft import fft2,ifft2, ifftshift, fftshift
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt


#Fase output di rs_lines_Fresnel
test_phase=rs_phase;
# test_phase=rs_phase_lines;


"""Parameters"""
d=8e-6;                 #[m] pixelsize
d*=0.25
lam=532e-9;             #[m] wavelength
resx=1920;              #SLM x resolution
resy=1080;              #SLM y resolution
res=1080;
# Lx=15.36e-3; Ly=8.64e-3;
xcoords=np.linspace(-1.0,1.0,resy)*d*float(resy)/2.0;
ycoords=np.linspace(-1.0,1.0,resy)*d*float(resy)/2.0;
x,y=np.meshgrid(xcoords,ycoords);
                             
"Voglio ricostruire il piano coniugato a quello che sarebbe a distanza z"
"Con Demagnificazione"
z=0.2;
zmin=(d*res)/(2*np.tan( np.arcsin( lam / (2*d) ) )); #SLM zmin
zconj=1 / ((1 / zmin) - (1 / z));
Magnification=zconj/z;
print(" Magnification = " + str(Magnification))

"Creo fase di una lente divergente"
max_lens_phase = - np.pi*(x**2+y**2)/(lam*zmin);
#marco=1*np.exp(1j*max_lens_phase);
#plt.imshow(np.angle(marco))

"Aggiungo all'ologramma di fase di una lente divergente"
phase_lensed=((test_phase+max_lens_phase) % (2*np.pi)) - np.pi;

"Il campo da propagare è esponenziale con la fase ottenuta + 2xZeroPadding"
u_toprop=1*np.exp(1j*phase_lensed);
u_toprop=np.pad(u_toprop, pad_width=((540, 540), (540, 540)))

"Plotto con dimensioni Fisiche"
extent_toprop=[-u_toprop.shape[0]*d/2,u_toprop.shape[0]*d/2,-u_toprop.shape[1]*d/2,u_toprop.shape[1]*d/2];
plt.figure(1);
plt.imshow(np.angle(u_toprop), extent=extent_toprop)
plt.title("Holo phase + Lens");

"Propago il campo di zconj"
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
    k=2*np.pi/wvl
    
    fx= fy = np.arange(-1/(2*dx), 1/(2*dx), 1/L)
    FX, FY = np.meshgrid(fx,fy)
    
    H=np.exp(-1.j*np.pi*wvl*z*(FX**2+FY**2))*np.exp(1.j*k*z)
    H=fftshift(H)
    
    U1=fft2(fftshift(u1))
    U2=H*U1

    u2=ifftshift(ifft2(U2))    
    
    return u2

u_propped=propTF(u_toprop,d,lam,zconj);

"Plotto con dimensioni fisiche"
plt.figure(2);
extent_propped=[x / Magnification for x in extent_toprop]
plt.imshow(np.abs(u_propped), extent=extent_propped)
plt.title("Reconstructed field at z="+ str(z));

"Proviamo a Plottare slice 2D del campo 3D"
steps=5;
zs=np.linspace(0.8, 1.2, steps)
zconj=1 / ((1 / zmin) - (1 / zs));
Magnification=zconj/zs;

u_3d=np.zeros(u_toprop.shape+(steps,));
extents_propped=np.zeros((4,steps));
for i in range(zs.shape[0]):
    extents_propped[:,i]=np.array([x / Magnification[i] for x in extent_toprop])
    u_3d[:,:,i]=np.abs(propTF(u_toprop,d,lam,zconj[i]))
    u_3d[:,:,i]=np.flipud(u_3d[:,:,i]) #Il coniugato è al contrario giustamente


fig,ax=plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.35);
xxx =plt.imshow(u_3d[:,:,0],extent=extents_propped[0])
# plt.get_current_fig_manager().window.state('zoomed')
plt.title("Propped Field")

axSlider1=plt.axes([0.1, 0.2, 0.8, 0.05])

slider1=Slider(axSlider1,
               'Z propagation',
               valmin=zs[0],
               valmax=zs[steps-1],
               valinit=zs[0],
               valstep=np.abs(zs[steps-1]-zs[0])/(steps-1))

def val_update(val):
    i=np.where(zs==slider1.val)[0][0]
    xxx.set_data(u_3d[:,:,i])
    xxx.set_extent(extents_propped[:,i])
    # xxx.set_title("Proppped Field at"+ str(zs[i]))
    plt.title("Propped Field at z= "+ str(zs[i]) + "m")
    plt.draw()

slider1.on_changed(val_update)
plt.show()        

    
# data=u_3d[:,:,1]
# ax1 = fig.add_subplot(121)
# ax1.imshow(data, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0,1,0,1])

# # show the 3D rotated projection
# ax2 = fig.add_subplot(122, projection='3d')
# Z=np.ones(x.shape);
# ax2.plot_surface(x, y, Z, rstride=1, cstride=1, facecolors=plt.cm.BrBG(data), shade=False)
