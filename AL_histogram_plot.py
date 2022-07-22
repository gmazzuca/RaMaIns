import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import matplotlib.cm as cm

n = 2000
beta_values = [5,10,20]
bound_values = [2,1.5,1.2]

for beta,bound in zip(beta_values,bound_values):
    basename = "eig_AL_n_%d_beta_%0.1f" %(n,beta) # for AL
    eig = np.loadtxt('Total' + basename+'.dat', dtype=np.complex_)


    modulo_eig = np.abs(eig)
    x= np.real(eig)
    y = np.imag(eig)


    
    plt.figure()
    plt.hist(modulo_eig, bins=800, density=True)
    plt.title(r'Ablowitz-Ladik , $\beta =%0.1f$, $n=%d$' %(beta,n), fontsize = 18)
    plt.xlim((0,5))
    plt.xlabel(r'$|\lambda|$')
    plt.savefig('AL_focusing_modulo_kde_n_%d_beta_%0.1f.png' %(n,beta))

    plt.figure()
    plt.hist2d( x,y,bins=600, density=True,  cmap = 'jet', cmin = 0.001)
    plt.title(r'Ablowitz-Ladik , $\beta =%0.0f$, $n=%d$' %(beta,n), fontsize = 18)
    plt.xlabel(r'$\Re(\lambda)$')
    plt.ylabel(r'$\Im(\lambda)$', rotation=0)
    plt.xlim((-bound,bound))
    plt.ylim((-bound,bound))
    plt.savefig('AL_focusing_kde_n_%d_beta_%0.1f.png' %(n,beta))
    
    plt.figure()
    hist, xedges, yedges = np.histogram2d(x, y, bins=(300,300), density = True, range =[[-2,2],[-2,2]] )
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()

    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    personal_color_map =  lambda x : cmap(x) if (x>0) else (1,1,1,0.00001)
    # scale each z to [0,1], and get their rgb values
    rgba = [personal_color_map((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax.view_init(elev = 75, azim=45)
    plt.title(r'Ablowitz-Ladik , $\beta =%0.0f$, $n=%d$' %(beta,n), fontsize = 18)
    plt.xlabel(r"$\Re(\lambda)$",size = 14)
    plt.ylabel(r"$\Im(\lambda)$",size = 14)
    plt.savefig('AL_focusing_3d_v2_n_%d_beta_%0.1f.png' %(n,beta), trasparent = True)
