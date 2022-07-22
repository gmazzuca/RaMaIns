import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

beta_values = [2,5]
eta_values = [1,1]
off_set_values = [2,2,5]
n = 1000
for (beta,eta,off_set) in zip(beta_values, eta_values,off_set_values):
        eig = np.loadtxt("INB_additive_eig_n_%d_beta_%0.1f_eta_%0.1f_offset_%d.txt" %(n,beta,eta,off_set), dtype = 'complex')
        x = np.real(eig)
        y = np.imag(eig)

        hist, xedges, yedges = np.histogram2d(x, y, bins=(300,300), density = True)


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
        cmap = cm.get_cmap('Blues') # Get desired colormap - you can change this!
        personal_color_map =  lambda x : cmap(x) if (x>0) else (1,1,1,0.00001)
        # scale each z to [0,1], and get their rgb values
        rgba = [personal_color_map((k-min_height)/max_height) for k in dz] 

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average', alpha = 0.3)
        ax.view_init(elev = 60, azim=30)
        phi = lambda x : eta/beta*np.exp(2*np.pi*1j*x*off_set) + np.exp(-2*np.pi*1j*x) 
        tfine = np.linspace(0,1,num=300)
        ax.plot3D(np.real(phi(tfine)), np.imag(phi(tfine)),np.zeros(len(tfine)), 'r-',linewidth = 4)
        plt.title(r"INB additive, $\beta = %d,\, \eta = %d,\, k=%d$" %(beta,eta,off_set),size = 18)
        plt.xlabel(r"$\Re(\lambda)$",size = 14)
        plt.ylabel(r"$\Im(\lambda)$",size = 14)
        plt.savefig('INB_additive_eig_beta_%0.1f_eta_%0.1f_offset_%d.png' %(beta,eta,off_set))
