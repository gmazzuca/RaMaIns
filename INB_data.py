import numpy as np
import sys


if len(sys.argv) < 6:
    print('error, give me: n,k,beta,eta,trials,number\n')
    exit()

particles = int(sys.argv[1])
off_set = int(sys.argv[2])
beta = float(sys.argv[3])
eta  = float(sys.argv[4])
trials = int(sys.argv[5])
number  = int(sys.argv[6])
eig_add = []
eig_mul = []

for j in range(trials):
    u = np.random.chisquare(2*eta, size = particles)/beta*0.5
    # additive
    L_add = np.diag(u[:off_set],particles - off_set) + np.diag(u[off_set:], -off_set) + np.diag(np.ones(particles-1), 1) + np.diag([1],-particles + 1)
    eig_add = np.append(eig_add, np.linalg.eigvals(L_add))

    #multiplicative

    L_mul = np.diag(np.ones(off_set),particles - off_set) + np.diag(np.ones(particles - off_set), -off_set) + np.diag(u[:-1], 1) + np.diag(u[-1:],-particles + 1)
    eig_mul = np.append(eig_mul, np.linalg.eigvals(L_mul))
    if j % 100 == 0:
        np.savetxt('INB_additive_eig_n_%d_beta_%0.1f_eta_%0.1f_offset_%d_%05d.txt' %(particles,beta,eta,off_set,number), eig_add)
        np.savetxt('INB_multiplicative_eig_n_%d_beta_%0.1f_eta_%0.1f_offset_%d_%05d.txt' %(particles,beta,eta,off_set,number), eig_mul)

np.savetxt('INB_additive_eig_n_%d_beta_%0.1f_eta_%0.1f_offset_%d_%05d.txt' %(particles,beta,eta,off_set,number), eig_add)
np.savetxt('INB_multiplicative_eig_n_%d_beta_%0.1f_eta_%0.1f_offset_%d_%05d.txt' %(particles,beta,eta,off_set,number), eig_mul)

