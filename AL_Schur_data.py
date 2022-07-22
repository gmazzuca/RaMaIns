!pip install icecream
import numpy as np
from numpy.random import standard_cauchy as Cauchy
import scipy
from scipy.special import gamma as Gamma
import numba
from numba import vectorize,jit,float32,float64,complex128,complex64,guvectorize
from numpy.random import uniform
from scipy.stats import gumbel_r
import matplotlib.cm as cm

# General Function
@vectorize([float64(complex128),float32(complex128)])
def my_abs(x):
    # Compute the modulus of a complex number
    return np.sqrt(x.real**2 + x.imag**2)

  
@vectorize([float64(complex128),float32(complex128)])
def abs2(x):
    return x.real**2 + x.imag**2
  
  
#Specific functions
  

  def reject_sampling(n,beta):
    m_tilde = Gamma(beta + 1) * np.sqrt(np.pi)/Gamma(beta+0.5)
    accepted_samples = []
    gap = n
    f_over_g_m_tilde = lambda x : 1/(1+x*x)**(beta)


    while (gap > 0):
        temporary_samples = Cauchy(int(gap*5))
        uniform_samples = np.random.uniform(low=0.0, high=1.0, size=len(temporary_samples))
        
        id_accepted_samples, = np.where(uniform_samples <= f_over_g_m_tilde(temporary_samples))
        accepted_samples = np.append(accepted_samples, temporary_samples[id_accepted_samples])

        gap = n - len(accepted_samples)

    return accepted_samples[:n]


def inverse_sampling(n,beta):
    theta = np.random.uniform(low=0.0, high=2*np.pi, size=n)
    x = np.random.uniform(low=0.0, high=1.0,size = n)
    F_inv = lambda x : np.sqrt((1-x)**(-1/beta) - 1)
    rho = F_inv(x)

    return rho*np.exp(1j*theta)

def create_E(alpha):
    fun_rho = lambda x : np.sqrt(1 - abs2(x))
    rho = fun_rho(alpha)
    n = len(alpha)
    L = np.zeros((n,n), dtype=complex)
    M = np.zeros((n,n), dtype=complex)

    even_entries = np.arange(0,n,2)
    odd_entries = np.arange(1,n,2)
    L[even_entries,even_entries] = -np.conjugate(alpha[even_entries])
    L[odd_entries,odd_entries] = alpha[even_entries]
    L[odd_entries,even_entries] = rho[even_entries]
    L[even_entries,odd_entries] = rho[even_entries]

    
    M[odd_entries,odd_entries] = -np.conjugate(alpha[odd_entries])
    M[even_entries[1:],even_entries[1:]] = alpha[odd_entries[:-1]]
    M[odd_entries[:-1],even_entries[1:]] = rho[odd_entries[:-1]]
    M[even_entries[1:],odd_entries[:-1]] = rho[odd_entries[:-1]]
    M[0,0] = -np.conjugate(alpha[odd_entries[-1]])
    M[0,n-1] = rho[odd_entries[-1]]
    M[n-1,0] = rho[odd_entries[-1]]


    return np.dot(L,M)
  
  
  
  #Main
  
n = 3000
beta = 2
trials = 100
eig_AL = []
eig_S = []

for k in range(trials):
    alpha_AL = inverse_sampling(n,beta)
    E_AL = create_E(alpha_AL)
    eig_AL =np.append(eig_AL,np.linalg.eigvals(E_AL))
    
    alpha_S = reject_sampling(n,beta)
    E_S = create_E(alpha_S)
    eig_S =np.append(eig_S,np.linalg.eigvals(E_S))


np.savetxt('eigenvalues_S_focusing_beta_%0.3f_n_%d_trials_%d_%05d.txt' %(beta,n,trials, number), eig_S)
np.savetxt('eigenvalues_AL_focusing_beta_%0.3f_n_%d_trials_%d_%05d.txt' %(beta,n,trials, number), eig_AL)


