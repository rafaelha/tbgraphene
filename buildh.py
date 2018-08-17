import numpy as np
import tinyarray as ta
import numpy as np
from numpy import sqrt, pi, exp, tanh
from numpy.random import rand
import matplotlib.pyplot as plt

class buildh:
    """
    This class builds a meanfield Hamiltonian for TBG assuming singlet pairing up 
    to nearest-neighbors.
    Use as follows:
    
    # create class. Nx defines discretization of k-space. mu is chemical potential
    tbg = buildh(Nx=10, mu=-1e-3)

    # tbg includes an array of Nx*3*Nx discretized points of the Brillouin zone
    # tbg.kx    x coordinates
    # tbg.ky    y coordinates

    # First, one needs to set all wanted meanfield order parameters. In total, there are
    # 18 meanfields
    tbg.dA = 0.5
    tbg.dB = 0.4
    tbg.d1m = 1
    # ...

    # The function buildh.matrix() returns an array with Hamiltonians for all k in the BZ
    H = tbg.matrix()

    # Here, H[:,:,0] is the 8-by-8 Hamiltonian for k = (tbg.kx[0], tbg.ky[0])

    # One can also plot the bandstructure of the tight-binding model
    # for the case of 0 mean field coupling
    tbg.bandstructure()
    """


    def __init__(self, Nx, U=0, U_=0, V=0, V_=0, mu=-1e-3):
        """
        Nx  -   Number of points along short axis of Brillouin zone (BZ is parallelogram)
                The number of points along the second axis (y-axis) is 3*Nx
        U   -   Onsite coupling, inter-valley
        Up  -   Onsite coupling, intra-valley
        V   -   nn coupling, inter-valley
        Vp  -   nn coupling, intra-valley
        """
        self.mu = mu

        self.U = U
        self.U_ = U_
        self.V = V
        self.V_ = V_

        """
        Initialze meanfield order parameters
        """
        # onsite inter-valley
        self.dA = 0     # \Delta_{A}
        self.dB = 0     # \Delta_{B}
        # onsite intra-valley
        self.dAp_ = 0   # \Delta_{A,+}'
        self.dAm_ = 0   # \Delta_{A,-}'
        self.dBp_ = 0   # \Delta_{B,+}'
        self.dBm_ = 0   # \Delta_{B,-}'
        # nn inter-valley
        self.d1p = 0    # \Delta_{1,+}
        self.d2p = 0    # \Delta_{2,+}
        self.d3p = 0    # \Delta_{3,+}
        self.d1m = 0    # \Delta_{1,-}
        self.d2m = 0    # \Delta_{2,-}
        self.d3m = 0    # \Delta_{3,-}
        # nn intra-valley
        self.d1p_ = 0    # \Delta_{1,+}'
        self.d2p_ = 0    # \Delta_{2,+}'
        self.d3p_ = 0    # \Delta_{3,+}'
        self.d1m_ = 0    # \Delta_{1,-}'
        self.d2m_ = 0    # \Delta_{2,-}'
        self.d3m_ = 0    # \Delta_{3,-}'

        """
        define lattice vectors [a1, a2] in real space
        and lattice vectors [b1, b2] in reciprocal space
        and nn-vectors [r1, r2, r3] in real space
        """
        self.a1 = ta.array( [sqrt(3)/2, 1/2] ) # real space
        self.a2 = ta.array( [0, 1] )
        self.b1 = ta.array( [4*pi/sqrt(3), 0] )  # reciprocal space
        self.b2 = ta.array( [-2*pi/sqrt(3), 2*pi] )
        self.r1 = ta.array( [1/sqrt(3), 0] ) # real space nn-vector
        self.r2 = ta.array( [-1/sqrt(3), 1] ) / 2
        self.r3 = ta.array( [-1/sqrt(3), -1] ) / 2

        """
        Create two 1D arrays with x- and y-coordinates of BZ.
        BZ is a parallelogram. Nx is number of discrete points in the 
        BZ along x. Ny=3*Nx.
        """
        def buildBZ(Nx):
            Ny = 3 * Nx

            I, J = np.meshgrid(np.arange(Nx), np.arange(Ny))
            I = I.T
            J = J.T

            z1 = ta.array( [0, 4*pi] )
            z2 = ta.array( [2*pi/sqrt(3), -2*pi/3] ) 

            KX = J*z1[0]/Ny + I*z2[0]/Nx
            KY = J*z1[1]/Ny + I*z2[1]/Nx

            # reshape k points as a 1d-array
            return KX.reshape((Nx*Ny,)), KY.reshape((Nx*Ny,))
        self.kx, self.ky = buildBZ(Nx)
        self.NN = len(self.kx)

        """
        Create arrays with various phase factors
        """
        self.__nkr1 =  -1j * (self.kx * self.r1[0] + self.ky * self.r1[1])  
        self.__nkr2 =  -1j * (self.kx * self.r2[0] + self.ky * self.r2[1]) 
        self.__nkr3 =  -1j * (self.kx * self.r3[0] + self.ky * self.r3[1]) 
        self.__exp_nkr1 = exp( self.__nkr1 )
        self.__exp_nkr2 = exp( self.__nkr2 )
        self.__exp_nkr3 = exp( self.__nkr3 )
        self.__exp_kr1 = self.__exp_nkr1.conj()
        self.__exp_kr2 = self.__exp_nkr2.conj()
        self.__exp_kr3 = self.__exp_nkr3.conj()

        """
        Build tight-binding hamiltonian H
        H[:,:,0] is 8-by-8 Hamiltonian at k=(kx[0], ky[0])
        """
        self.__data = np.loadtxt('eff_hopping_fu_ver2.dat')
        H0 = np.zeros((4,4,self.kx.size), dtype='complex128')
        H0[0:2,0:2] = self.__buildh0(1, self.kx, self.ky)
        H0[2:4,2:4] = self.__buildh0(-1, self.kx, self.ky)

        H0_bottom = np.zeros((4,4,self.kx.size), dtype='complex128')
        H0_bottom[0:2,0:2] = self.__buildh0(1, -self.kx, -self.ky)
        H0_bottom[2:4,2:4] = self.__buildh0(-1, -self.kx, -self.ky)

        self.H = np.zeros((8,8,self.kx.size), dtype='complex128') 
        self.H[0:4, 0:4] = H0
        self.H[4:8, 4:8] = -H0_bottom


    def __buildh0(self, xi, kx, ky):
        """
        creates 2x2 matrix of the Hamiltonian for a single valley xi=+-1
        """
        nkr1 =  -1j * (kx * self.a1[0] + ky * self.a1[1])  
        nkr2 =  -1j * (kx * self.a2[0] + ky * self.a2[1]) 
        mat = np.zeros((2,2,kx.size), dtype='complex128')
        def matrix_el(fromorb, toorb, xi):
            f = self.__data[ \
                            np.logical_and( self.__data[:,2]==(fromorb+1),\
                                            self.__data[:,3]==(toorb+1) ) ]
            m = f[:,0] # x index for lattice vector
            n = f[:,1] # y index for lattice vector
            t = f[:,4] + xi * 1j * f[:,5]
            exp_nk = exp( np.einsum('i,j', n, nkr1) + \
                          np.einsum('i,j', m, nkr2) + \
                          (fromorb - toorb) * (-1j * kx / sqrt(3)) ) # include phase factor for
                                                                     # hopping from different sublattice
            return np.einsum('i,ij', t, exp_nk)
        mat[0,0] = matrix_el(0,0, xi) - self.mu * np.ones(self.kx.shape)
        mat[0,1] = matrix_el(1,0, xi)
        mat[1,0] = matrix_el(0,1, xi)
        mat[1,1] = matrix_el(1,1, xi) - self.mu * np.ones(self.kx.shape)
        return mat

    def matrix(self):
        """
        This function updates the off-diagonal blocks of self.H according
        to the order parameters (self.DA, self.DB, ...) and returns an 
        array H of shape (8, 8, len(kx)).
        H[:,:,0] is the 8-by-8 Hamiltonian at point k=(kx[0], ky[0])
        """
        alpha_k_plus = self.d1p * self.__exp_kr1 \
                + self.d2p * self.__exp_kr2 \
                + self.d3p * self.__exp_kr3        # \alpha_{+}(k)
        alpha_nk_plus = self.d1p * self.__exp_nkr1 \
                + self.d2p * self.__exp_nkr2 \
                + self.d3p * self.__exp_nkr3    # \alpha_{+}(-k)
        alpha_k_minus = self.d1m * self.__exp_kr1 \
                + self.d2m * self.__exp_kr2 \
                + self.d3m * self.__exp_kr3       # \alpha_{-}(k)
        alpha_nk_minus = self.d1m * self.__exp_nkr1 \
                + self.d2m * self.__exp_nkr2 \
                + self.d3m * self.__exp_nkr3   # \alpha_{-}(-k)

        alpha_k_plus_ = self.d1p_ * self.__exp_kr1 \
                + self.d2p_ * self.__exp_kr2 \
                + self.d3p_ * self.__exp_kr3        # \alpha_{+}'(k)
        alpha_nk_plus_ = self.d1p_ * self.__exp_nkr1 \
                + self.d2p_ * self.__exp_nkr2 \
                + self.d3p_ * self.__exp_nkr3    # \alpha_{+}'(-k)
        alpha_k_minus_ = self.d1m_ * self.__exp_kr1 \
                + self.d2m_ * self.__exp_kr2 \
                + self.d3m_ * self.__exp_kr3       # \alpha_{-}'(k)
        alpha_nk_minus_ = self.d1m_ * self.__exp_nkr1 \
                + self.d2m_ * self.__exp_nkr2 \
                + self.d3m_ * self.__exp_nkr3   # \alpha_{-}'(-k)

        self.H[0,6] = self.dA
        self.H[0,7] = alpha_k_plus
        self.H[1,6] = alpha_nk_minus
        self.H[1,7] = self.dB

        self.H[2,4] = self.dA
        self.H[2,5] = alpha_k_minus
        self.H[3,4] = alpha_nk_plus
        self.H[3,5] = self.dB

        self.H[0,4] = self.dAp_
        self.H[0,5] = alpha_k_plus_
        self.H[1,4] = alpha_nk_plus_
        self.H[1,5] = self.dBp_

        self.H[2,6] = self.dAm_
        self.H[2,7] = alpha_k_minus_
        self.H[3,6] = alpha_nk_minus_
        self.H[3,7] = self.dBm_

        self.H[4:8, 0:4] = np.swapaxes(self.H[0:4, 4:8].conj(), 0, 1) # swap axes to insert h.c.
        return self.H

    def eval_meanfields(self):
        """
        This function diagonalizes the Hamiltonian and uses the eigenvectors
        to evaluate the expectation values to calculate all meanfields (Eq. 36-39 in notes)
        at zero temperature
        """
        def expectL(S, i, j):
            x = np.sum(S[i+4, 0:4].conj() * S[j,0:4])
            return x 
        def expectR(S, i, j):
            x = np.sum(S[i, 4:8] * S[j+4,4:8].conj())
            return x 
        
        # onsite inter-valley
        dAk = np.zeros(self.NN, dtype='complex128')     # \Delta_{A}
        dBk = np.zeros(self.NN, dtype='complex128')     # \Delta_{B}
        # onsite intra-valley
        dApk_ = np.zeros(self.NN, dtype='complex128')   # \Delta_{A,+}'
        dAmk_ = np.zeros(self.NN, dtype='complex128')   # \Delta_{A,-}'
        dBpk_ = np.zeros(self.NN, dtype='complex128')   # \Delta_{B,+}'
        dBmk_ = np.zeros(self.NN, dtype='complex128')   # \Delta_{B,-}'
        # nn inter-valley
        a_plus = np.zeros(self.NN, dtype='complex128')
        a_minus = np.zeros(self.NN, dtype='complex128')
        b_plus = np.zeros(self.NN, dtype='complex128')
        b_minus = np.zeros(self.NN, dtype='complex128')
        # nn intra-valley
        a_plus_ = np.zeros(self.NN, dtype='complex128')
        a_minus_ = np.zeros(self.NN, dtype='complex128')
        b_plus_ = np.zeros(self.NN, dtype='complex128')
        b_minus_ = np.zeros(self.NN, dtype='complex128')
        for i in np.arange(self.NN):
            ev, S = np.linalg.eigh(self.H[:,:,i])
            dAk[i] = expectR(S, 2, 0) - expectL(S, 2, 0)
            dBk[i] = expectR(S, 3, 1) - expectL(S, 3, 1)

            dApk_[i] = expectL(S, 0, 0)
            dBpk_[i] = expectL(S, 1, 1)
            dAmk_[i] = expectL(S, 2, 2)
            dBmk_[i] = expectL(S, 3, 3)

            a_plus[i] = expectR(S, 3, 0)
            b_plus[i] = expectL(S, 3, 0)
            a_minus[i] = expectR(S, 1, 2)
            b_minus[i] = expectL(S, 1, 2)

            a_plus_[i] = expectR(S, 1, 0)
            b_plus_[i] = expectL(S, 1, 0)
            a_minus_[i] = expectR(S, 3, 2)
            b_minus_[i] = expectL(S, 3, 2)

        self.dA = self.U/2 * np.mean(dAk)
        self.dB = self.U/2 * np.mean(dBk)

        self.dAp_ = -self.U_ * np.mean(dApk_)
        self.dBp_ = -self.U_ * np.mean(dBpk_)
        self.dAm_ = -self.U_ * np.mean(dAmk_)
        self.dBm_ = -self.U_ * np.mean(dBmk_)

        self.d1p = self.V/2 * np.mean(a_plus * self.__exp_kr1 - b_plus * self.__exp_nkr1)
        self.d2p = self.V/2 * np.mean(a_plus * self.__exp_kr2 - b_plus * self.__exp_nkr2)
        self.d3p = self.V/2 * np.mean(a_plus * self.__exp_kr3 - b_plus * self.__exp_nkr3)
        self.d1m = self.V/2 * np.mean(a_minus * self.__exp_kr1 - b_minus * self.__exp_nkr1)
        self.d2m = self.V/2 * np.mean(a_minus * self.__exp_kr2 - b_minus * self.__exp_nkr2)
        self.d3m = self.V/2 * np.mean(a_minus * self.__exp_kr3 - b_minus * self.__exp_nkr3)

        self.d1p_ = self.V_/2 * np.mean(a_plus_ * self.__exp_kr1 - b_plus_ * self.__exp_nkr1)
        self.d2p_ = self.V_/2 * np.mean(a_plus_ * self.__exp_kr2 - b_plus_ * self.__exp_nkr2)
        self.d3p_ = self.V_/2 * np.mean(a_plus_ * self.__exp_kr3 - b_plus_ * self.__exp_nkr3)
        self.d1m_ = self.V_/2 * np.mean(a_minus_ * self.__exp_kr1 - b_minus_ * self.__exp_nkr1)
        self.d2m_ = self.V_/2 * np.mean(a_minus_ * self.__exp_kr2 - b_minus_ * self.__exp_nkr2)
        self.d3m_ = self.V_/2 * np.mean(a_minus_ * self.__exp_kr3 - b_minus_ * self.__exp_nkr3)

    def bandstructure(self, res=100):
        """
        This function plots the tight-binding bandstructure of TBG with all meanfields set to 0
        """
        def buildPathBZ(res):
            # res - number of points per line in BZ
            k1=2*pi/sqrt(3) # K point
            k2=2*pi/3
            m1=2*pi/sqrt(3) # M point
            m2=0
            kx = np.block([np.linspace(k1, 0, res), \
                           np.linspace(0, m1, res), \
                           np.linspace(m1, k1, res)])
            ky = np.block([np.linspace(k2, 0, res), \
                           np.linspace(0, m2, res), \
                           np.linspace(m2, k2, res)])
            d1 = np.sqrt(k1**2 + k2**2)
            d2 = np.sqrt((m1)**2 + (m2)**2)
            d3 = np.sqrt((m1-k1)**2 + (m2-k2)**2)
            path = np.block([np.linspace(0, d1, res), \
                             np.linspace(d1, d1+d2, res), \
                             np.linspace(d1+d2, d1+d2+d3, res)])

            return kx, ky, path, d1, d1+d2
        kx, ky, path, gamma, m = buildPathBZ(res)
        def diagH0(xi):
            H0 = self.__buildh0(xi, kx, ky)
            en = np.zeros((len(kx),2))
            for i in np.arange(len(kx)):
                hk = H0[:,:,i]
                #hk += hk.T.conj()
                energies, vecs = np.linalg.eigh(hk)
                en[i,:] = energies
            return en * 1000
        plt.ion()
        plt.figure()
        en1 = diagH0(xi=1)
        en2 = diagH0(xi=-1)
        plt.plot(path, en1, 'k', label=r'$\xi=+$')
        plt.plot(path, en2, 'r-.', label=r'$\xi=-$')
        plt.ylabel('E in meV')
        plt.xlabel('BZ')
        plt.axvline(gamma, c='k',lw=0.5)
        plt.axvline(m, c='k',lw=0.5)
        plt.xlim(np.min(path),np.max(path))
        plt.ylim((-5,5))
        plt.legend()
    def plotBZ(self):
        """
        shows a scatter-plot of all point in the Brillouin zone
        """
        plt.figure()
        plt.ion()
        plt.scatter(self.kx/pi, self.ky/pi)
        plt.axis('equal')
        plt.xlabel('$k_x/\pi$')
        plt.ylabel('$k_y/\pi$')
