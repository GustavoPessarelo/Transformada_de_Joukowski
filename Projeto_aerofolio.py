import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def Juc(z, lam): #Transformada de Joukowski
    return z+(lam**2)/z
def circulo(C, R):
    t = np.linspace(0,2*np.pi, 200)
    return C+R*np.exp(1j*t)
def paraRadianos(graus):
    return graus*np.pi/180

plt.rcParams['figure.figsize'] = 8, 8

xis = list(range(0,11))
yps = [10] * 11

#Parametros do aerofolio
def linhasDeFluxo(alpha = 10, beta = 10, V_inf = 1, R = 1, proporcao = 1.2):
    alpha = paraRadianos(alpha) #Angulo de ataque
    beta = paraRadianos(beta) #Parametro de Joukowski - centro do circulo
    if proporcao<=1: 
        raise ValueError('R/lambda deve ser >1')
    lam = R/proporcao #lam é o parametro da transformada de Joukowski

    center_c = lam-R*np.exp(-1j*beta) #Centro do circulo
    x = np.arange(-3,3, 0.1)
    y = np.arange(-3,3, 0.1)
    x,y = np.meshgrid(x,y)
    z = x+1j*y
    z = ma.masked_where(np.absolute(z-center_c)<=R, z)
    Z = z-center_c
    Gamma = -4*np.pi*V_inf*R*np.sin(beta+alpha) #circulacao
    
    U = np.zeros(Z.shape, dtype=complex)
    with np.errstate(divide='ignore'):
        for m in range(Z.shape[0]):
            for n in range(Z.shape[1]): 
                 U[m,n] = Gamma*np.log((Z[m,n]*np.exp(-1j*alpha))/R)/(2*np.pi)
    c_flow = V_inf*Z*np.exp(-1j*alpha) + (V_inf*np.exp(1j*alpha)*R**2)/Z - 1j*U #Fluxo complexo

    J = Juc(z, lam) #Transformada de Joukowski do plano complexo menos o disco D
    Circulo = circulo(center_c, R)
    Aerofolio = Juc(Circulo, lam)# airfoil 
    return J, c_flow.imag, Aerofolio


J, stream_func, Aerofolio=linhasDeFluxo()
niveis = np.arange(-2.8, 3.8, 0.2).tolist()

fig = plt.figure()
ax = fig.add_subplot(111)
cp = ax.contour(J.real, J.imag, stream_func, levels=niveis, colors='blue', linewidths=1, linestyles='solid')

ax.plot(Aerofolio.real, Aerofolio.imag)
ax.set_aspect('equal')

plt.show()