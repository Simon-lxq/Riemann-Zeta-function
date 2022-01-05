import numpy as np
from mpmath import mp
from scipy import special
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation

config = {
    "xtick.direction":'in',
    "ytick.direction":'in',
    "font.family":'serif',
    "font.size": 16,
    "mathtext.fontset":'stix',
    # "font.serif": ['SimSun'],
    "scatter.edgecolors":'none',
}
rcParams.update(config)

# https://math.dartmouth.edu/archive/m56s13/public_html/Nguyen_proj.pdf

def zetaEMS(s, N=100, v=100):
    sum1 = mp.mpc(0)
    s = mp.mpc(s)
    for n in range(1, N):
        sum1 = sum1 + (n ** (-s))

    sum1 = sum1 + ((N ** (1 - s)) / mp.mpc(s - 1))
    sum1 = sum1 + ((N ** (-s)) / mp.mpc(2))

    sum2 = 0
    for k1 in range(1, v + 1):
        t1 = (bernoulli(2 * k1) / mp.mpc(fact(2 * k1)))
        prd = mp.mpc(1)
        for h in range(0, (2 * k1) - 1):
            prd = prd * (s + mp.mpc(h))
        t2 = prd
        t3 = N ** (1 - s - (2 * k1))
        sum2 = sum2 + (t1 * t2 * t3)
    return np.complex_(sum1 + sum2)

def bernoulli(val):
    return special.bernoulli(val)[val]

def fact(d):
    pd = 1
    for a in range(1, d + 1):
        pd = pd * a
    return pd

def nCr(n, r):
    return mp.mpf(fact(n)) / mp.mpf(fact(r) * fact(n - r))

m = 1201
c = 0.5+np.linspace(0,int(1e4/20),int(1e4)+1)*1j
# z = np.array([zetaEMS(i) for i in c])
z = np.loadtxt('zeta',dtype=np.complex_)

fig = plt.figure(figsize=(6,6))
plt.grid(ls='--')
plt.scatter(0,0,c='black',s=60)
plt.xlabel(r'$real$')
plt.ylabel(r'$imag$')

title_ani = plt.title(r'$s=%s+%.2fi$'%(c[0].real,c[0].imag))
line1_ani = plt.plot(z[:m].real,z[:m].imag,'blue')[0]
line2_ani = plt.plot(z[:m].real,-z[:m].imag,'red')[0]

def updata_line(n):
    title_ani.set_text(r'$s=%s\pm%.2fi$'%(c[n].real,c[n].imag))
    line1_ani.set_data(z[:n].real, z[:n].imag)
    line2_ani.set_data(z[:n].real,-z[:n].imag)
    return [title_ani,line1_ani,line2_ani]

ani = animation.FuncAnimation(fig=fig,func=updata_line,frames=np.arange(0,m),interval=10)
plt.axis('scaled')
plt.xlim(-2,4)
plt.ylim(-3,3)
plt.tight_layout()
ani.save('Zeta.gif')
plt.show()
