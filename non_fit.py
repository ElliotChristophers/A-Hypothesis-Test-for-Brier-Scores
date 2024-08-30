import numpy as np
from scipy.stats import kstest, beta
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure()
idx=0
while True:
    n = round(10 ** np.random.uniform(np.log10(50), 3))
    v_f, w_f = np.random.uniform(0.5,5,size=2)
    f = np.random.beta(v_f,w_f,size=n)

    rand_its = 10000
    s = np.zeros(rand_its)
    for i in range(rand_its):
        x = np.where(np.random.random(n) < f, 1, 0)
        s[i] = np.mean((f - x) ** 2)
    e_s = np.mean(f - f**2)

    var_s = np.sum(f*(1-f)**4+(1-f)*f**4)
    term = (f - f**2)
    var_s += np.sum(term[:, None] * term[None, :]) - np.sum(term**2)
    var_s /= len(f)**2
    var_s -= e_s**2
    
    v_s = e_s*(e_s*(1-e_s)/var_s-1)
    w_s = (1-e_s)/e_s * v_s

    ks_p = kstest(s, 'beta', args=(v_s, w_s))[1]
    if ks_p < 0.001:
        ax = fig.add_subplot(331+idx)
        hx = np.linspace(min(s),max(s),1001)
        ax.hist(s,range=(min(s),max(s)),bins=100,density=True,color='0.8')
        ax.plot(hx,beta.pdf(hx,v_s,w_s),color='black')
        ax.set_xlabel(r"$\hat{S}_j$",fontsize=13)
        ax.set_ylabel('Density')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        idx+=1
        print(idx)
        if idx == 9:
            break
plt.tight_layout()
plt.show()