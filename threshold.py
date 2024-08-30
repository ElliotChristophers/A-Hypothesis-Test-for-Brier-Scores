import numpy as np
from tqdm import tqdm
from scipy.stats import kstest, beta
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


num_simulations = 10000
rand_its = 10000

v_f_dist = np.zeros(num_simulations)
w_f_dist = np.zeros(num_simulations)
n_dist = np.zeros(num_simulations)
e_calc = np.zeros(num_simulations)
var_calc = np.zeros(num_simulations)
e_dist = np.zeros(num_simulations)
var_dist = np.zeros(num_simulations)
kolmogorov_smirnov = np.zeros(num_simulations)
fraction = np.zeros(num_simulations)

for idx in tqdm(range(num_simulations)):
    n = round(10**np.random.uniform(np.log10(5), 3))
    v_f, w_f = 10**np.random.uniform(-1,3,size=2)
    f = np.random.beta(v_f,w_f,size=n)

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

    v_f_dist[idx] = v_f
    w_f_dist[idx] = w_f
    n_dist[idx] = n
    e_calc[idx] = e_s
    var_calc[idx] = var_s
    e_dist[idx] = np.mean(s)
    var_dist[idx] = np.var(s)
    kolmogorov_smirnov[idx] = ks_p
    fraction[idx] = e_s/np.sqrt(1/n*(np.mean(f*(1-f)**4+(1-f)*f**4)-e_s**2))


df = pd.DataFrame(np.array([v_f_dist,w_f_dist,n_dist,e_calc,var_calc,e_dist,var_dist,kolmogorov_smirnov,fraction]).T, columns = ['v_f_dist','w_f_dist','n_dist','e_calc','var_calc','e_dist','var_dist','ks','fraction'])
df['ks5'] = 1*(df['ks']>=0.05)
df['ks1'] = 1*(df['ks']>=0.01)
np.corrcoef(df['fraction'],df['e_calc']/np.sqrt(df['var_calc']))

fig = plt.figure()
cols = ['ks','ks5','ks1']
hline = [0.475,0.903,0.968]
for i in range(3):
    l = []
    for z in np.linspace(0,50,1000):
        l.append([z,df[df['e_calc']/np.sqrt(df['var_calc'])<z][cols[i]].mean(),df[df['e_calc']/np.sqrt(df['var_calc'])>=z][cols[i]].mean()])
    l = np.array(l).T
    ax = fig.add_subplot(311+i)
    if i == 0:
        ax.plot(l[0],l[2],color='0.5',label=r'$\frac{E[S]}{\sqrt{Var[S]}}\geq z$')
        ax.plot(l[0],l[1],color='0.8',label=r'$\frac{E[S]}{\sqrt{Var[S]}}<z$')
        ax.legend(bbox_to_anchor=[0,1.3],loc='upper left')
    else:
        ax.plot(l[0],l[2],color='0.5')
        ax.plot(l[0],l[1],color='0.8')
    ax.hlines(hline[i],0,50,color='0',linewidth=0.5)
    ax.set_ylim(-1e-2,1+1e-2)
    ax.set_xlim(-1e-2,50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i == 0:
        ax.set_ylabel(r"$\bar{p}_{ks}$")
    elif i == 1:
        ax.set_ylabel(r"$\text{Pr}(p_{ks}\geq0.05)$")
    else:
        ax.set_xlabel(r"$z$",fontsize=13)
        ax.set_ylabel(r"$\text{Pr}(p_{ks}\geq0.01)$")
plt.show()

df_lower = df[df['e_calc']/np.sqrt(df['var_calc'])<10]
(df_lower['n_dist'] >= 10**2*df_lower['var_calc']/df_lower['e_calc']**2).mean()