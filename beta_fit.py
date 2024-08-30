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
v_w_dist = np.zeros(num_simulations)
n_dist = np.zeros(num_simulations)
e_calc = np.zeros(num_simulations)
var_calc = np.zeros(num_simulations)
e_dist = np.zeros(num_simulations)
var_dist = np.zeros(num_simulations)
kolmogorov_smirnov = np.zeros(num_simulations)

for idx in tqdm(range(num_simulations)):
    n = round(10 ** np.random.uniform(np.log10(50), 3))
    v_f, w_f = np.random.uniform(0.5,5,size=2)
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
    v_w_dist[idx] = w_f
    n_dist[idx] = n
    e_calc[idx] = e_s
    var_calc[idx] = var_s
    e_dist[idx] = np.mean(s)
    var_dist[idx] = np.var(s)
    kolmogorov_smirnov[idx] = ks_p


df = pd.DataFrame(np.array([v_f_dist,v_w_dist,n_dist,e_calc,var_calc,e_dist,var_dist,kolmogorov_smirnov]).T, columns = ['v_f','w_f','n_dist','e_calc','var_calc','e_dist','var_dist','ks'])
df['v_s_calc'] = df['e_calc']*(df['e_calc']*(1-df['e_calc'])/df['var_calc']-1)
df['w_s_calc'] = (1-df['e_calc'])/df['e_calc'] * df['v_s_calc']
df['ks5'] = 1*(df['ks']>=0.05)
df['ks1'] = 1*(df['ks']>=0.01)
df['ks01'] = 1*(df['ks']>=0.001)

df[['ks5','ks1','ks01']].mean(axis=0)



y = df['ks']
base_model = BetaModel(y,np.ones(y.shape)).fit(method='newton')
X = df[['v_f','w_f']].copy()
X['v_f_2'] = X['v_f']**2
X['w_f_2'] = X['w_f']**2
X = sm.add_constant(X)
no_n_model = BetaModel(y,X).fit(method='newton')
X = df['n_dist'].copy()
X = sm.add_constant(X)
X['n_dist_2'] = X['n_dist']**2
n_model = BetaModel(y,X).fit(method='newton')

base_model.bic
no_n_model.bic
n_model.bic

X = df['n_dist'].copy()
X = sm.add_constant(X)
for i in range(2,8):
    X[f'n_dist_{i}'] = X['n_dist']**i
n_model = BetaModel(y,X).fit(method='newton')
coefs = n_model.params.values[:-1]
n = np.linspace(50,1000,10001)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(n,[expit(np.dot(coefs,[ni**i for i in range(len(coefs))])) for ni in n],color='0.5')
ax.set_xlabel(r'$n$',fontsize=13)
ax.set_ylabel(r'$\bar{p}_{ks}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

p_rand = []
for i in tqdm(range(len(df))):
    p_rand.append(kstest(beta.rvs(df.loc[i,'v_s_calc'],df.loc[i,'w_s_calc'],size=10000), 'beta', args=(df.loc[i,'v_s_calc'],df.loc[i,'w_s_calc']))[1])


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(sorted(df['ks']),np.arange(len(df))/len(df),color='0.5')
ax.plot(sorted(p_rand),np.arange(len(df))/len(df),color='0.8')
ax.set_xlabel(r"$p_{ks}$",fontsize=13)
ax.set_ylabel("Cumulative Distribution")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

df[['e_calc','e_dist']].corr()
df[['var_calc','var_dist']].corr()