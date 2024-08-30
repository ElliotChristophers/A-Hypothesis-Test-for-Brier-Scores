import numpy as np
from tqdm import tqdm
from scipy.stats import beta
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


num_simulations = 1000
rand_its = 10000

complete_decisions = []
for i in tqdm(range(num_simulations)):
    v_f, w_f = np.random.uniform(0.5,5,size=2)
    
    all_decisions = []
    for i in range(rand_its):
        n = round(10 ** np.random.uniform(np.log10(50), 3))
        f = np.random.beta(v_f,w_f,size=n)
        decisions = []
        for delta in [0,0.125,0.25]:
            x = np.where(np.random.random(n) < (1-delta)*f+delta*(v_f/(v_f+w_f)), 1, 0)
            s = np.mean((f - x) ** 2)
            e_s = np.mean(f - f**2)

            var_s = np.sum(f*(1-f)**4+(1-f)*f**4)
            term = (f - f**2)
            var_s += np.sum(term[:, None] * term[None, :]) - np.sum(term**2)
            var_s /= len(f)**2
            var_s -= e_s**2

            v_s = e_s*(e_s*(1-e_s)/var_s-1)
            w_s = (1-e_s)/e_s * v_s
            decision = []
            for sl in [0.01,0.05,0.1]:
                decision.append(1*(s <= beta.ppf(1-sl,v_s,w_s)))
            decisions.append(decision)
        all_decisions.append(decisions)
    complete_decisions.append(np.mean(np.array(all_decisions),axis=0))
df = pd.DataFrame(np.mean(np.array(complete_decisions),axis=0),columns=['0.01','0.05','0.1'],index=['0','0.125','0.25'])

df_r = pd.DataFrame([(df.loc[d,a],float(d),float(a)) for d in df.index for a in df.columns],columns=['p','d','a'])
X = df_r[['d','a']]
X = sm.add_constant(X)
sm.OLS(df_r['p'],X).fit().rsquared

l = []
complete_decisions = np.array(complete_decisions)
da = [[float(d),float(a)] for d in df.index for a in df.columns]
for i in range(len(complete_decisions)):
    flat = complete_decisions[i].flatten()
    for j in range(len(flat)):
        l.append([flat[j]]+da[j])
df_r = pd.DataFrame(l,columns=['p','d','a'])
X = df_r[['d','a']]
X = sm.add_constant(X)
sm.OLS(df_r['p'],X).fit().rsquared