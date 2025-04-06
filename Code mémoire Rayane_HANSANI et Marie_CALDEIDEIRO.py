#!/usr/bin/env python
# coding: utf-8

# # Téléchargement des données et des librairies


#On charge les données de Yahoo Finance
#get_ipython().system('pip install yfinanc')


#importation des librairies
import pandas as pd # on renomme les librairies
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf 
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import gaussian_kde, norm, iqr, skew, kurtosis, jarque_bera, kstest, anderson
from statsmodels.stats.diagnostic import lilliefors
import scipy.signal as ss
import pylab
import datetime as dt


# Importation des données du CAC 40 depuis le 1er Mars 1990 (la première valeur répertoriée par Yahoo finance)
cac40 = yf.download("^FCHI", start="1990-03-01", end="2024-04-30") 
cac40


#### Calcul des rendements logarithmiques périodiques


# extraction des prix à la clôture 
pt_d_all = cac40["Adj Close"]
pt_d_all = pt_d_all.rename('Pt.d')

pt_d_all.index = pd.to_datetime(pt_d_all.index)  
pt_d_all.head()


#calcul du log des niveaux hebdomadaires (w), mensuels (m) et annuel (y)
pt_w_all = pt_d_all.resample('W').last()
pt_m_all = pt_d_all.resample('M').last()
pt_y_all = pt_d_all.resample('Y').last()

pt_w_all = pt_w_all.rename('pt.w.all')
pt_m_all = pt_m_all.rename('pt.m.all')
pt_y_all = pt_y_all.rename('pt.y.all')


#calcul du log des rendements 
rt_d_all_temp = pt_d_all.diff()
rt_m_all_temp = pt_m_all.diff()
rt_y_all_temp = pt_y_all.diff()
rt_d_all_temp
rt_m_all_temp
rt_y_all_temp


#log des rendements mensuels
rt_m_all_temp = pt_m_all.diff()
rt_m_all_temp


#log des rendements annuels
rt_y_all_temp = pt_y_all.diff()
rt_y_all_temp


rt_d_all = pt_d_all.diff().dropna() # on ne peut pas calculer la première observation
rt_w_all = pt_w_all.diff().dropna()  
rt_m_all = pt_m_all.diff().dropna()     
rt_y_all = pt_y_all.diff().dropna()
rt_d_all = rt_d_all.rename('rt_d_all')
rt_w_all = rt_w_all.rename('rt_w_all')
rt_m_all = rt_m_all.rename('rt_m_all')
rt_y_all = rt_y_all.rename('rt_y_all')
rt_d_all.head()


#### Lien avec la loi normale

# Créer des sous-graphiques
fig, axs = plt.subplots(1 , 2, figsize=(18, 9))

# Histogramme des rendements quotidiens et courbe de la loi normale
axs[0].hist(rt_d_all_temp, bins=50, density=True, color="orange")
norm_y = stats.norm.pdf(np.linspace(rt_d_all_temp.min(), rt_d_all_temp.max()), loc=np.mean(rt_d_all_temp), scale=np.std(rt_d_all_temp))
axs[0].plot(np.linspace(rt_d_all_temp.min(), rt_d_all_temp.max()), norm_y, color="blue", linewidth=1)
axs[0].set_xlabel("log des rendements quotidiens")
axs[0].set_title("Figure 1: Histogramme des rendements quotidiens et distribution de la loi normale")

# Histogramme des rendements quotidiens et courbe de la loi normale
axs[1].hist(rt_w_all, bins=50, density=True, color="pink")
norm_y = stats.norm.pdf(np.linspace(rt_w_all.min(), rt_w_all.max()), loc=np.mean(rt_w_all), scale=np.std(rt_w_all))
axs[1].plot(np.linspace(rt_w_all.min(), rt_w_all.max()), norm_y, color="blue", linewidth=1)
axs[1].set_xlabel("log des rendements quotidiens")
axs[1].set_title("Figure 2: Histogramme des rendements quotidiens et distribution de la loi normale")

# Ajustement de l'affichage
plt.tight_layout()
plt.show()

#QQ-plot de la loi normale et des rendements quotidiens, hebdomadaires, mensuels et annuels

# Créer des sous-graphiques
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Courbe de probabilité du log des rendements quotidiens 
stats.probplot(rt_d_all, dist="norm", plot=axs[0,0], fit=True)
axs[0,0].set_title("Figure 3:Courbe de probabilité du log des rendements quotidiens ")
axs[0, 0].grid(True)

# Courbe de probabilité du log des rendements hebdomadaires
stats.probplot(rt_w_all, dist="norm", plot=axs[0,1], fit=True)
axs[0,1].set_title("Figure 4: Courbe de probabilité du log des rendements hebdomadaires ")
axs[0, 1].grid(True)

# Courbe de probabilité du log des rendements mensuels
stats.probplot(rt_m_all, dist="norm", plot=axs[1,0], fit=True)
axs[1,0].set_title("Figure 5: Courbe de probabilité du log des rendements mensuels ")
axs[1, 0].grid(True)

# Courbe de probabilité du log des rendements annuels
stats.probplot(rt_y_all, dist="norm", plot=axs[1,1], fit=True)
axs[1,1].set_title("Figure 6: Courbe de probabilité du log des rendements annuels ")
axs[1, 1].grid(True)

# Ajustement de l'affichage
plt.tight_layout()
plt.show()

rt_d_skew = skew(rt_d_all, nan_policy='omit')
rt_d_kurt = kurtosis(rt_d_all, nan_policy='omit')

print("La skewness est:", rt_d_skew)
print("Le kurtosis est:", rt_d_kurt)

rt_d_kurt = kurtosis(rt_d_all, nan_policy='omit')
rt_w_kurt = kurtosis(rt_w_all, nan_policy='omit')
rt_m_kurt = kurtosis(rt_m_all, nan_policy='omit')
rt_y_kurt = kurtosis(rt_y_all, nan_policy='omit')

print("Journalier: ", round(rt_d_kurt,3))
print("Hebdomadaire: ", round(rt_w_kurt,3))
print("Mensuel: ", round(rt_m_kurt,3))
print("Annuel: ", round(rt_y_kurt,3))


#### Test de Jarque Bera


JB_rt_d = jarque_bera(rt_d_all)

print("JB Stat: ", round(JB_rt_d[0],3))
print("JB p-value: ", JB_rt_d[1])


print("JB p-value", "journalier", "rendements:", jarque_bera(rt_d_all)[1])
print("JB p-value", "hebdomadaire", "rendements:", jarque_bera(rt_w_all)[1])
print("JB p-value", "mensuel", "rendements:", jarque_bera(rt_m_all)[1])
print("JB p-value", "annuel", "rendements:", jarque_bera(rt_y_all)[1])


T = len(rt_y_all)
((T/6)*(skew(rt_y_all))**2)+ (T/24*(kurtosis(rt_y_all)-3)**2)


a = jarque_bera(rt_y_all)
x = rt_y_all
n = len(x)         ## Nombre d'observationz
m1 = sum(x)/n         ## Moyenne
m2 = sum((x-m1)**2)/n  
m3 = sum((x-m1)**3)/n 
m4 = sum((x-m1)**4)/n 
b1 = (m3/m2**(3/2))**2  
b2 = (m4/m2**2)    
STATISTIC = n*b1/6+n*(b2-3)**2/24

print("Manuellemenbt:",STATISTIC)
print("en utilisant la fonction:",a[0])


p_value = 1 - stats.chi2.cdf(STATISTIC, df=2)
print("La p-value associée est:",p_value)


#### Preparation calcul des VaRs

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

# On choisit le même intervalle de temps que précédemment
endDate = "2020-04-29"
startDate = "1990-03-01"

# Créer une liste de tickers
tickers = ['^FCHI','^GSPC','^N225']

# On charge les prix ajustés pour les tickers
adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start = startDate, end = endDate)
    adj_close_df[ticker] = data['Adj Close']

print(adj_close_df)


# calcul des log-returns journaliers
log_returns = np.log(adj_close_df/adj_close_df.shift(1))
log_returns  = log_returns.dropna()

print(log_returns)


# Horizon temporel
days=10
print(days)


# on crée un portefeuille equipondéré
portfolio_value = 1000000
weights = np.array([1/len(tickers)]*len(tickers))
print(weights)


#### VaR historique

# calcul des rendements historiques du porte-feuille pondéré
historical_returns = (log_returns * weights).sum(axis =1) 
print(historical_returns)

#on trie par ordre croissant
sorted_historical_returns = historical_returns.sort_values()
print("Sorted Historical Returns:")
print(sorted_historical_returns)

#on calcule la VaR QUOTIDIENNE avec un niveau de confiance de 95%
confidence_interval90 = 0.90
var_1day90 = sorted_historical_returns.quantile(1-confidence_interval90)
confidence_interval95 = 0.95
var_1day95 = sorted_historical_returns.quantile(1-confidence_interval95)
confidence_interval99 = 0.99
var_1day99 = sorted_historical_returns.quantile(1-confidence_interval99)

print("La VaR historique sur un jour à un niveau de confiance de 90% est de:", round(-var_1day90*portfolio_value),"dollars")
print("La VaR historique sur un jour à un niveau de confiance de 95% est de:", round(-var_1day95*portfolio_value),"dollars")
print("La VaR historique sur un jour à un niveau de confiance de 99% est de:", round(-var_1day99*portfolio_value),"dollars")

# on ajuste à l'horizon temporel (le choix de la valeur de 'days')
VaR90 = round(var_1day90*np.sqrt(days)*portfolio_value)
print("La Var historique sur 10 jours à un niveau de confiance de {}% est de: {:.2f} dollars".format(int(confidence_interval90 * 100), -VaR90))
VaR95 = round(var_1day95*np.sqrt(days)*portfolio_value)
print("La Var historique sur 10 jours à un niveau de confiance de {}% est de: {:.2f} dollars".format(int(confidence_interval95 * 100), -VaR95))
VaR99 = round(var_1day99*np.sqrt(days)*portfolio_value)
print("La Var historique sur 10 jours à un niveau de confiance de {}% est de: {:.2f} dollars".format(int(confidence_interval99 * 100), -VaR99))

#On trace l'histogramme
import matplotlib.pyplot as plt

return_window = days
range_returns = sorted_historical_returns
range_returns = sorted_historical_returns.dropna()

range_returns_dollar = range_returns * portfolio_value

plt.hist(range_returns_dollar.dropna(), bins=50, density=True, label='Distribution des pertes du portefeuille')
plt.xlabel(f'Pertes du portefeuille en {return_window} jours (Valeur en dollars)')
plt.ylabel('Fréquence')
plt.title(f'Figure 7: Distribution des pertes du portefeuille en {return_window} jours')

plt.axvline(-VaR90, color='r', linestyle='dashed', linewidth=2, label=f'VaR à un niveau de confiance de 90%')
plt.axvline(-VaR95, color='g', linestyle='dashed', linewidth=2, label=f'VaR à un niveau de confiance de 95%')
plt.axvline(-VaR99, color='y', linestyle='dashed', linewidth=2, label=f'VaR à un niveau de confiance de 99%')
plt.ylim(top=plt.ylim()[1] * 1.3)
plt.legend()
plt.show()


#### VaR paramétrique

# moyenne des 'days' jours précédents pour chaque jour
historical_x_day_returns = historical_returns.rolling(window=days).sum()

## covariance
cov_matrix = log_returns.cov()
print(cov_matrix)

## Écart-type des rendements du portefeuille
portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
print(portfolio_std_dev)


historical_x_day_returns = historical_returns.rolling(window=days).sum()
historical_x_day_returns_dollar = historical_x_day_returns * portfolio_value

# Calcul de la VaR paramétrique pour différents niveaux de confiance
confidence_levels = [0.90, 0.95, 0.99]

VaRs_parametric = {}
for cl in confidence_levels:
    VaR = norm.ppf(1 - cl) * portfolio_std_dev * np.sqrt(days) * portfolio_value
    VaRs_parametric[cl] = -VaR  # On prend la valeur absolue car VaR est toujours positive
    print(f"La VaR paramétrique sur {days} jours à un niveau de confiance de {int(cl * 100)}% est de: {round(VaRs_parametric[cl])} dollars")

# Tracé de l'histogramme des rendements en dollars
plt.figure(figsize=(10, 6))
plt.hist(historical_x_day_returns_dollar.dropna(), bins=50, density=True, alpha=0.5, label=f'Pertes en {days}-jours')

# Couleurs pour les lignes VaR
colors = ['r', 'g', 'y']

# Ajout des lignes VaR à chaque niveau de confiance
for cl, color in zip(confidence_levels, colors):
    plt.axvline(x=VaRs_parametric[cl], color=color, linestyle='--', linewidth=2, label=f'VaR à un niveau de confiance de {int(cl * 100)}%')

# Étiquettes et titres
plt.xlabel(f'Pertes du Portefeuille en {days}-jours ($)')
plt.ylabel('Fréquence')
plt.title(f'Figure 8: Distribution des pertes du Portefeuille et estimation de la VaR paramétrique VaR sur {days}-jours')
plt.legend()
plt.grid(True)
plt.show()



#### VaR de Monte-Carlo

# On créer deux focntions pour calculer le rendement moyen et l'écart-type des rendements
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)

def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

# matrice de covariance
cov_matrix = log_returns.cov()
print(cov_matrix)

# On calcule
portfolio_expected_return = expected_return(weights, log_returns)
portfolio_std_dev = standard_deviation (weights, cov_matrix)
print("L'espérance du rendement quotidien est de",portfolio_expected_return)
print(portfolio_std_dev)


# def fonction pour la simulation de Monte Carlo 
# on choisit la loi normale pour modéliser les actifs du portefeuille
def random_z_score():
    return np.random.normal(0, 1)

# créer une fonction pour calculer scenarioGainLoss
def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
    return portfolio_value * portfolio_expected_return * days + portfolio_value * portfolio_std_dev * z_score * np.sqrt(days)

# On éxécute la simulation 10000 fois
simulations = 10000
scenarioReturn = []

for i in range(simulations):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days))

# On calcule la VaR
confidence_interval90 = 0.90
VaR90 = np.percentile(scenarioReturn, 100 * (1 - confidence_interval90))
VaR90 = round(VaR90)
print("La VaR de Monte-Carlo à cl 90% est:",-VaR90,"dollars")

confidence_interval95 = 0.95
VaR95 = np.percentile(scenarioReturn, 100 * (1 - confidence_interval95))
VaR95 = round(VaR95)
print("La VaR de Monte-Carlo à cl 95% est:",-VaR95,"dollars")

confidence_interval99 = 0.99
VaR99 = np.percentile(scenarioReturn, 100 * (1 - confidence_interval99))
VaR99 = round(VaR99)
print("La VaR de Monte-Carlo à cl 99% est:",-VaR99,"dollars")

#On trace
plt.hist(scenarioReturn, bins=50, density=True, label='Distribution des pertes du portefeuille')
plt.xlabel('Pertes du Portefeuille en dollars')
plt.ylabel('Fréquence')
plt.title(f'Figure 9: Distribution des Pertes du portefeuille en 10 jours et VaRs')
plt.axvline(-VaR90, color='r', linestyle='dashed', linewidth=2, label=f'VaR à un niveau de confiance de 90%')
plt.axvline(-VaR95, color='g', linestyle='dashed', linewidth=2, label=f'VaR à un niveau de confiance de 95%')
plt.axvline(-VaR99, color='y', linestyle='dashed', linewidth=2, label=f'VaR à un niveau de confiance de 99%')
plt.ylim(top=plt.ylim()[1] * 1.3)
plt.legend()
plt.show()


#### Calcul des lettres grecques
from scipy.stats import norm
import numpy as np

def calcul_de_d1(S, K, r, T, sigma):
    return np.log(S/K) + (r + sigma**2/2)*T/(sigma*np.sqrt(T))
                                              
def calcul_de_d2(sigma, T, d1):
    return d1-sigma*np.sqrt(T)

N = norm.cdf
n = norm.pdf

def delta_call(d1):
    return N(d1)

def delta_put(d1):
    return N(d1)-1

def gamma_call(S,sigma,T,d1):
    return n(d1)/(S*sigma*np.sqrt(T))

def gamma_put(S,sigma,T,d1):
    return n(d1)/(S*sigma*np.sqrt(T))
    
def theta_call(S, K, T, r, sigma, d1, d2):
    return ((-S*n(d1)*sigma/(2*np.sqrt(T)))-r*K*np.exp(-r*T)*N(d2))/365  #pour avoir theta en jours

def theta_put(S, K, T, r, sigma, d1, d2):
    return ((-S*n(d1)*sigma/(2*np.sqrt(T)))-r*K*np.exp(-r*T)*N(-d2))/365

def vega_call(S, T, d1):
    return S*np.sqrt(T)*n(d1)

def vega_put(S, T, d1):
    return S*np.sqrt(T)*n(d1)

def rho_call(K, T, r, d2):
    return K*T*np.exp(-r*T)*N(d2)/100

def rho_put(K, T, r, d2):
    return -K*T*np.exp(-r*T)*N(-d2)/100



#exemple d'un call sur le S&P500 de strike=5400 et qui expire dans une semaine
S=5432
K=5400
T=7/365         #expiration dans une semaine
r=4.5/100       #le taux d'intérêt de l'indice du S&P500 est de 4.5% le 14 Juin 2024
sigma=0.0814
d1 = calcul_de_d1(S, K, r, T, sigma)
d2 = calcul_de_d2(sigma, T, d1)

delta=delta_call(d1)
gamma=gamma_call(S,sigma,T,d1)
theta=theta_call(S, K, T, r, sigma, d1, d2)
vega=vega_call(S, T, d1)
rho=rho_call(K, T, r, d2)

print("Le delta du call est", delta)
print("Le gamma du call est", gamma)
print("Le theta du call est", theta)
print("Le vega du call est", vega)
print("Le rho du call est", rho)

delta=delta_put(d1)
gamma=gamma_put(S,sigma,T,d1)
theta=theta_put(S, K, T, r, sigma, d1, d2)
vega=vega_put(S, T, d1)
rho=rho_put(K, T, r, d2)

print("Le delta du put est", delta)
print("Le gamma du put est", gamma)
print("Le theta du put est", theta)
print("Le vega du put est", vega)
print("Le rho du put est", rho)





