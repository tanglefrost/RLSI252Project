import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
import Algorithms

def calc_beta(m_list, delta, K):
    m = np.sum(m_list)
    m_ratio = np.sqrt(np.array(m_list)) / m
    return m_ratio.sum() * np.sqrt(2 * np.log(K / delta))

def conf_bound(n, K, delta):
    return np.sqrt(np.log(K / delta) / (2 * np.maximum(n, 1e-8)))


def run_bandit(instance_means, m_list, T, algorithm, alpha=0.2):
    '''
    instance_means: array, arm true means
    m_list: list, offline samples per arm
    T: horizon
    algorithm: 'UCB', 'LCB', 'OTO'
    '''
    K = len(instance_means)
    delta = 1 / (T ** 2)
    # Offline statistics
    mu_hat_0, n_init = np.zeros(K), np.zeros(K)
    s = np.zeros(K)
    for i in range(K):
        mi = m_list[i]
        if mi > 0:
            offline_samples = np.random.binomial(1, instance_means[i], mi)
            mu_hat_0[i] = np.mean(offline_samples)
            n_init[i] = mi
            s[i] = offline_samples.sum()
        else:
            mu_hat_0[i] = 0.0
            n_init[i] = 0
    m = np.sum(m_list)
    pi = np.array(m_list) / m if m > 0 else np.ones(K)/K
    mu0 = np.dot(pi, instance_means)

    n = n_init.copy()
    # regret记录
    history_regret = []
    oto = Algorithms.OTOBandit(instance_means, m_list, T, alpha)

    for t in range(T):
        if algorithm == 'UCB':
            mu_hat = np.divide(s, n, out=np.zeros_like(n, dtype=float), where=n>0)
            cb = conf_bound(n, K, delta)
            ucb = mu_hat + cb
            ucb[n == 0] = 1
            i_star = np.argmax(ucb)
        elif algorithm == 'LCB':
            mu_hat = np.divide(s, n, out=np.zeros_like(n, dtype=float), where=n>0)
            cb = conf_bound(n, K, delta)
            lcb = mu_hat - cb
            lcb[n == 0] = -1
            i_star = np.argmax(lcb)
        elif algorithm == 'OTO':
            i_star, reward = oto.step(t)
        elif algorithm == 'SoftOTO':
            i_star, reward = oto.step(t)
        if algorithm in ['UCB', 'LCB']:
            reward = np.random.binomial(1, instance_means[i_star])
            n[i_star] += 1
            s[i_star] += reward
        regret = np.max(instance_means) - instance_means[i_star]
        history_regret.append(regret)
    return np.cumsum(history_regret)


