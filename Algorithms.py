import numpy as np
import matplotlib.pyplot as plt


def calc_beta(m_list, delta, K):
    m = np.sum(m_list)
    m_ratio = np.sqrt(np.array(m_list) / m + 1e-12)  # Avoid zero
    return m_ratio.sum() * np.sqrt(2 * np.log(K / delta))

def conf_bound(n, K, delta):
    return np.sqrt(np.log(K / delta) / (2 * np.maximum(n, 1e-8)))

def UCB_step(n, s, t, K):
    delta = 1/(t**2)
    mu = np.divide(s, n, out=np.zeros_like(n, dtype=float), where=n>0)
    cb = np.sqrt(np.log(K/delta) / (2 * np.maximum(n, 1e-8)))
    ucb = mu + cb
    ucb[n==0] = 1
    return np.argmax(ucb)

def LCB_step(n, s, t, K):
    delta = 1/(t**2)
    mu = np.divide(s, n, out=np.zeros_like(n, dtype=float), where=n>0)
    cb = np.sqrt(np.log(K/delta) / (2 * np.maximum(n, 1e-8)))
    lcb = mu - cb
    lcb[n==0] = -1
    return np.argmax(lcb)

class OTOBandit:
    def __init__(self, means, m_list, T=2, alpha=0.2):
        self.K = len(means)
        self.means = np.array(means)
        self.T = T
        self.alpha = alpha
        self.delta = 1 / (T ** 2)
        # Offline statistics
        self.n = np.zeros(self.K)
        self.s = np.zeros(self.K)
        self.mu_hat_0 = np.zeros(self.K)
        for i in range(self.K):
            mi = m_list[i]
            if mi > 0:
                offline_samples = np.random.binomial(1, self.means[i], mi)
                self.mu_hat_0[i] = np.mean(offline_samples)
                self.n[i] = mi
                self.s[i] = offline_samples.sum()
        self.TU = np.zeros(self.K)
        self.TL = np.zeros(self.K)
        self.beta = calc_beta(m_list, self.delta, self.K)
        lcb0 = self.mu_hat_0 - conf_bound(np.maximum(self.n, 1), self.K, self.delta)
        lcb0[self.n == 0] = -np.inf
        L0 = np.argmax(lcb0)
        self.gamma = lcb0[L0] - self.alpha * self.beta

    def step(self, t, flag=True):
        if flag == False:
            self.delta = 1 / ((t+1) ** 2)
        mu_hat = np.zeros(self.K)
        for i in range(self.K):
            if self.n[i] > 0:
                mu_hat[i] = self.s[i] / self.n[i]
            else:
                mu_hat[i] = 0.0
        cb = conf_bound(self.n, self.K, self.delta)
        ucb = mu_hat + cb
        lcb = mu_hat - cb
        ucb[self.n == 0] = 1
        lcb[self.n == 0] = -1
        mu_max_ucb = np.max(ucb)
        mu_max_lcb = np.max(lcb)
        budget = np.sum(self.TU * (lcb - self.gamma)) + (mu_max_ucb - self.gamma) + ((self.TL.sum() + self.T - (t+1)) * self.alpha * self.beta)
        if budget > 0:
            i_star = np.argmax(ucb)
            self.TU[i_star] += 1
        else:
            i_star = np.argmax(lcb)
            self.TL[i_star] += 1
        reward = np.random.binomial(1, self.means[i_star])
        self.n[i_star] += 1
        self.s[i_star] += reward
        return i_star, reward