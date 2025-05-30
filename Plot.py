import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
import Run_Bandit
import Algorithms


def plot_CertainH():
    np.random.seed(42)
    K, m = 20, 2000
    m_list = [200] * 10 + [0] * 10
    # instance 1
    means1 = np.array([0.5]*10 + [0.25]*10)
    # instance 2
    means2 = np.array([0.5]*10 + [0.25]*9 + [0.75])
    for instance_idx, (means, label) in enumerate([(means1, 'Instance 1'), (means2, 'Instance 2')]):
        for horizon, color in zip([200, 2000], ['tab:blue', 'tab:orange']):
            regrets = {}
            for alg in ['LCB', 'UCB', 'OTO', 'SoftOTO']:
                vals = []
                for rep in range(200):
                    vals.append(run_bandit(means, m_list, horizon, alg, alpha=0.2))
                regrets[alg] = np.mean(vals, axis=0)
                std = np.std(vals, axis=0)
                plt.plot(regrets[alg], label=alg)
                plt.fill_between(np.arange(horizon), regrets[alg]-2*std, regrets[alg]+2*std, alpha=0.2)
            plt.title(f"{label}, T={horizon}")
            plt.xlabel('iteration')
            plt.ylabel('Regret')
            plt.xlim(0, horizon)
            plt.legend()
            plt.tight_layout()
            plt.show()

def plot_UnknownH():
    np.random.seed(42)
    K, m = 20, 2000
    m_list = [200] * 10 + [0] * 10
    # instance 1
    means1 = np.array([0.5]*10 + [0.25]*10)
    # instance 2
    means2 = np.array([0.5]*10 + [0.25]*9 + [0.75])
    for instance_idx, (means, label) in enumerate([(means1, 'Instance 1'), (means2, 'Instance 2')]):
        for horizon, color in zip([200, 2000], ['tab:blue', 'tab:orange']):
            regrets = {}
            for alg in ['LCB', 'UCB', 'OTO', 'SoftOTO']:
                vals = []
                for rep in range(200):
                    vals.append(run_bandit_UH(means, m_list, horizon, alg, alpha=0.6))
                regrets[alg] = np.mean(vals, axis=0)
                std = np.std(vals, axis=0)
                plt.plot(regrets[alg], label=alg)
                plt.fill_between(np.arange(horizon), regrets[alg]-2*std, regrets[alg]+2*std, alpha=0.2)
            plt.title(f"{label}, T={horizon}")
            plt.xlabel('iteration')
            plt.ylabel('Regret')
            plt.xlim(0, horizon)
            plt.legend()
            plt.tight_layout()
            plt.show()
