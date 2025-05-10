# RLSI252Project
Using for the course

The multi - armed bandit problem is a classic model for balancing exploration and exploitation in uncertain decision - making environments. Traditional online learning methods assume that learners start from scratch to collect data. However, in practical scenarios, learners often have access to some historical data. The OTO algorithm proposed in the paper "Balancing optimism and pessimism in offline - to - online learning"(https://arxiv.org/pdf/2502.08259) targets this offline - to - online learning scenario, aiming to balance optimistic and pessimistic strategies to meet the needs of different learning time horizons.

Objectives:
1.	To deeply understand the principles and mechanisms of the OTO algorithm, including how it dynamically adjusts its exploration strategy based on offline data.
2.	To accurately implement the OTO algorithm based on the paper's description, ensuring the correctness and readability of the code.
3.	To replicate the key experiments in the paper, including performance evaluations under different offline data distributions and time horizons, and to compare the performance of the OTO algorithm with that of the UCB and LCB algorithms, thereby verifying its advantages in offline - to - online learning scenarios.
4.	To explore potential improvement directions for the OTO algorithm, such as optimizing the budget calculation method or improving the mechanism for switching exploration strategies, and to validate the effectiveness of these improvements through experiments.


Expected Outcomes:

1.	Successfully implement the OTO algorithm and verify that its performance matches the description in the paper through experiments.
2.	Replicate the key experimental results in the paper, including the performance of the OTO algorithm under different offline data distributions and time horizons, as well as comparative analyses with the UCB and LCB algorithms.
3.	Attempt to propose an improvement plan for the OTO algorithm, verify it through experiments, and provide new ideas and methods for the development of the offline - to - online learning field.
