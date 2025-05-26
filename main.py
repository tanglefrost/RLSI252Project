import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
import Plot,Algorithms,Run_Bandit


print("Certain horizon experiment")
Plot.plot_CertainH()

print("Unknown horizon experiment")
Plot.plot_UnknownH()
