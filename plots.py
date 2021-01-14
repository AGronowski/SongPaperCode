import matplotlib.pyplot as plt
import numpy as np


auc = []
dp = []
mi_xz_u = []
mi_z_u = []
with open('examples/results_mh_4.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        #ignore empty line
        if line.strip() == '':
            continue
        if line[0] == '#':
            continue
        split = line.split(' ')
        auc.append(float(split[1]))
        dp.append(float(split[3]))
        mi_xz_u.append(float(split[5]))
        mi_z_u.append(float(split[7]))


plt.title(r"Test AUC vs $I(X:Z|A)$")
plt.xlabel(r"$I(X;Z|A)$")
plt.ylabel("Test AUC")
plt.plot(mi_xz_u,auc,'o')
plt.show()

plt.title(r"$\Delta DP$ vs $I(Z;A)$ bound")
plt.xlabel(r"approximation of $I(Z;A)$ upper bound")
plt.ylabel(r"$\Delta DP$")
plt.plot(mi_z_u,dp,'o')
plt.show()

plt.title(r"Test AUC vs $\Delta DP$")
plt.xlabel(r"$\Delta DP$")
plt.ylabel("Test AUC")
plt.plot(dp,auc,'o')
plt.show()