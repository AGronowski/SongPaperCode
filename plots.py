import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

auc = []
dp = []
mi_xz_u = []
mi_z_u = []
accgap = []
e1 = []
e2 = []

with open('examples/results_mh_5.txt', 'r') as f:
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
        e1.append(float(split[9]))
        e2.append(float(split[11]))
        accgap.append(float(split[13]))

x=np.array(e1)
y=np.array(e2)
z=np.array(auc)

# Creating figure
fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection="3d")

# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

# Creating color map
# my_cmap = plt.get_cmap('hsv')

# Creating plot
sctt = ax.scatter3D(x, y, z,
                    alpha=0.8,
                    c=z,
                    marker='^')

plt.title(r"Song AUC vs $\epsilon_1$ and $\epsilon_2$")
ax.set_xlabel(r'$\epsilon_1$ I(Z;A) true upper bound ')
ax.set_ylabel(r'$\epsilon_2$ I(Z;A) tighter upper bound')
ax.set_zlabel('auc')
fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
plt.savefig("../../../plots/song3dauc.png",bbox_inches ="tight")
# show plot
plt.show()

z=dp

# Creating figure
fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection="3d")

# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

sctt = ax.scatter3D(x, y, z,
                    alpha=0.8,
                    c=z,
                    marker='^')

plt.title(r"Song Discrimination vs $\epsilon_1$ and $\epsilon_2$")
ax.set_xlabel(r'$\epsilon_1$ I(Z;A) true upper bound ')
ax.set_ylabel(r'$\epsilon_2$ I(Z;A) tighter upper bound')
ax.set_zlabel('discrimination gap')
fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
# show plot
plt.savefig("../../../plots/song3ddp.png",bbox_inches="tight")
plt.show()

# plt.title(r"Test AUC vs $\Delta DP$")
# plt.xlabel(r"$\Delta DP$")
# plt.ylabel("Test AUC")
# plt.plot(dp,auc,'o')
# plt.show()
#
#
#
# plt.title(r"Test AUC vs $I(X:Z|A)$")
# plt.xlabel(r"$I(X;Z|A)$")
# plt.ylabel("Test AUC")
# plt.plot(mi_xz_u,auc,'o')
# plt.show()
#
# plt.title(r"$\Delta DP$ vs $I(Z;A)$ bound")
# plt.xlabel(r"approximation of $I(Z;A)$ upper bound")
# plt.ylabel(r"$\Delta DP$")
# plt.plot(mi_z_u,dp,'o')
# plt.show()
#
