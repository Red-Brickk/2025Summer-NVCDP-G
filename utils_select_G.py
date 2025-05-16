import numpy as np
import pandas as pd
from DGP import DGP, SieveBasis
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from Estimate import Iter_model

def GIC(GG, Theta_hat4S, Theta_hat4Y, Sigma_SZ, Sigma_SXZ,
        tilde_x, Ytensor, Gvec, lambda_NT, mu_G=0.9):
    # Sigma_SZ: NN PP KK NN1JJ 1
    # Sigma_SXZ: NN PP KK NN1JJ QQIJJ
    # Theta_hat4S: NN PP QQ1JJ 1
    # tilde_x: NN TT PP3J 1
    # Ytensor: NN TT PP3J 1
    # Theta_hat4Y: NN 1 PP3 1

    NN, _, PP3J, _ = Theta_hat4Y.shape
    Theta_hat4Y = Theta_hat4Y.reshape((NN, PP3J, 1))

    Qfunc4S = np.mean((Sigma_SZ - np.einsum("npkij, npjl->npkil", Sigma_SXZ, Theta_hat4S[Gvec]))**2, axis=(0,1,3,4))
    Qfunc4Y = np.mean((Ytensor - np.einsum('ntpi, npi -> nti', tilde_x, Theta_hat4Y[Gvec]))**2)

    var_S = np.var(Sigma_SZ, axis=2)
    Qfunc4S_normed = np.mean(Qfunc4S/(var_S+1e-5))

    var_Y = np.var(Ytensor)
    Qfunc4Y_normed = Qfunc4Y / (var_Y + 1e-5)

    gic = np.log((mu_G * Qfunc4S_normed + (1 - mu_G) * Qfunc4Y_normed)) + lambda_NT * GG
    return gic


def selectG(Timevec, S0, Stensor, W, Ztensor, Ytensor,A, Abar, J0, Glist, lambda_NT, mu_G=0.5):
    GIC_list = [0 for GG in Glist]
    for index, GG in enumerate(Glist):
        res_dict4S, res_dict4Y = Iter_model(
            Timevec=Timevec,
            S0mat=S0, Stensor=Stensor,
            Wmat=W, Ztensor=Ztensor,
            Ytensor=Ytensor,
            Atensor=A, Abar=Abar,
            Gnum=GG, J0=J0,
            max_iter=100, tol=1e-6
        )
        Gvec = res_dict4S["Gvec"]
        Theta_hat4S = res_dict4S["Theta_hat4S"]
        Theta_hat4Y = res_dict4Y["Theta_hat4Y"]
        Sigma_SZ = res_dict4S["Sigma_SZ4S"]
        Sigma_SXZ = res_dict4S["Sigma_SXZ4S"]
        tilde_x = res_dict4Y["tilde_x"]

        GIC_list[index] = GIC(GG=GG, Theta_hat4S=Theta_hat4S,
                              Theta_hat4Y=Theta_hat4Y, Sigma_SZ=Sigma_SZ,
                              Sigma_SXZ=Sigma_SXZ, tilde_x=tilde_x,
                              Ytensor=Ytensor, Gvec=Gvec, lambda_NT=lambda_NT, mu_G=mu_G)
        # 打印每个 G 对应的 GIC 值
        # print(f"G = {GG}, GIC = {GIC_list[index]:.6f}")


    # --- 碎石图（elbow plot） ---
    plt.figure(figsize=(8, 5))
    plt.plot(Glist, GIC_list, marker='o', linestyle='-', linewidth=5)
    plt.xlabel('Number of Groups (G)', fontsize=20)
    plt.ylabel('GIC Value', fontsize=20)
    # plt.title(f'GIC vs Number of Groups (G) lambda={lambda_NT} mu={mu_G}')
    plt.xticks(Glist)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"plots//Select_G_elbow.png")
    # # plt.show()
    plt.close()
    # --------------------------

    # 找到 GIC 最小值对应的 G
    min_GIC = min(GIC_list)  # 找到最小的 GIC 值
    best_G = Glist[GIC_list.index(min_GIC)]  # 找到对应的 G 值
    # print(f"Selected G with minimum GIC: G = {best_G}, GIC = {min_GIC:.6f}")

    return best_G

if __name__ == '__main__':
    print("Hello Brick!")
