import numpy as np
import pandas as pd
from DGP import DGP, SieveBasis
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from utils_Estimate import Sieve_to_coeff, ConstructX4Smodel, ConstructX4Ymodel, calculateSigma_with_sieve, SignificanceTest



def EstimateStateModel(XtX, XtY, bt, Gnum=3, Gvec=None):
    if Gvec is None:
        raise ValueError("Please give a reference Gvec.")
    NN, PP, QQ1JJ, _ = XtX.shape
    Theta_hat = np.zeros((Gnum, PP, QQ1JJ, 1))

    for gg in range(Gnum):
        g_index = np.where(Gvec == gg)
        if len(g_index) == 0:
            continue

        XtX_gsum = np.sum(XtX[g_index], axis=(0))  # PP (QQ+1)*JJ (QQ+1)*JJ
        XtY_gsum = np.sum(XtY[g_index], axis=(0))  # PP (QQ+1)*JJ 1
        # eye = np.zeros((PP, QQ1JJ, QQ1JJ))
        # for pp in range(PP):
        #     eye[pp] = np.eye(QQ1JJ)
        # EIGS = np.linalg.eigvals(XtX_gsum)
        # EIGS_regularized = np.linalg.eigvals(XtX_gsum+eye)
        Theta_hat[gg] = np.einsum('ijk, ikl->ijl', np.linalg.pinv(XtX_gsum), XtY_gsum)

    # Theta_hat = np.minimum(Theta_hat, 5)
    coeff_hat = Sieve_to_coeff(Theta_hat, bt, mode="S")
    res_dict = {
        "Theta_hat": Theta_hat,
        "coeff_hat": coeff_hat,
    }

    return res_dict


def EstimateRewardModel(XtX, XtY, bt, Gnum=3, Gvec=None):
    if Gvec is None:
        raise ValueError("Please give a reference Gvec.")

    NN, _, PP3JJ, _ = XtX.shape
    Theta_hat = np.zeros((Gnum, 1, PP3JJ, 1))
    Gvec = np.array(Gvec)

    for gg in range(Gnum):
        g_index = np.where(Gvec == gg)
        if len(g_index) == 0:
            continue

        XtX_gsum = np.sum(XtX[g_index], axis=0)  # 1 (PP+3)*JJ (PP+3)*JJ
        XtY_gsum = np.sum(XtY[g_index], axis=0)  # 1 (PP+3)*JJ 1
        Theta_hat[gg] = np.einsum('ijk, ikl->ijl', np.linalg.pinv(XtX_gsum), XtY_gsum)

    # Theta_hat = np.minimum(Theta_hat, 5)
    coeff_hat = Sieve_to_coeff(Theta_hat, bt, mode="Y")
    res_dict = {
        "Theta_hat": Theta_hat,
        "coeff_hat": coeff_hat,
    }

    return res_dict


def UpdateGvec_with_S_and_Y(Sigma_SZ4S, Sigma_SXZ4S, Theta_hat4S,
                            tilde_x, Ytensor, Theta_hat4Y,
                            nu):
    # 计算矩阵与向量内积 (NN, GG, PP, KK, NNJJ, 1)
    product4S = np.einsum('npkiq, gpql->ngpkil', Sigma_SXZ4S, Theta_hat4S)
    residual4S = product4S - Sigma_SZ4S[:, None, :, :, :, :]
    loss4S = np.sum(residual4S ** 2, axis=(2, 3, 4, 5))

    inner_product4Y = np.einsum('ntpl, gipl -> ntgl', tilde_x, Theta_hat4Y)
    residual4Y = Ytensor[:, :, None, :] - inner_product4Y
    # 计算损失值：(NN, GG)
    loss4Y = np.sum(residual4Y ** 2, axis=(1, 3))
    loss = nu * loss4S + (1 - nu) * loss4Y
    Gvec = np.argmin(loss, axis=1)

    return Gvec

def UpdateGvec_nonYW(tilde_x4S, Stensor, Theta_hat4S,
                     tilde_x, Ytensor, Theta_hat4Y,
                     nu):
    # 计算矩阵与向量内积
    # tilde_x4S NN TT PP QQ1J 1
    # Theta_hat4S GG PP QQ1JJ 1
    # Stensor NN TT PP 1
    product4S = np.einsum('ntpqi, gpql->ngtpi', tilde_x4S, Theta_hat4S)
    residual4S = product4S - Stensor[:, None, :, :, :]
    loss4S = np.sum(residual4S ** 2, axis=(2, 3, 4))

    inner_product4Y = np.einsum('ntpl, gipl -> ntgl', tilde_x, Theta_hat4Y)
    residual4Y = Ytensor[:, :, None, :] - inner_product4Y
    # 计算损失值：(NN, GG)
    loss4Y = np.sum(residual4Y ** 2, axis=(1, 3))
    loss = nu * loss4S + (1 - nu) * loss4Y
    Gvec = np.argmin(loss, axis=1)

    return Gvec


def Iter_model(Timevec, S0mat, Stensor, Wmat, Ztensor, Ytensor, Atensor, Abar, Gnum=3, J0=2, Gvec_real=None,
               splinetype="b_spline", IV_type="z", max_iter=100, tol=1e-6, oracle=False, nu=1):
    """
        Iteratively compute Gvec and Theta_hat using a method similar to K-Means.

        Args:
            Timevec (numpy.ndarray): Time vector.
            S0mat (numpy.ndarray): Initial state matrix.
            Stensor (numpy.ndarray): State tensor.
            Wmat (numpy.ndarray): Weight matrix.
            Ztensor (numpy.ndarray): Input tensor.
            Gnum (int): Number of groups. Default is 3.
            J0 (int): Number of spline basis functions. Default is 2.
            Gvec (numpy.ndarray): Initial group vector. Default is None (random initialization).
            splinetype (str): Type of spline. Default is "cubic".
            max_iter (int): Maximum number of iterations. Default is 100.
            tol (float): Convergence tolerance for Gvec. Default is 1e-6.

        Returns:
            dict: A dictionary containing the final Gvec and Theta_hat.
        """
    NN, TT, PP, _ = Stensor.shape
    _, _, KK, _ = Ztensor.shape
    QQ = PP + KK + 4
    ST1 = np.concatenate([S0mat, Stensor[:, :-1, :, :]], axis=1)
    ST1bar = np.einsum('ij,jklm->iklm', Wmat, ST1[:, :, :, :])

    Xtensor4S = ConstructX4Smodel(ST1, ST1bar, Ztensor, Atensor, Abar)
    bt = SieveBasis(Timevec, spline_type=splinetype, J0=J0, visualize=False)

    # Sigma 的第三个维度是 工具变量的维度
    Sigma_SZ4S = np.zeros((NN, PP, NN * (J0 + 2), 1))
    Sigma_SXZ4S = np.zeros((NN, PP, NN * (J0 + 2), (QQ + 1) * (J0 + 2)))
    tilde_x4S = np.zeros((NN, TT, PP, (QQ+1)*(J0+2), 1))

    if IV_type == "z" or IV_type == "y":
        Sigma_SZ4S, Sigma_SXZ4S = calculateSigma_with_sieve(ST1, Stensor, Xtensor4S, Ztensor, Wmat,
                                                            bt, IV_type=IV_type)  # Sigma_SXZ: NN PP NN*JJ (QQ+1)*JJ
        XtX4S = np.einsum('ijKlm, ijKmn->ijKln', np.transpose(Sigma_SXZ4S, (0, 1, 2, 4, 3)), Sigma_SXZ4S)
        XtX4S = np.sum(XtX4S, axis=2, keepdims=False)
        XtY4S = np.einsum('ijKlm, ijKmn->ijKln', np.transpose(Sigma_SXZ4S, (0, 1, 2, 4, 3)), Sigma_SZ4S)
        XtY4S = np.sum(XtY4S, axis=2, keepdims=False)
    elif IV_type == "1":
        tilde_x4S = np.einsum("ntpq, tj -> ntpqj", np.concatenate((ST1, Xtensor4S), axis=-1), bt)
        tilde_x4S = tilde_x4S.reshape((NN, TT, PP, (QQ + 1) * (J0 + 2), 1))
        XtX4S = np.einsum("ntpim, ntpjm ->ntpij", tilde_x4S, tilde_x4S)
        XtX4S = np.sum(XtX4S, axis=1)
        XtY4S = np.einsum("ntpqm, ntpm -> ntpqm", tilde_x4S, Stensor)
        XtY4S = np.sum(XtY4S, axis=1)
    else:
        raise ValueError("Please give correct IV type.")

    # Xtensor4S NN TT PP QQ
    # Stensor NN TT PP 1
    # XtX4S NN PP QQ1J QQ1J

    Xtensor4Y = ConstructX4Ymodel(Stensor, Atensor, Abar)
    Xtensor4Y = Xtensor4Y.reshape((NN, TT, (PP + 3), 1))
    tilde_x = np.einsum('ntpk, tj -> ntpjk', Xtensor4Y, bt)
    tilde_x = tilde_x.reshape(NN, TT, (PP + 3) * (J0 + 2), 1)

    XtX4Y = np.einsum('ntij, ntkj->ntik', tilde_x, tilde_x)
    XtX4Y = np.sum(XtX4Y, axis=1, keepdims=True)
    XtY4Y = np.einsum('ntij, ntj->ntij', tilde_x, Ytensor)
    XtY4Y = np.sum(XtY4Y, axis=1, keepdims=True)

    Theta_hat4S = np.zeros((Gnum, PP, (QQ + 1) * (J0 + 4 - 2), 1))
    coeff_hat4S = np.zeros((Gnum, TT, PP, (QQ + 1)))
    Theta_hat4Y = np.zeros((Gnum, PP, (PP + 3) * (J0 + 4 - 2), 1))
    coeff_hat4Y = np.zeros((Gnum, TT, PP, (PP + 3)))

    if Gvec_real is None:
        pass
        # print("No Gvec real is given.")
        # raise ValueError("Please give Gvec real.")

    Gvec = np.random.choice([x for x in range(Gnum)], size=NN, replace=True)

    iter_count = 0

    if oracle:
        res_dict4S = EstimateStateModel(XtX4S, XtY4S, bt, Gnum=Gnum, Gvec=Gvec)
        res_dict4Y = EstimateRewardModel(XtX4Y, XtY4Y, bt, Gnum=Gnum, Gvec=Gvec)
        res_dict = {}
        res_dict["Gvec"] = Gvec
        res_dict["Theta_hat4S"] = res_dict4S["Theta_hat"]
        res_dict["Theta_hat4Y"] = res_dict4Y["Theta_hat"]
        res_dict["coeff_hat4S"] = res_dict4S["coeff_hat"]
        res_dict["coeff_hat4Y"] = res_dict4Y["coeff_hat"]
        res_dict["Sigma_SZ4S"] = Sigma_SZ4S
        res_dict["Sigma_SXZ4S"] = Sigma_SXZ4S
        return res_dict

    #############
    # IMPORTANT INITIAL VALUE!
    Gvec_pseudo = np.array([i for i in range(NN)])
    res_dict = EstimateStateModel(XtX4S, XtY4S, bt, Gnum=NN, Gvec=Gvec_pseudo)
    Theta_hat_by_ind = res_dict["Theta_hat"]
    hat_flattened = Theta_hat_by_ind.reshape(NN, -1)  # (NN, TT * PP * QQ)
    hat_flattened = np.clip(hat_flattened, a_min=-1, a_max=1)
    kmeans = KMeans(n_clusters=Gnum, init="k-means++", random_state=42)
    cluster_labels_hat = kmeans.fit_predict(hat_flattened)
    Gvec = cluster_labels_hat
    prev_Gvec = Gvec.copy()
    #############
    if Gvec_real is None:
        pass
    else:
        ARI = adjusted_rand_score(Gvec, Gvec_real)
        # if ARI < 0.1:
        #     Gvec = np.random.choice([gg for gg in range(Gnum)], size=NN)

        # ARI = adjusted_rand_score(Gvec, Gvec_real)
        # print(f"The classification ARI at iter count 0: {ARI}.")

    while iter_count < max_iter:
        iter_count += 1

        # Step 1: Estimate Theta_hat for S with current Gvec
        res_dict4S = EstimateStateModel(XtX4S, XtY4S, bt, Gnum=Gnum, Gvec=Gvec)
        Theta_hat4S = res_dict4S["Theta_hat"]
        coeff_hat4S = res_dict4S["coeff_hat"]

        # Step 2: Estimate Theta_hat for Y with current Gvec
        res_dict4Y = EstimateRewardModel(XtX4Y, XtY4Y, bt, Gnum=Gnum, Gvec=Gvec)
        Theta_hat4Y = res_dict4Y["Theta_hat"]
        coeff_hat4Y = res_dict4Y["coeff_hat"]

        if IV_type == "y" or IV_type == "z":
            # Step 1: Update Gvec based on the new Theta_hat
            Gvec = UpdateGvec_with_S_and_Y(Sigma_SZ4S=Sigma_SZ4S, Sigma_SXZ4S=Sigma_SXZ4S, Theta_hat4S=Theta_hat4S,
                                           tilde_x=tilde_x, Ytensor=Ytensor, Theta_hat4Y=Theta_hat4Y,
                                           nu=nu)
        elif IV_type == "1":
            Gvec = UpdateGvec_nonYW(tilde_x4S=tilde_x4S, Stensor=Stensor, Theta_hat4S=Theta_hat4S,
                                    tilde_x=tilde_x, Ytensor=Ytensor, Theta_hat4Y=Theta_hat4Y,
                                    nu=nu)
        else:
            raise ValueError("Please give correct IV type")

        # if Gvec_real is None:
        #     pass
        # else:
        #     ARI = adjusted_rand_score(Gvec, Gvec_real)
        #     print(f"The classification ARI at iter count {iter_count}: {ARI}.")

        # Check for convergence
        if np.allclose(Gvec, prev_Gvec, atol=tol):
            print(f"Converged after {iter_count} iterations.")
            break

        # Update previous Gvec
        prev_Gvec = Gvec.copy()

    else:
        print(f"Reached maximum iterations ({max_iter}) without convergence.")

    # print(f"Iter count: {iter_count} when estimating S model.")

    return ({"Gvec": Gvec, "Theta_hat4S": Theta_hat4S, "coeff_hat4S": coeff_hat4S,
            "Sigma_SZ4S": Sigma_SZ4S, "Sigma_SXZ4S": Sigma_SXZ4S},
            {"Gvec": Gvec,"Theta_hat4Y": Theta_hat4Y,"coeff_hat4Y": coeff_hat4Y, "tilde_x": tilde_x})



if __name__ == '__main__':
    print("Hello Brick!")
