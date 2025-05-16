import numpy as np
from scipy.stats import t

def ConstructX4Smodel(ST1, ST1bar, Ztensor, Atensor, Abar):
    NN, TT, PP, _ = ST1.shape
    _, _, KK, _ = Ztensor.shape
    Xtensor = np.concatenate((
        np.ones((NN, TT, PP, 1)),
        np.tile(ST1, (1, 1, 1, PP)),
        ST1bar,
        np.tile(Atensor, reps=(1, 1, PP, 1)),
        np.tile(Abar, reps=(1, 1, PP, 1)),
        np.tile(np.reshape(Ztensor, newshape=(NN, TT, 1, KK)), reps=(1, 1, PP, 1))
    ), axis=-1)

    return Xtensor  ## NN TT PP QQ


def ConstructX4Ymodel(Stensor, Atensor, Abar):
    NN, TT, PP, _ = Stensor.shape
    Xtensor = np.concatenate((
        np.ones((NN, TT, 1, 1)),
        np.reshape(Stensor, (NN, TT, 1, PP)),
        np.tile(Atensor, reps=(1, 1, 1, 1)),
        np.tile(Abar, reps=(1, 1, 1, 1)),
    ), axis=-1)

    return Xtensor  ## NN TT 1 PP3

def calculateSigma_with_sieve(ST1, Stensor, Xtensor, Ztensor, Wmat, bt=None, IV_type="z"):
    """
    :param Stensor: NN*TT*PP
    :param Xtensor: NN*TT*PP*QQ
    :return:
    """

    NN, TT, PP, _ = ST1.shape
    _, _, KK, _ = Ztensor.shape
    JJ = bt.shape[-1]
    QQ = PP + KK + 4

    Sbartensor = np.einsum('ij,jklm->iklm', Wmat, Stensor[:, :, :, :])

    if IV_type == "z" or IV_type == "y":
        IV_dim = NN * JJ
    else:
        raise ValueError("Please give legal IV type!")

    # for each coefficient
    Sigma_SZ = np.zeros((NN, PP, KK, IV_dim, 1))  # NN PP KK NN*JJ 1
    Sigma_SbarZ = np.zeros((NN, PP, KK, IV_dim, JJ))  # NN PP KK NN*JJ JJ
    Sigma_XZ = np.zeros((NN, PP, KK, IV_dim, QQ * JJ))  # NN PP KK NN*JJ QQ*JJ

    for ii in range(NN):
        for pp in range(PP):
            for kk in range(KK):
                for tt in range(1, TT):
                    if IV_type == "z":
                        IV = np.kron(Ztensor[:, tt, kk, :],
                                     np.reshape(bt[tt:(tt + 1), :], newshape=(JJ, 1))
                                     )  # NN*JJ 1
                    elif IV_type == "y":
                        IV = np.kron(ST1[:, tt, pp, :],
                                     np.reshape(bt[tt:(tt + 1), :], newshape=(JJ, 1))
                                     )  # NN*JJ 1
                    else:
                        raise ValueError("Please give legal IV type!")

                    # IV = Ztensor[:, tt, kk, :]
                    Sigma_SZ[ii, pp, kk, :, :] += Stensor[ii, tt, pp] * IV
                    Sigma_SbarZ[ii, pp, kk, :, :] += IV @ np.kron(
                        Sbartensor[ii, tt, pp:(pp + 1), :], bt[tt:(tt + 1), :])
                    Sigma_XZ[ii, pp, kk, :, :] += IV @ np.kron(
                        Xtensor[ii, tt, pp:(pp + 1), :],
                        bt[tt:(tt + 1), :])

    return Sigma_SZ / TT, np.concatenate((Sigma_SbarZ / TT, Sigma_XZ / TT), axis=-1)


def Sieve_to_coeff(Theta_hat, bt, mode):
    TT, JJ = bt.shape
    if mode == "S":
        GG, PP, Q1J, _ = Theta_hat.shape
        QQ1 = int(Q1J / JJ)
        coeff_hat = np.zeros(shape=(GG, TT, PP, QQ1))
        for tt in range(TT):
            ## (QQ QQ1JJ) @ GG PP Q1J 1 GG PP QQ 1
            coeff_hat[:, tt:(tt + 1), :, :] = np.reshape(
                np.einsum('il, jklm-> jkim', np.kron(np.eye(QQ1), bt[tt:(tt + 1), ]), Theta_hat),
                newshape=(GG, 1, PP, QQ1))

    elif mode == "Y":
        GG, _, P2J, _ = Theta_hat.shape
        PP3 = int(P2J / JJ)
        coeff_hat = np.zeros(shape=(GG, TT, PP3, 1))
        for tt in range(TT):
            ## (PP2 PP21JJ) @ GG 1 P2J 1 = GG 1 PP2 1
            coeff_hat[:, tt:(tt + 1), :, :] = np.reshape(
                np.einsum('il, jklm-> jkim', np.kron(np.eye(PP3), bt[tt:(tt + 1), ]), Theta_hat),
                newshape=(GG, 1, PP3, 1))

    else:
        raise ValueError("Please give correct estimation mode.")

    return coeff_hat  # GG TT PP Q1 or GG TT PP3 1


def SignificanceTest(Theta_hat, X_list, Y_list):
    Gnum, PP, QQ1JJ, _ = Theta_hat.shape
    t_values = np.zeros((Gnum, PP, QQ1JJ))
    p_values = np.ones((Gnum, PP, QQ1JJ))

    for gg in range(Gnum):
        Xg = X_list[gg]  # shape: (n_g, PP, QQ1JJ)
        Yg = Y_list[gg]  # shape: (n_g, PP)
        n_g = Xg.shape[0]

        if n_g <= QQ1JJ:
            continue  # 自由度不足，跳过

        for pp in range(PP):
            Xg_pp = Xg[:, pp, :]  # (n_g, QQ1JJ)
            Yg_pp = Yg[:, pp]     # (n_g,)
            Theta_pp = Theta_hat[gg, pp, :, 0]  # (QQ1JJ,)

            # 预测值与残差
            Y_pred = Xg_pp @ Theta_pp
            residuals = Yg_pp - Y_pred
            sigma2 = np.sum(residuals**2) / (n_g - QQ1JJ)  # 残差方差

            # 协方差矩阵
            XtX = Xg_pp.T @ Xg_pp
            XtX_inv = np.linalg.pinv(XtX)
            se = np.sqrt(np.diag(sigma2 * XtX_inv))  # 标准误差 (QQ1JJ,)

            # t 统计量与 p 值
            with np.errstate(divide='ignore', invalid='ignore'):
                t_stat = Theta_pp / se
                p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=n_g - QQ1JJ))

            t_values[gg, pp, :] = t_stat
            p_values[gg, pp, :] = p_val

    return {
        "t_values": t_values,
        "p_values": p_values
    }
