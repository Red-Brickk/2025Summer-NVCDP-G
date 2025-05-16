import numpy as np
import torch
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from utils_Estimate import ConstructX4Smodel, ConstructX4Ymodel

# C-LASSO 目标函数
def classo_loss(beta4S_list, alpha4S_list, beta4Y_list, alpha4Y_list,
                X4Y, Y4Y, X4S, Y4S, kappa, p4S, p4Y, JJ, PP):
    N, T, _ = X4Y.shape
    mse_loss4Y = 0.0
    penalty4Y = 0.0
    mse_loss4S = 0.0
    penalty4S = 0.0

    for i in range(N):
        X4Yi = X4Y[i]  # (T, p)
        Y4Yi = Y4Y[i]  # (T,)
        beta4Y_i = beta4Y_list[i]  # (p,)
        resid4Y = Y4Yi - X4Yi @ beta4Y_i  # (T,)
        mse_loss4Y += torch.sum(resid4Y ** 2)

        # 乘积惩罚项
        prod4Y = 1.0
        for alpha_k in alpha4Y_list:
            prod4Y = prod4Y * torch.norm(beta4Y_i - alpha_k)  # 乘法保留梯度
        penalty4Y += prod4Y

    mse_loss4Y = mse_loss4Y / (N * T)
    penalty4Y = penalty4Y * (kappa / N)

    for i in range(N):
        X4Si = X4S[i]  # (T, p)
        Y4Si = Y4S[i]  # (T,)
        beta4S_i = beta4S_list[i]  # (p,)
        X4Si_reshaped = X4Si.reshape(N * JJ, PP, p4S)  # (T, Q, p)
        Y4Si_pred = torch.einsum('tqp,qp->tq', X4Si_reshaped, beta4S_i)  # (T, Q)
        resid4S = Y4Si - Y4Si_pred
        mse_loss4S += torch.sum(resid4S ** 2)

        # 乘积惩罚项
        prod4S = 1.0
        for alpha_k in alpha4S_list:
            prod4S *= torch.norm(beta4S_i - alpha_k, p='fro')  # 乘法保留梯度
        penalty4S += prod4S

    mse_loss4S = mse_loss4S / (N * N * JJ)
    penalty4S = penalty4S * (kappa / N)

    return mse_loss4Y + penalty4Y + mse_loss4S + penalty4S



def classification_evaluation(group_assignments, group_assignment_est):
    # 构造混淆矩阵（行：真实组，列：估计组）
    conf_mat = confusion_matrix(group_assignments, group_assignment_est)

    # 使用 Hungarian 算法找到最优标签匹配
    row_ind, col_ind = linear_sum_assignment(-conf_mat)  # 注意是最大匹配，用负号

    # 构造映射表：估计组号 -> 对应的真实组号
    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # 应用映射，将 group_assignment_est 转换为与真实标签对齐的版本
    group_assignment_aligned = np.array([mapping[g] for g in group_assignment_est])

    # 计算准确率
    acc = accuracy_score(group_assignments, group_assignment_aligned)
    print(f"分类准确率: {acc:.4f}")


def construct_XY4S(ST1, Stensor, Ztensor, Atensor, Abar, Wmat, bt, NN, TT, PP):
    JJ = bt.shape[-1]

    Sbar = np.einsum('ij,jklm->iklm', Wmat, Stensor[:, :, :, :])
    ST1bar = np.einsum('ij,jklm->iklm', Wmat, ST1[:, :, :, :])
    Xtensor = ConstructX4Smodel(ST1, ST1bar, Ztensor, Atensor, Abar)
    XSbar = np.concatenate((Xtensor, Sbar), axis=-1)
    _, _, _, QQ1 = XSbar.shape
    IV = np.zeros((TT, NN*JJ))
    for tt in range(1, TT):
        IV[tt] = np.kron(Ztensor[:, tt, 0, 0], np.reshape(bt[tt:(tt + 1), :], newshape=(JJ, )))

    Stensor_exp = Stensor

    Y4S_temp = Stensor_exp[:, :, np.newaxis, :] *IV[np.newaxis, :, :, np.newaxis]
    Y4S = np.mean(Y4S_temp, axis=1, keepdims=False) # NN TT PP*NN*JJ

    Sigma_XSbarZ = np.zeros((NN, PP, TT, NN*JJ, QQ1*JJ))

    for tt in range(TT):
        IV_t = IV[tt]
        b_t = bt[tt]

        XS_tt = XSbar[:, tt, :, :]
        XS_tt_reshape = XS_tt.reshape(NN*PP, QQ1)
        kron_all = np.array([np.kron(xs, b_t) for xs in XS_tt_reshape]) # NN*PP QQ1*JJ
        # IV_t NN*JJ, kron_all NN*PP QQ1*JJ
        Sigma_tt = IV_t[:, np.newaxis, np.newaxis] * kron_all.T[np.newaxis, :, :] # NN*JJ NN*PP QQ1*JJ
        Sigma_XSbarZ[:, :, tt, :, :] += Sigma_tt.T.reshape(NN, PP, NN*JJ, QQ1*JJ)

    # for ii in range(NN):
    #     for pp in range(PP):
    #         for tt in range(1, TT):
    #             Sigma_XSbarZ[ii, pp, tt, :, :] += IV[tt] @ np.kron(
    #                 XSbar[ii, tt, pp:(pp + 1), :], bt[tt:(tt + 1), :])

    Sigma_XSbarZ = np.mean(Sigma_XSbarZ, axis=2, keepdims=True) # NN PP NN*JJ QQ1*JJ
    X4S = Sigma_XSbarZ.reshape(NN, NN*JJ, PP, QQ1*JJ) # NN TT PP*NN*JJ QQ1*JJ

    return X4S, Y4S

def construct_XY4Y(Stensor, Atensor, Abar, Ytensor, bt, NN, TT):
    JJ = bt.shape[-1]
    Y4Y = Ytensor.reshape((NN,TT,1)) # NN TT 1
    Xtensor4Y = ConstructX4Ymodel(Stensor, Atensor, Abar)
    # bt TT JJ
    _, _, _, PP3 = Xtensor4Y.shape
    kron_all = Xtensor4Y * bt[np.newaxis, :, :, np.newaxis]

    X4Y = kron_all.reshape(NN, TT, PP3*JJ) # NN TT PP3*JJ

    return X4Y, Y4Y







