import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
from DGP import SieveBasis
from utils import fill_in_res_mat, fill_in_rep_mat, plot_residuals, plot_coefficients
from utils_Fit import Fit4S, Fit4Y
from Classo_util import classo_loss, classification_evaluation, construct_XY4S, construct_XY4Y
from utils_Estimate import Sieve_to_coeff
from tqdm import tqdm
from DGP import DGP


# 设置随机种子确保可重复
np.random.seed(0)

# 模拟数据参数
NN_list = [100]
TT_list = [50, 100, 200]
KK = 1
PP = 1
QQ = 2
GG = 3    # 隐藏组数
J0_list = [4]
oracle_list = [False]
rep_total = 10
rep_list = [ii for ii in range(rep_total)]
IV_code = 3

# network params
K4net = 4
sbm_lambda = 0.5  # (1-sbm_lambda) is the probability of an edge between different communities
sbm_nu = 0.5  # the connecting probability bonus within community
sbm_mat = sbm_nu * (sbm_lambda * np.eye(K4net)) + (1 - sbm_lambda) * np.ones((K4net, K4net))
sbm_param_dict = {"K": K4net, "sbm_matrix": sbm_mat}
BA_param_dict = {"m": 3}
eps_sigma = 0.01
splinetype = "b_spline"
seed = 2024

SAVE_MODEL = False

# store result
Ghat_rep_mat = np.zeros((len(rep_list), 1))

ii = 0
start_time = time.time()
res_mat = np.zeros((len(NN_list) * len(TT_list) * len(oracle_list) * len(J0_list), (11 + 3 * 4)))
res_mat4Y = np.zeros((len(NN_list) * len(TT_list) * len(oracle_list) * len(J0_list), (11 + 3 * 4)))
res_mat4G = np.zeros((len(NN_list) * len(TT_list) * len(oracle_list) * len(J0_list), 5))

for N in NN_list:
    for T in TT_list:
        for J0 in J0_list:
            JJ = J0 + 2
            rep_res_mat = np.zeros((len(rep_list), 4))
            rep_res_mat4Y = np.zeros((len(rep_list), 4))
            for oracle in oracle_list:
                for rep in tqdm(rep_list, desc=f"NN={N}, TT={T}, J0={J0}, oracle={oracle}"):
                    # 隐藏组的真实回归系数（每个组一个 alpha）
                    # alphas4Y_true = [np.random.randn(p4Y) for _ in range(GG)]
                    # alphas4S_true = [np.random.randn(QQ, p4S) for _ in range(GG)]
                    # # 每个个体分到一个组
                    # group_assignments = np.random.choice(GG, size=N)
                    #
                    # # 构造 X 和 Y
                    # X4Y = np.random.randn(N, T, p4Y)
                    # Y4Y = np.zeros((N, T))
                    # X4S = np.random.randn(N, T, QQ*p4S)
                    # Y4S = np.zeros((N, T, QQ))

                    # for i in range(N):
                    #     alpha4S_i = alphas4S_true[group_assignments[i]]  # (Q, p)
                    #     X4S_i = X4S[i].reshape(T, QQ, p4S)  # (T, Q, p)
                    #     for t in range(T):
                    #         Y4S[i, t] = np.einsum('qp,qp->q', X4S_i[t], alpha4S_i) + np.random.randn(QQ) * 0.5
                    #
                    #     alpha4Y_i = alphas4Y_true[group_assignments[i]]
                    #     Y4Y[i] = X4Y[i] @ alpha4Y_i + np.random.randn(T) * 0.5  # 加入噪声

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                    dgp = DGP(NN=N, TT=T, PP=PP, KK=1, GG=GG, net_name="sbm", eps_sigma=eps_sigma,
                              net_extra_param_dict=sbm_param_dict, spline_type=splinetype)
                    dgp.GenerateStateTensor()
                    dgp.GenerateRewardTensor()

                    group_assignments = dgp.Gvec

                    bt = SieveBasis(dgp.Timevec, spline_type=splinetype, J0=J0, visualize=False)
                    ST1 = np.concatenate([dgp.S0, dgp.Stensor[:, :-1, :, :]], axis=1)
                    X4S, Y4S = construct_XY4S(
                        ST1=ST1,
                        Stensor=dgp.Stensor,
                        Ztensor=dgp.Ztensor,
                        Atensor=dgp.A,
                        Abar=dgp.Abar,
                        Wmat=dgp.W,
                        bt=bt,
                        NN=N, TT=T, PP=PP
                    )

                    X4Y, Y4Y = construct_XY4Y(
                        Stensor=dgp.Stensor,
                        Atensor=dgp.A,
                        Abar=dgp.Abar,
                        Ytensor=dgp.Ytensor,
                        bt=bt,
                        NN=N, TT=T
                    )

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    # 超参数
                    kappa = 1.0
                    _, _, _, p4S = X4S.shape
                    _, _, p4Y = X4Y.shape
                    # 转换数据为 torch.Tensor
                    X4Y_torch = torch.tensor(X4Y, dtype=torch.float32)  # (N, T, Q)
                    Y4Y_torch = torch.tensor(Y4Y, dtype=torch.float32)  # (N, T)
                    X4S_torch = torch.tensor(X4S, dtype=torch.float32)  # (N, PP*NN*JJ, QQ1*JJ)
                    Y4S_torch = torch.tensor(Y4S, dtype=torch.float32)  # (N, PP*NN*JJ)

                    # 初始化 beta_i 和 alpha_k（需要梯度）
                    beta4S_list = nn.ParameterList([nn.Parameter(torch.randn(PP, p4S)) for _ in range(N)])
                    alpha4S_list = nn.ParameterList([nn.Parameter(torch.randn(PP, p4S)) for _ in range(GG)])
                    beta4Y_list = nn.ParameterList([nn.Parameter(torch.randn(p4Y)) for _ in range(N)])
                    alpha4Y_list = nn.ParameterList([nn.Parameter(torch.randn(p4Y)) for _ in range(GG)])


                    # 收集所有参数
                    all_params = list(beta4S_list.parameters()) + list(alpha4S_list.parameters()) + list(beta4Y_list.parameters()) + list(alpha4Y_list.parameters())

                    # 构建优化器
                    optimizer = torch.optim.Adam(all_params, lr=0.05)

                    # 训练循环
                    num_epochs = 100
                    for epoch in tqdm(range(num_epochs), desc="Training C-LASSO", miniters=100):
                        optimizer.zero_grad()
                        loss = classo_loss(beta4S_list, alpha4S_list, beta4Y_list, alpha4Y_list,
                                    X4Y_torch, Y4Y_torch, X4S_torch, Y4S_torch, kappa, p4S=p4S, p4Y=p4Y, PP=PP, JJ=JJ)
                        loss.backward()
                        optimizer.step()

                        if (epoch + 1) % 50 == 0 or epoch == 0:
                            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")


                    # 将 beta 和 alpha 转为 NumPy 方便后续使用
                    beta4Y_est = [b.detach().numpy() for b in beta4Y_list]
                    alpha4Y_est = [a.detach().numpy() for a in alpha4Y_list]
                    beta4S_est = [b.detach().numpy() for b in beta4S_list]
                    alpha4S_est = [a.detach().numpy() for a in alpha4S_list]

                    # 聚类分组：每个 beta_i 分配到最近的 alpha_k
                    group_assignment_est = []

                    for i in range(N):
                        distances4S = [np.linalg.norm(beta4S_est[i] - alpha_k) for alpha_k in alpha4S_est]
                        distances4Y = [np.linalg.norm(beta4Y_est[i] - alpha_k) for alpha_k in alpha4Y_est]
                        distances_total = [distances4S[k] + distances4Y[k] for k in range(GG)]
                        assigned_group = np.argmin(distances_total)  # 最小距离对应的组
                        group_assignment_est.append(assigned_group)

                    group_assignment_est = np.array(group_assignment_est)  # (N,)

                    classification_evaluation(group_assignments=group_assignments,
                                              group_assignment_est=group_assignment_est)

                    Theta_hat4S =np.stack(alpha4S_est, axis=0)[:,:,:,np.newaxis]
                    Theta_hat4Y =np.stack(alpha4Y_est, axis=0)[:,np.newaxis,:,np.newaxis]

                    coeff_hat4S = Sieve_to_coeff(Theta_hat4S, bt, mode="S")
                    coeff_hat4Y = Sieve_to_coeff(Theta_hat4Y, bt, mode="Y")

                    res_dict4S = {"Gvec": group_assignment_est, "Theta_hat4S": alpha4S_est, "coeff_hat4S": coeff_hat4S,
                                "Sigma_SZ4S": Y4S, "Sigma_SXZ4S": X4S}
                    res_dict4Y = {"Gvec": group_assignment_est,"Theta_hat4Y": alpha4Y_est,"coeff_hat4Y": coeff_hat4Y, "tilde_x": X4Y}


                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    rep_res_mat[rep, :] = fill_in_rep_mat(res_dict4S, dgp, mode="S")
                    rep_res_mat4Y[rep, :] = fill_in_rep_mat(res_dict4Y, dgp, mode="Y")
                    if rep==0:
                        plot_coefficients(res_dict4S, dgp, mode="S", plot_true=True)
                        plot_coefficients(res_dict4Y, dgp, mode="Y", plot_true=True)

                    StensorFit = Fit4S(res_dict4S, S0mat=dgp.S0, Stensor=dgp.Stensor,
                                       Ztensor=dgp.Ztensor, Atensor=dgp.A, Wmat=dgp.W)
                    YtensorFit = Fit4Y(res_dict4Y, Stensor=dgp.Stensor, Atensor=dgp.A, Wmat=dgp.W)

                    if rep == 0:
                        # plot_time_series(StensorFit, Gvec=res_dict["Gvec"], mode="S")
                        plot_residuals(Tensor=dgp.Stensor, Tensor_fit=StensorFit, mode="S")
                        plot_residuals(Tensor=dgp.Ytensor, Tensor_fit=YtensorFit, mode="Y")

                    if SAVE_MODEL:
                        Smodel_params, Ymodel_params, network_params = dgp.dump_coeffs()
                        res_dict4S = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for
                                      key, value in res_dict4S.items()}
                        res_dict4Y = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for
                                      key, value in res_dict4Y.items()}

                        data_to_save = {
                            "NN": N, "TT": T, "PP": PP, "KK": KK, "W": dgp.W.tolist(),
                            "Smodel_params": Smodel_params,
                            "Ymodel_params": Ymodel_params,
                            "network_params": network_params,
                            "res_dict4S": res_dict4S,
                            "res_dict4Y": res_dict4Y
                        }
                        filename = f"json//results_NN{N}_TT{T}_oracle{oracle}_J0{J0}_rep{rep}.json"

                        # 将数据保存为 JSON 文件
                        # with open(filename, "w") as json_file:
                        #     json.dump(data_to_save, json_file, indent=4)
                        # print(f"Saved data to {filename}.")
                        # print(f"Total reward: {np.sum(dgp.Ytensor)}.")

    # # Estimating GIC
    # unique_elements, counts = np.unique(Ghat_rep_mat, return_counts=True)
    # res_mat4G[ii, 0:10] = [NN, TT, J0, oracle, PP, KK, sbm_lambda, sbm_nu, eps_sigma, rep_total]
    # for index, elements in enumerate(unique_elements):
    #     res_mat4G[ii, 10 + int(elements - 1)] = counts[index]

                res_mat[ii, 0:11] = [IV_code, N, T, J0, oracle, PP, KK, sbm_lambda, sbm_nu, eps_sigma, rep_total]
                res_mat[ii, 11:] = fill_in_res_mat(rep_res_mat=rep_res_mat)
                res_mat4Y[ii, 0:11] = [IV_code, N, T, J0, oracle, PP, KK, sbm_lambda, sbm_nu, eps_sigma,
                                       rep_total]
                res_mat4Y[ii, 11:] = fill_in_res_mat(rep_res_mat=rep_res_mat4Y)

                ii += 1

# 将结果保存为 CSV 文件
columns = ['IV_type', 'NN', 'TT', 'J0', "oracle", "PP", "KK", "sbm_lambda", "sbm_nu", "eps_sigma", "rep",
'Mean_ARI', 'Mean_NMI', 'Mean_l2_norm_by_ind', 'Mean_l1_norm_by_ind', 'Std_ARI', 'Std_NMI',
'Std_l2_norm_by_ind', 'Std_l1_norm_by_ind', 'RMSE_ARI', 'RMSE_NMI', 'RMSE_l2_norm_by_ind', 'RMSE_l1_norm_by_ind']
res_df = pd.DataFrame(res_mat, columns=columns)
res_df.to_csv(
f"table//CLASSO results iteration S model NN={NN_list[0]} IV type={IV_code}.csv", index=False)

res_df4Y = pd.DataFrame(res_mat4Y, columns=columns)
res_df4Y.to_csv(f"table//CLASSO results iteration Y model NN={NN_list[0]} IV type={IV_code}.csv", index=False)

