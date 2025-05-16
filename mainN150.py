import time
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from DGP import DGP
from utils import (visualize_grouping, fill_in_rep_mat,
                   fill_in_res_mat, plot_coefficients, plot_time_series, plot_residuals)
from Estimate import Iter_model
from utils_Fit import Fit4S, Fit4Y
from utils_select_G import selectG

if __name__ == '__main__':
    print("Hello Brick!")
    # settings
    NN_list = [150]  # the number of subjects
    TT_list = [300, 500]  # the length of horizon
    c_lambda1_list = [1e-5]
    oracle_list = [False]
    J0_list = [4]
    PP = 1
    KK = 1
    QQ = PP + KK + 4
    GG = 3
    splinetype = "b_spline"

    IV_type_list = ["1", "z", "y"]
    IV_code = 0
    IV_type = IV_type_list[IV_code]
    rep_total = 200
    rep_list = [ii for ii in range(rep_total)]

    # network params
    K4net = 4
    sbm_lambda = 0.5  # (1-sbm_lambda) is the probability of an edge between different communities
    sbm_nu = 0.5  # the connecting probability bonus within community
    sbm_mat = sbm_nu * (sbm_lambda * np.eye(K4net)) + (1 - sbm_lambda) * np.ones((K4net, K4net))
    sbm_param_dict = {
        "K": K4net,
        "sbm_matrix": sbm_mat
    }
    BA_param_dict = {
        "m": 3
    }
    eps_sigma = 0.01
    seed = 2024

    # parameter for GIC
    lambda_NT_list = [0.2]
    Glist = [1, 2, 3, 4, 5]

    # store result
    rep_res_mat = np.zeros((len(rep_list), 6))
    rep_res_mat4Y = np.zeros((len(rep_list), 6))
    Ghat_rep_mat = np.zeros((len(rep_list), 1))

    SAVE_MODEL = False

    ii = 0
    start_time = time.time()
    res_mat = np.zeros((len(NN_list) * len(TT_list) * len(oracle_list) * len(J0_list), (11 + 3 * 4)))
    res_mat4Y = np.zeros((len(NN_list) * len(TT_list) * len(oracle_list) * len(J0_list), (11 + 3 * 4)))
    res_mat4G = np.zeros((len(NN_list) * len(TT_list) * len(oracle_list) * len(J0_list), 5))

    for NN in NN_list:
        # sbm_param_dict["sbm_matrix"] = 0.1 *((np.log(NN)/NN) * (np.eye(K4net)) + (np.log(NN)/NN) * np.ones((K4net, K4net)))
        for TT in TT_list:
            for oracle in oracle_list:
                for J0 in J0_list:
                    rep_res_mat = np.zeros((len(rep_list), 4))
                    rep_res_mat4Y = np.zeros((len(rep_list), 4))
                    for rep in tqdm(rep_list, desc=f"NN={NN}, TT={TT}, J0={J0}, oracle={oracle}"):
                        dgp = DGP(NN=NN, TT=TT, PP=PP, KK=KK, GG=GG, net_name="sbm", eps_sigma=eps_sigma,
                                  net_extra_param_dict=sbm_param_dict, seed=(seed + rep), spline_type=splinetype)
                        dgp.GenerateStateTensor()
                        dgp.GenerateRewardTensor()
                        if rep == 0:
                            # pass
                            dgp.plot_time_series()
                            dgp.visualize_network()

                        if oracle:
                            Gvec = dgp.Gvec
                        else:
                            Gvec = np.random.choice([x for x in range(GG)], size=NN, replace=True)

                        # # Testing GIC
                        # for lambda_NT in lambda_NT_list:
                        #     Ghat = selectG(
                        #         Timevec=dgp.Timevec,
                        #         S0=dgp.S0, Stensor=dgp.Stensor,
                        #         W=dgp.W, Ztensor=dgp.Ztensor,
                        #         Ytensor=dgp.Ytensor,
                        #         A=dgp.A, Abar=dgp.Abar,
                        #         J0=J0, Glist=Glist, lambda_NT=lambda_NT
                        #     )
                        #
                        #     Ghat_rep_mat[rep] = Ghat

                        res_dict4S, res_dict4Y = Iter_model(
                            Timevec=dgp.Timevec,
                            S0mat=dgp.S0, Stensor=dgp.Stensor,
                            Wmat=dgp.W, Ztensor=dgp.Ztensor,
                            Ytensor=dgp.Ytensor,
                            Atensor=dgp.A, Abar=dgp.Abar,
                            Gnum=dgp.GG, J0=J0,
                            Gvec_real=dgp.Gvec,
                            splinetype=splinetype,
                            IV_type=IV_type,
                            max_iter=100, tol=1e-6, nu=0.9, oracle=oracle
                        )

                        rep_res_mat[rep, :] = fill_in_rep_mat(res_dict4S, dgp, mode="S")
                        # plot_coefficients(res_dict4S, dgp, mode="S", plot_true=True)

                        rep_res_mat4Y[rep, :] = fill_in_rep_mat(res_dict4Y, dgp, mode="Y")
                        # plot_coefficients(res_dict4Y, dgp, mode="Y", plot_true=True)

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
                                "NN": NN, "TT": TT, "PP": PP, "KK": KK, "W": dgp.W.tolist(),
                                "Smodel_params": Smodel_params,
                                "Ymodel_params": Ymodel_params,
                                "network_params": network_params,
                                "res_dict4S": res_dict4S,
                                "res_dict4Y": res_dict4Y
                            }
                            filename = f"json//results_NN{NN}_TT{TT}_oracle{oracle}_J0{J0}_rep{rep}.json"

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

                    res_mat[ii, 0:11] = [IV_code, NN, TT, J0, oracle, PP, KK, sbm_lambda, sbm_nu, eps_sigma, rep_total]
                    res_mat[ii, 11:] = fill_in_res_mat(rep_res_mat=rep_res_mat)
                    res_mat4Y[ii, 0:11] = [IV_code, NN, TT, J0, oracle, PP, KK, sbm_lambda, sbm_nu, eps_sigma,
                                           rep_total]
                    res_mat4Y[ii, 11:] = fill_in_res_mat(rep_res_mat=rep_res_mat4Y)

                    ii += 1

    # 将结果保存为 CSV 文件
    columns = ['IV_type', 'NN', 'TT', 'J0', "oracle", "PP", "KK", "sbm_lambda", "sbm_nu", "eps_sigma", "rep",
               'Mean_ARI', 'Mean_NMI', 'Mean_l2_norm_by_ind', 'Mean_l1_norm_by_ind', 'Std_ARI', 'Std_NMI',
               'Std_l2_norm_by_ind', 'Std_l1_norm_by_ind', 'RMSE_ARI', 'RMSE_NMI', 'RMSE_l2_norm_by_ind', 'RMSE_l1_norm_by_ind']
    res_df = pd.DataFrame(res_mat, columns=columns)
    res_df.to_csv(
        f"table//results iteration S model NN={NN_list[0]} IV type={IV_code}.csv", index=False)

    res_df4Y = pd.DataFrame(res_mat4Y, columns=columns)
    res_df4Y.to_csv(f"table//results iteration Y model NN={NN_list[0]} IV type={IV_code}.csv", index=False)

    # columns4G = ['NN', 'TT', 'J0', "oracle","PP", "KK", "sbm_lambda", "sbm_nu", "eps_sigma","rep",
    #              "1", "2", "3", "4", "5"]
    #
    # rep_mat4G = pd.DataFrame(Ghat_rep_mat, columns=columns4G)
    # rep_mat4G.to_csv(f"rep mat for G NN={NN_list[0]}.csv", index=False)
