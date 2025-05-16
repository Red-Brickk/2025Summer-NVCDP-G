import numpy as np
import pandas as pd
from DGP import DGP, SieveBasis
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def plot_time_series(tensor):
    NN, TT, PP, _ = np.shape

    # 去掉最后一维 -> (NN, TT, PP)
    data = tensor.squeeze(-1)

    # 指定要画的维度
    p = 1  # 选第 1 个向量维度（下标从 0 开始）

    plt.figure(figsize=(10, 6))

    for n in range(NN):  # 遍历所有个体
        plt.plot(data[n, :, p], label=f'个体 {n + 1}')

    plt.xlabel('时间点')
    plt.ylabel(f'维度 {p + 1} 的值')
    plt.title(f'维度 {p + 1} 在不同个体间的变化趋势')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_grouping(Gvec_hat, Gvec_real, title="Comparison of Coefficients"):
    cm = confusion_matrix(Gvec_real, Gvec_hat)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Estimated Groups")
    plt.ylabel("True Groups")
    plt.title(title + f'confusion matrix heatmap')

    filename = f"{title.replace(' ', '_')} confusion matrix heatmap.png"
    plt.savefig(filename, format='png')
    # # plt.show()
    plt.close()
    print(f"Figure saved as {filename}")

def EstMeasure(coeff_hat, real_coeff):
    """
    :param coeff_hat: GG, TT, PP, QQ1
    :param real_coeff: GG, TT, PP, QQ1
    :return:
    """

    GG, _, _, _ = coeff_hat.shape

    # 计算所有组之间的 L2 误差
    cost_matrix = np.zeros((GG, GG))
    for i in range(GG):
        for j in range(GG):
            cost_matrix[i, j] = np.sqrt(np.mean((coeff_hat[i] - real_coeff[j]) ** 2))

    # 使用匈牙利算法寻找最佳匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 重新排列 coeff_hat，使其与 real_coeff 对应
    coeff_hat = coeff_hat[row_ind]
    real_coeff = real_coeff[col_ind]

    # coeff_hat GG TT PP QQ 1 （对 S 模型）
    # coeff_hat GG TT PP 1 （对 Y 模型）
    l2_norm = np.sqrt(np.mean((coeff_hat - real_coeff) ** 2))
    l1_norm = np.max(np.abs(coeff_hat - real_coeff))
    l2_norm_by_group = np.zeros((GG, 1))
    l1_norm_by_group = np.zeros((GG, 1))

    for gg in range(GG):
        l2_norm_by_group[gg] = np.sqrt(np.mean((coeff_hat[gg] - real_coeff[gg]) ** 2))
        l1_norm_by_group[gg] = np.max(np.abs(coeff_hat[gg] - real_coeff[gg]))

    return {"l2_norm": l2_norm,
            "l1_norm": l1_norm,
            "l2_norm_by_group": l2_norm_by_group,
            "l1_norm_by_group": l1_norm_by_group}

def evaluate_grouping(Gvec_hat, Gvec_real, title="Comparison of Coefficients"):
    Gvec_hat = Gvec_hat.flatten()
    Gvec_real = Gvec_real.flatten()
    ari_score = adjusted_rand_score(Gvec_real, Gvec_hat)
    nmi_score = normalized_mutual_info_score(Gvec_real, Gvec_hat)

    # 返回结果
    results = {
        'ARI': ari_score,
        'NMI': nmi_score
    }

    return results


def fill_in_rep_mat(res_dict, dgp, mode):
    if mode == "S":
        real_coeff = dgp.Xi
        theta_hat = res_dict["Theta_hat4S"]
        coeff_hat = res_dict["coeff_hat4S"]
    elif mode == "Y":
        real_coeff = np.concatenate((dgp.bg, dgp.betag, dgp.gamma1g, dgp.gamma2g), axis=-2)
        theta_hat = res_dict["Theta_hat4Y"]
        coeff_hat = res_dict["coeff_hat4Y"]
    else:
        raise ValueError("Please give correct mode.")

    Gvec = res_dict["Gvec"]
    coeff_hat_by_ind = coeff_hat[Gvec]
    Gvec_real = dgp.Gvec

    real_coeff_by_ind = real_coeff[Gvec_real]

    ############
    diff_dict_by_ind = EstMeasure(coeff_hat=coeff_hat_by_ind, real_coeff=real_coeff_by_ind)
    l2_norm_by_ind = diff_dict_by_ind["l2_norm"]
    l1_norm_by_ind = diff_dict_by_ind["l1_norm"]
    ############

    res_dict2 = evaluate_grouping(Gvec_hat=Gvec, Gvec_real=Gvec_real)
    ARI = res_dict2["ARI"]
    NMI = res_dict2["NMI"]

    return ARI, NMI, l2_norm_by_ind, l1_norm_by_ind


def fill_in_res_mat(rep_res_mat):
    fill_in = [0 for i in range(3 * len(rep_res_mat[0]))]
    fill_in[0:(len(rep_res_mat[0]))] = np.mean(rep_res_mat, axis=0)
    fill_in[(len(rep_res_mat[0])):(2 * len(rep_res_mat[0]))] = np.sqrt(
        np.var(rep_res_mat, axis=0))
    fill_in[(2 * len(rep_res_mat[0])):(3 * len(rep_res_mat[0]))] = np.sqrt(np.mean(rep_res_mat**2, axis=0))

    return fill_in


def plot_coefficients(res_dict, dgp, mode, plot_true=True, plot_hat=True):
    if mode == "S":
        coeff_hat = res_dict["coeff_hat4S"]
    if mode == "Y":
        coeff_hat = res_dict["coeff_hat4Y"]
    # coeff_hat_by_ind = res_dict["coeff_hat_by_ind"] # NN TT PP QQ+1 1
    if plot_hat:
        Gvec_hat = res_dict["Gvec"]
    if plot_true:
        Gvec_real = dgp.Gvec

    color_list = [
        '#4FC3F7',  # 亮蓝色
        '#FFB74D',  # 亮橙色
        '#BA68C8',  # 亮紫色
        '#81C784',  # 亮绿色
        '#FFD54F',  # 亮黄色
        '#64B5F6',  # 湖蓝色
        '#E57373',  # 粉红红
        '#A1887F',  # 浅棕灰
    ]

    if mode == "S":
        GG, TT, PP, QQ1 = coeff_hat.shape
        if plot_true:
            real_coeff = dgp.Xi
        # real_coeff_by_ind = real_coeff[Gvec_real]  # NN TT PP QQ+1 1

        for pp_index in range(PP):
            for qq_index in range(QQ1):
                plt.figure(figsize=(12, 8))
                for gg in range(GG):
                    color = color_list[gg]
                    if plot_hat:
                        coeff_hat_i = coeff_hat[gg, :, pp_index, qq_index]
                        plt.plot(range(TT), coeff_hat_i, label=f"Group {gg+1} hat", linestyle="--", color=color)
                    if plot_true:
                        real_coeff_i = real_coeff[gg, :, pp_index, qq_index]
                        plt.plot(range(TT), real_coeff_i, label=f"Group {gg+1} real", linestyle="-", color=color)

                plt.legend()
                plt.title(f"S model Estimated Coefficients v.s. Real coefficients for pp={pp_index}, qq={qq_index}.")
                plt.xlabel("Time")
                plt.ylabel("Coefficients")
                plt.savefig(
                    f"plots//S model Estimated Coefficients v.s. Real coefficients for pp={pp_index}, qq={qq_index}.")
                # # plt.show()
                plt.close()

    elif mode == "Y":
        GG, TT, PP3, _ = coeff_hat.shape
        if plot_true:
            real_coeff = np.concatenate((dgp.bg, dgp.betag, dgp.gamma1g, dgp.gamma2g), axis=-2)
            real_coeff_by_ind = real_coeff[Gvec_real]  # NN TT PP QQ+1 1

        for pp_index in range(PP3):
            plt.figure(figsize=(12, 8))
            for gg in range(GG):
                color = color_list[gg]
                if plot_hat:
                    coeff_hat_i = coeff_hat[gg, :, pp_index, 0]
                    plt.plot(range(TT), coeff_hat_i, label=f"Group {gg+1} hat", linestyle="--", color=color)
                if plot_true:
                    real_coeff_i = real_coeff[gg, :, pp_index, 0]
                    plt.plot(range(TT), real_coeff_i, label=f"Group {gg+1} real", linestyle="-", color=color)

            plt.legend()
            plt.title(f"Y model Estimated Coefficients v.s. Real coefficients for pp={pp_index}.")
            plt.xlabel("Time")
            plt.ylabel("Coefficients")
            plt.savefig(f"plots//Y model Estimated Coefficients v.s. Real coefficients for pp={pp_index}.")
            # # plt.show()
            plt.close()

    else:
        raise ValueError("Please give correct mode.")


def plot_time_series(Tensor, Gvec, mode=None):
    if mode is None:
        raise ValueError("Please specify the plotting mode.")
    NN = Tensor.shape[0]
    if Gvec is None:
        Gvec = np.zeros((NN, 1))
    line_styles = ['-', '--', '-.', ':']

    # 画状态曲线
    if mode == "S":
        Stensor = Tensor
        NN, TT, PP, _ = Stensor.shape
        for pp_index in range(PP):
            fig, ax = plt.subplots(figsize=(12, 6))
            data_state = Stensor[:, :, pp_index, 0]
            for i in range(NN):
                ax.plot(range(TT), data_state[i, :], alpha=0.6, linewidth=1.2,
                        linestyle=line_styles[Gvec[i] - 1],
                        label=f"State {i + 1}" if i < 5 else "_nolegend_")

            ax.set_title(f"State Curves (pp_index={pp_index})")
            ax.set_ylabel("State Value")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)

            plt.tight_layout()
            plt.savefig(f"plots//state_curves_pp_{pp_index}.png")
            # # plt.show()
            plt.close()

    # 画外部变量曲线
    if mode == "Z":
        Ztensor = Tensor
        NN, TT, KK, _ = Ztensor.shape
        for kk_index in range(KK):
            fig, ax = plt.subplots(figsize=(12, 6))
            data_EV = Ztensor[:, :, kk_index, 0]
            for i in range(NN):
                ax.plot(range(TT), data_EV[i, :], alpha=0.6, linewidth=1.2,
                        linestyle=line_styles[Gvec[i] - 1],
                        label=f"State {i + 1}" if i < 5 else "_nolegend_")

            ax.set_title(f"Exogenous Variable Curves (kk_index={kk_index})")
            ax.set_ylabel("Exogenous Variable Value")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)

            plt.tight_layout()
            plt.savefig(f"plots//exogenous_variable_curves_kk_{kk_index}.png")
            # plt.show()
            plt.close()

    # 画奖励曲线
    if mode == "Y":
        Ytensor = Tensor
        NN, TT, PP, _ = Ytensor.shape
        fig, ax = plt.subplots(figsize=(12, 6))
        data_reward = Ytensor[:, :, 0]
        for i in range(NN):
            ax.plot(range(TT), data_reward[i, :], alpha=0.6, linewidth=1.2,
                    linestyle=line_styles[Gvec[i] - 1],
                    label=f"Reward {i + 1}" if i < 5 else "_nolegend_")

        ax.set_title("Reward Curves")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Reward Value")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)

        plt.tight_layout()
        plt.savefig(f"plots//reward_curves.png")
        # plt.show()
        plt.close()


def plot_residuals(Tensor, Tensor_fit, mode=None):
    if mode == "S":
        NN, TT, PP, _ = Tensor.shape
        residualTensor = Tensor - Tensor_fit
        colors = plt.cm.tab10(np.linspace(0, 1, NN))

        for pp in range(PP):
            plt.figure(figsize=(10, 6))  # 创建一个新的图形窗口
            for nn in range(NN):  # 遍历 NN 维度
                # 提取第 nn 个个体在第 pp 个特征上的残差随时间变化的曲线
                residuals = residualTensor[nn, :, pp, 0]
                # plt.plot(range(TT), residuals, label=f'Individual {nn}')

                # 提取第 nn 个个体在第 pp 个特征上的真实值和拟合值
                real_values = Tensor[nn, :, pp, 0]
                fit_values = Tensor_fit[nn, :, pp, 0]

                # 画出真实值和拟合值
                color = colors[nn]
                plt.plot(range(TT), real_values, label=f'Real Values Individual {nn}', linestyle='-', color=color)
                plt.plot(range(TT), fit_values, label=f'Fit Values Individual {nn}', linestyle='--', color=color)

            # plt.title(f'Residuals over Time for Feature {pp}')
            plt.title(f'Real v.s. Fitted values over Time for Feature {pp}')
            plt.xlabel('Time (TT)')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plots//Residuals over Time for Feature {pp}.png")
            # # plt.show()
            plt.close()

            # 画 Q-Q 图（散点图）
            plt.figure(figsize=(6, 6))
            plt.scatter(Tensor[:, :, pp, 0].flatten(), Tensor_fit[:, :, pp, 0].flatten(), alpha=0.5)
            min_val = min(Tensor[:, :, pp, 0].min(), Tensor_fit[:, :, pp, 0].min())
            max_val = max(Tensor[:, :, pp, 0].max(), Tensor_fit[:, :, pp, 0].max())
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)  # 添加 y=x 参考线
            plt.xlabel('Actual Values')
            plt.ylabel('Fitted Values')
            plt.title(f'QQ Plot for Fit {mode} pp={pp}')
            plt.grid(True)
            plt.savefig(f"plots//QQ_Plot_for_Fit_{mode}_pp={pp}.png")
            plt.close()

    elif mode == "Y":
        NN, TT, _ = Tensor.shape
        residualTensor = Tensor - Tensor_fit
        colors = plt.cm.tab10(np.linspace(0, 1, NN))
        plt.figure(figsize=(10, 6))  # 创建一个新的图形窗口
        for nn in range(NN):  # 遍历 NN 维度
            # 提取第 nn 个个体在第 pp 个特征上的残差随时间变化的曲线
            residuals = residualTensor[nn, :, 0]

            # 提取第 nn 个个体在第 pp 个特征上的真实值和拟合值
            real_values = Tensor[nn, :, 0]
            fit_values = Tensor_fit[nn, :, 0]

            # 画出真实值和拟合值
            color = colors[nn]
            plt.plot(range(TT), real_values, label=f'Real Values Individual {nn}', linestyle='-', color=color)
            plt.plot(range(TT), fit_values, label=f'Fit Values Individual {nn}', linestyle='--', color=color)

        # plt.title(f'Residuals over Time Fit Y')
        plt.title(f'Real v.s. Fitted values over Time of Y')
        plt.xlabel('Time (TT)')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots//Residuals over Time Fit Y.png")
        # # plt.show()
        plt.close()
        # 画 Q-Q 图（散点图）
        # 计算实际值和拟合值的一维向量
        actual = Tensor.flatten()
        fitted = Tensor_fit.flatten()

        # 计算百分位数范围（例如只看中间 98%）
        lower_q = 0
        upper_q = 99.9

        x_min, x_max = np.percentile(actual, [lower_q, upper_q])
        y_min, y_max = np.percentile(fitted, [lower_q, upper_q])

        plt.figure(figsize=(6, 6))
        plt.scatter(actual, fitted, alpha=0.5)
        plt.plot([x_min, x_max], [x_min, x_max], color='red', linestyle='--', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Fitted Values')
        plt.title(f'QQ Plot for Fit {mode}')
        plt.grid(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(f"plots//QQ_Plot_for_Fit_{mode}.png")
        plt.close()
    else:
        raise ValueError




def regularize1dim(Tensor):
    mean = Tensor.mean()
    std = Tensor.std()
    return (Tensor - mean) / (std + 1e-3)


if __name__ == '__main__':
    print("Hello Brick!")
