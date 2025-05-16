import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题

def compare_S_params(Smodel_params0, Smodel_params_hat, GG):
    ag0 = np.array(Smodel_params0["ag"])
    zeta1g0 = np.array(Smodel_params0["zeta1g"])
    zeta2g0 = np.array(Smodel_params0["zeta2g"])
    D1g0 = np.array(Smodel_params0["D1g"])
    D2g0 = np.array(Smodel_params0["D2g"])
    Phi1g0 = np.array(Smodel_params0["Phi1g"])
    Phi2g0 = np.array(Smodel_params0["Phi2g"])
    coeffs0 = np.concatenate((D1g0, ag0, Phi1g0, D2g0, zeta1g0, zeta2g0, Phi2g0), axis=-1)

    ag_hat = np.array(Smodel_params_hat["ag"])
    zeta1g_hat = np.array(Smodel_params_hat["zeta1g"])
    zeta2g_hat = np.array(Smodel_params_hat["zeta2g"])
    D1g_hat = np.array(Smodel_params_hat["D1g"])
    D2g_hat = np.array(Smodel_params_hat["D2g"])
    Phi1g_hat = np.array(Smodel_params_hat["Phi1g"])
    Phi2g_hat = np.array(Smodel_params_hat["Phi2g"])
    coeffs_hat = np.concatenate((D1g_hat, ag_hat, Phi1g_hat, D2g_hat, zeta1g_hat, zeta2g_hat, Phi2g_hat), axis=-1)

    NN, TT, PP, QQ1 = coeffs0.shape
    coeff_name = ["D1g", "ag", "Phi1g", "D2g", "zeta1g", "zeta2g", "Phi2g"]

    for pp_index in range(PP):
        for qq_index in range(QQ1):
            plt.figure(figsize=(12, 8))
            for gg in range(GG):
                coeff_hat_i = coeffs_hat[gg, :, pp_index, qq_index]
                real_coeff_i = coeffs0[gg, :, pp_index, qq_index]
                color = plt.cm.rainbow(gg / GG)
                plt.plot(range(TT), coeff_hat_i, label=f"Group {gg} hat", linestyle="--", color=color)
                plt.plot(range(TT), real_coeff_i, label=f"Group {gg} real", linestyle="-", color=color)


            plt.legend()
            plt.title(f"S model Estimated Coefficients v.s. Real coefficients for pp={pp_index}, "+coeff_name[qq_index])
            plt.xlabel("Time")
            plt.ylabel("Coefficients")
            plt.savefig(f"plots//S model Estimated Coefficients v.s. Real coefficients for pp={pp_index}, "+coeff_name[qq_index]+".png")
            # # plt.show()
            plt.close()

def compare_Y_params(Ymodel_params0, Ymodel_params_hat, GG):
    bg0 = np.array(Ymodel_params0["bg"])
    gamma1g0 = np.array(Ymodel_params0["gamma1g"])
    gamma2g0 = np.array(Ymodel_params0["gamma2g"])
    betag0 = np.array(Ymodel_params0["betag"])

    coeffs0 = np.concatenate((bg0, betag0, gamma1g0, gamma2g0), axis=-1)

    bg_hat = np.array(Ymodel_params_hat["bg"])
    gamma1g_hat = np.array(Ymodel_params_hat["gamma1g"])
    gamma2g_hat = np.array(Ymodel_params_hat["gamma2g"])
    betag_hat = np.array(Ymodel_params_hat["betag"])

    coeffs_hat = np.concatenate((bg_hat, betag_hat, gamma1g_hat, gamma2g_hat), axis=-1)

    NN, TT, _, PP3 = coeffs0.shape
    coeff_name = ["b1g", "betag", "gamma1g", "gamma2g"]

    for pp_index in range(PP3):
        plt.figure(figsize=(12, 8))
        for gg in range(GG):
            coeff_hat_i = coeffs_hat[gg, :, :, pp_index]
            real_coeff_i = coeffs0[gg, :, :, pp_index]
            color = plt.cm.rainbow(gg / GG)
            plt.plot(range(TT), coeff_hat_i, label=f"Group {gg} hat", linestyle="--", color=color)
            plt.plot(range(TT), real_coeff_i, label=f"Group {gg} real", linestyle="-", color=color)


        plt.legend()
        plt.title(f"Y model Estimated Coefficients v.s. Real coefficients for "+coeff_name[pp_index])
        plt.xlabel("Time")
        plt.ylabel("Coefficients")
        plt.savefig(f"plots//Y model Estimated Coefficients v.s. Real coefficients for "+coeff_name[pp_index]+".png")
        # # plt.show()
        plt.close()

def centralize(data):
    return data-np.mean(data)

def policy_boxplot(labels, all_histories):
    # 准备数据框
    data = []
    for label, history in zip(labels, all_histories):
        for value in history:
            data.append({'Policy': label, 'Value Function': value[0]})

    df = pd.DataFrame(data)

    # 画箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Policy', y='Value Function', data=df)
    plt.title("Monte Carlo Estimated Value Function Distribution per Policy")
    plt.ylabel("Estimated Value Function")
    plt.xlabel("Policy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots//Policy_comparison_boxplot.png")
    # plt.show()
    plt.close()


def barplot(results_df, filename="plots//Policy_comparison_barplot.png", estimated_only=False):
    x = np.arange(len(results_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    if estimated_only:
        ax.bar(
            x, results_df["Estimated Mean"], width, yerr=results_df["Estimated Std"],
            label='Estimated Model', capsize=5, color='#FF7F0E'  # 橙色
        )
    else:
        ax.bar(
            x - width / 2, results_df["True Mean"], width, yerr=results_df["True Std"],
            label='True Model', capsize=5, color='#A6CEE3'  # 浅蓝
        )
        ax.bar(
            x + width / 2, results_df["Estimated Mean"], width, yerr=results_df["Estimated Std"],
            label='Estimated Model', capsize=5, color='#FDBF6F'  # 浅红
        )

    ax.set_xlabel("Policy", fontsize=15)
    ax.set_ylabel("Mean outcome", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Pi"], fontsize=20)
    plt.setp(ax.get_xticklabels(), ha="right")  # 设置倾斜角度和对齐方式
    ax.tick_params(axis='y', labelsize=20)

    ax.legend(fontsize=20)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def decode_res_dict(res_dict4S, res_dict4Y):
    Gvec = np.array(res_dict4S["Gvec"])
    coeff_hat4S = np.array(res_dict4S["coeff_hat4S"])
    coeff_hat4Y = np.array(res_dict4Y["coeff_hat4Y"])

    # coeff_hat4S GG TT PP Q1 1
    GG, TT, PP, Q1 = coeff_hat4S.shape

    D1g = coeff_hat4S[:, :, :, 0:1]
    ag = coeff_hat4S[:, :, :, 1:2]
    Phi1g = coeff_hat4S[:, :, :, 2:PP + 2]
    D2g = coeff_hat4S[:, :, :, PP + 2:PP + 3]
    zeta1g = coeff_hat4S[:, :, :, PP + 3:PP + 4]
    zeta2g = coeff_hat4S[:, :, :, PP + 4:PP + 5]
    Phi2g = coeff_hat4S[:, :, :, PP + 5:]

    Xi = np.concatenate((ag, Phi1g,
                         D2g, zeta1g, zeta2g, Phi2g), axis=-1)

    Smodel_params = {"ag": ag, "zeta1g": zeta1g, "zeta2g": zeta2g,
                     "D1g": D1g, "D2g": D2g, "Phi1g": Phi1g, "Phi2g": Phi2g,
                     "Xig": Xi}

    # coeff_hat4Y GG TT QQ 1
    bg = coeff_hat4Y[:, :, 0:1, :]
    betag = coeff_hat4Y[:, :, 1:PP + 1, :]
    gamma1g = coeff_hat4Y[:, :, PP + 1:PP + 2, :]
    gamma2g = coeff_hat4Y[:, :, PP + 2:PP + 3, :]

    Ymodel_params = {"bg": bg, "gamma1g": gamma1g, "gamma2g": gamma2g, "betag": betag}

    network_params = {"Gvec": Gvec}

    return Smodel_params, Ymodel_params, network_params