import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from Environment import Environment
from RL_utils import compare_S_params, compare_Y_params, policy_boxplot, barplot, decode_res_dict

def monte_carlo_estimate(pi, GG, NN, TT, KK, PP, W, network_params, Smodel_params, Ymodel_params, num_runs, eps_sigma4S, eps_sigma4Y, mode=""):
    rewards = np.zeros((num_runs, 1))
    for run in tqdm(range(num_runs), desc="Monte Carlo Runs"):

        S0tensor = np.random.normal(loc=0, scale=eps_sigma4S, size=(NN, PP, 1))
        Ztensor = np.random.normal(loc=0, scale=1, size=(NN, TT, KK, 1))
        env = Environment(GG=GG, NN=NN, TT=TT, KK=KK, PP=PP, W=W,
                          network_params=network_params,
                          Smodel_params=Smodel_params,
                          Ymodel_params=Ymodel_params,
                          Ztensor=Ztensor,
                          S0tensor=S0tensor,
                          eps_sigma4S=eps_sigma4S,
                          eps_sigma4Y=eps_sigma4Y)
        for tt in range(TT):
            obs = env.observe()
            S_current = obs["S_current"]
            S_past = obs["S_past"]
            A = pi(S_current, S_past, W)
            env.interact(A)

        Stensor, Ytensor = env.dump()
        total_reward = np.sum(Ytensor, axis=(0, 1, 2))
        rewards[run, 0] = total_reward / (NN * TT)

        if run == 0:
            env.plot_time_series(folder="plots//", mode=mode)

    return rewards

def pi0(S_current, S_past, Wmat):
    NN, PP, _ = S_current.shape
    A = np.zeros((NN, 1, 1))
    return A

def pi1(S_current, S_past, Wmat):
    NN, PP, _ = S_current.shape
    A = np.ones((NN, 1, 1))
    return A

def pirandom(S_current, S_past, Wmat):
    NN, PP, _ = S_current.shape
    A = np.random.binomial(1, 0.5, size=(NN, 1, 1))
    return A

def pithreshold(S_current, S_past, Wmat, threshold=0.5):
    # 假设 S_current 的最后一个变量代表风险评分
    score = S_current[:, 0, 0]  # 取第一个变量
    A = (score > threshold).astype(int).reshape(-1, 1, 1)
    return A

def pidelta(S_current, S_past, Wmat, delta=0.1):
    diff = S_current - S_past
    score = diff[:, 0, 0]
    A = (score > delta).astype(int).reshape(-1, 1, 1)
    return A

def pimix(S_current, S_past, Wmat, threshold=0.5, prob_if_low=0.2):
    score = S_current[:, 0, 0]
    A = np.zeros((len(score), 1, 1))
    for i in range(len(score)):
        if score[i] > threshold:
            A[i, 0, 0] = 1
        else:
            A[i, 0, 0] = np.random.binomial(1, prob_if_low)
    return A


def pineighbor_threshold(S_current, S_past, Wmat, threshold=0.2):
    N, P, _ = S_current.shape
    S_bar = (Wmat @ S_current[:, :, 0]).reshape(N, P, 1)
    score = S_bar[:, 0, 0]  # 取第一个变量
    A = (score > threshold).astype(int).reshape(N, 1, 1)
    return A


def pidiff_neighbor(S_current, S_past, Wmat, delta=0.2):
    N, P, _ = S_current.shape
    S_bar = (Wmat @ S_current[:, :, 0]).reshape(N, P, 1)
    diff = S_current - S_bar
    score = diff[:, 0, 0]  # 只取第一个变量
    A = (score > delta).astype(int).reshape(N, 1, 1)
    return A

# 指定 JSON 文件所在的文件夹路径
json_folder = "json"

# 获取文件夹中所有 JSON 文件的路径
json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith(".json")]

num_runs = 300
eps_sigma4S = eps_sigma4Y = 1
pi_list = [pi0, pi1, pirandom, pithreshold, pimix, pineighbor_threshold, pidiff_neighbor]
# pi_list = [pi0, pi1, pirandom, pithreshold, pimix]

json_count = 0
# 读取每个 JSON 文件
for json_file in json_files:
    for enc in ['utf-8', 'latin1', 'gbk', 'utf-16']:
        try:
            with open(json_file, "r", encoding=enc) as f:
                data = json.load(f)
            print(f"使用编码 {enc} 读取成功")
            break
        except Exception as e:
            print(f"尝试编码 {enc} 失败: {e}")

    # 初始化一个空的 DataFrame，用于存储每次遍历的信息
    results_df = pd.DataFrame(columns=["Pi", "NN", "TT", "PP", "KK",
                                       "True Mean", "True Std",
                                       "Estimated Mean", "Estimated Std"])

    network_params0 = data["network_params"]
    Smodel_params0 = data["Smodel_params"]
    Ymodel_params0 = data["Ymodel_params"]
    res_dict4S = data["res_dict4S"]
    res_dict4Y = data["res_dict4Y"]

    NN = data["NN"]
    TT = data["TT"]
    PP = data["PP"]
    KK = data["KK"]
    W = data["W"]
    GG = 3

    all_histories = []  # 每个元素是一个策略对应的 MC 历史列表
    labels = []  # 每个历史对应的策略标签

    for i in range(len(pi_list)):
        pi = pi_list[i]
        true_mc_history = monte_carlo_estimate(pi=pi, GG=GG, NN=NN, TT=TT, KK=KK, PP=PP, W=W,
                                               network_params=network_params0,
                                               Smodel_params=Smodel_params0,
                                               Ymodel_params=Ymodel_params0,
                                               num_runs=num_runs, eps_sigma4S=eps_sigma4S, eps_sigma4Y=eps_sigma4Y, mode="true")

        true_value_function = sum(true_mc_history)/len(true_mc_history)
        print(f"The true value function is {true_value_function}.")

        all_histories.append(true_mc_history)
        labels.append(f"Policy {i + 1} True")  # 或者用 pi 本身命名

        Smodel_params_hat, Ymodel_params_hat, network_params_hat = decode_res_dict(res_dict4S, res_dict4Y)

        estimated_mc_history = monte_carlo_estimate(pi=pi, GG=GG, NN=NN, TT=TT, KK=KK, PP=PP, W=W,
                                                    network_params=network_params_hat,
                                                    Smodel_params=Smodel_params_hat,
                                                    Ymodel_params=Ymodel_params_hat,
                                                    num_runs=num_runs, eps_sigma4S=eps_sigma4S, eps_sigma4Y=eps_sigma4Y, mode="simu")

        estimated_value_function = sum(estimated_mc_history)/len(estimated_mc_history)
        print(f"The estimated value function is {estimated_value_function}.")

        all_histories.append(estimated_mc_history)
        labels.append(f"Policy {i + 1} Estimated")  # 或者用 pi 本身命名

        # 将当前遍历的结果记录到临时 DataFrame 中
        temp_df = pd.DataFrame({
            "Pi": [i],
            "NN": [NN], "TT": [TT], "PP": [PP], "KK": [KK],
            "True Mean": np.mean(true_value_function), "True Std": np.std(true_mc_history),
            "Estimated Mean": np.mean(estimated_mc_history), "Estimated Std": np.std(estimated_mc_history)
        })

        # 使用 pd.concat() 来合并 DataFrame
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

    labels = [labels[i] for i in(0, 2, 1, 3)]
    all_histories = [all_histories[i] for i in(0, 2, 1, 3)]
    policy_boxplot(labels=labels, all_histories=all_histories)
    barplot(results_df, filename=f"plots//Policy_comparison_barplot_json{json_count}.png")
    results_df.to_excel(f"table//mc_comparison_results_json{json_count}.xlsx", index=False)
    json_count += 1

    # compare_S_params(Smodel_params0=Smodel_params0, Smodel_params_hat=Smodel_params_hat, GG=GG)
    # compare_Y_params(Ymodel_params0=Ymodel_params0, Ymodel_params_hat=Ymodel_params_hat, GG=GG)


