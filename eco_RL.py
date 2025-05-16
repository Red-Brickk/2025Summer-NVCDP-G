import json
import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from Environment import Environment
from RL_utils import compare_S_params, compare_Y_params, policy_boxplot, barplot, decode_res_dict


def compare_policies_ttest(all_histories, Pi_names):
    from scipy.stats import ttest_ind

    n = len(all_histories)
    results = []

    for i in range(n):
        for j in range(i + 1, n):
            t_stat, p_val = ttest_ind(all_histories[i], all_histories[j], equal_var=False)
            results.append({
                "Policy A": Pi_names[i],
                "Policy B": Pi_names[j],
                "t-stat": t_stat,
                "p-value": p_val,
                "Significant": p_val < 0.05
            })
    return pd.DataFrame(results)

def monte_carlo_estimate(pi, GG, NN, TT, KK, PP, W, network_params,
                         Smodel_params, Ymodel_params, Ztensor, S0tensor,
                         num_runs, eps_sigma4S, eps_sigma4Y, mode="", eco=False):
    rewards = np.zeros((num_runs, 1))
    for run in tqdm(range(num_runs), desc="Monte Carlo Runs"):
        env = Environment(GG=GG, NN=NN, TT=TT, KK=KK, PP=PP, W=W,
                          network_params=network_params,
                          Smodel_params=Smodel_params,
                          Ymodel_params=Ymodel_params,
                          Ztensor = Ztensor,S0tensor=S0tensor,
                          eps_sigma4S=eps_sigma4S, eps_sigma4Y=eps_sigma4Y)
        A_prev = None
        if eco:
            A_prev = np.zeros((NN, 1, 1)) + 0.1
        for tt in range(TT):
            obs = env.observe()
            S_current = obs["S_current"]
            S_past = obs["S_past"]
            if eco:
                A = pi(S_current, S_past, Wmat=W, A_prev=A_prev, t=tt)
                A_prev = A.copy()
            else:
                A = pi(S_current, S_past)
            env.interact(A)

        Stensor, Ytensor = env.dump()
        total_reward = np.sum(Ytensor, axis=(0, 1, 2))
        rewards[run, 0] = total_reward / (NN * TT)

        if run == 0:
            env.plot_time_series(folder="plots//", mode=mode)

    return rewards

def pi_const(S_current, S_past, A_prev, Wmat, A_base1=0.5, t=0):
    """策略 1：最低工资不变"""
    NN, PP, _ = S_current.shape
    A = np.full((NN, 1, 1), A_base1)
    return A

def pi_time_linear(S_current, S_past, A_prev, Wmat, A_base1=0.4, gamma=0.002, t=0):
    """策略 2：随时间增长"""
    NN, PP, _ = S_current.shape
    A = np.full((NN, 1, 1), A_base1 + gamma * t)
    return A


def pi_gdp_change(S_current, S_past, Wmat, A_prev=None, alpha=0.05, t=0):
    """策略 4：根据 GDP 增速变化调整"""
    gdp_diff = S_current[:,0:1,:] - S_past[:,0:1,:]
    if A_prev is None:
        A_prev = np.ones_like(gdp_diff) + 0.4  # 默认上期相对最低工资为 0.4
    A = A_prev + alpha * gdp_diff
    return A.reshape(-1, 1, 1)

def pi_gdp_threshold(S_current, S_past, Wmat, A_prev=None, threshold=0.03, delta=0.01, t=0):
    """策略 5：GDP 增速超过阈值才调整"""
    gdp = S_current[:,0:1,:]
    if A_prev is None:
        A_prev = np.ones_like(gdp)
    A = np.where(gdp > threshold, A_prev + delta, A_prev*0.99)
    return A.reshape(-1, 1, 1)



def pi_gdp_neighbor(S_current, S_past, A_prev=None, Wmat=None, beta_self=0.6, beta_neighbor=0.4, alpha=0.05, t=0):
    """策略 6：结合本州与邻州 GDP 增速决定最低工资调整"""
    NN, PP, _ = S_current.shape
    gdp_self = S_current[:, 0, 0]  # [N]

    # 计算邻居 GDP 加权平均
    if Wmat is None:
        raise ValueError("需要提供 Wmat")
    S_bar = Wmat @ gdp_self  # [N]

    # 综合 GDP 增速
    gdp_combined = beta_self * gdp_self + beta_neighbor * S_bar  # [N]

    # 初始化 A_prev
    if A_prev is None:
        A_prev = np.full((NN,), 0.4)

    A = A_prev + alpha * gdp_combined
    return A.reshape(-1, 1, 1)


def pi_gdp_catchup(S_current, S_past, A_prev=None, Wmat=None, delta=0.01, t=0):
    """策略 7：如果邻州 GDP 增速高于本州，最低工资小幅上调"""
    NN, PP, _ = S_current.shape
    gdp_self = S_current[:, 0, 0]  # [N]

    if Wmat is None:
        raise ValueError("需要提供 Wmat")
    S_bar = Wmat @ gdp_self  # [N]

    if A_prev is None:
        A_prev = np.full((NN,), 0.4)

    A = np.where(S_bar > gdp_self, A_prev + delta, A_prev)
    return A.reshape(-1, 1, 1)



def optimize_policy_parameters(
    policy_func,                     # 策略函数，如 pi_time_linear
    param_grid: dict,                # 参数网格，如 {"gamma": [0.0, 0.05, 0.1]}
    Ztensor,S0tensor,
    GG=3, NN=100, TT=20, KK=5, PP=3, W=None,
    network_params=None,
    Smodel_params=None,
    Ymodel_params=None,
    num_runs=10,
    eps_sigma4S=0.01,
    eps_sigma4Y=0.01,
):
    best_reward = -np.inf
    best_params = None
    all_results = []

    # 构建参数组合的笛卡尔积
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param_dict in param_combinations:
        # 构建一个封装函数（固定参数）
        def pi(S_current, S_past, A_prev=None, t=0):

            return policy_func(S_current, S_past, A_prev=A_prev, t=t, **param_dict)

        # Monte Carlo 评估该参数组合
        rewards = monte_carlo_estimate(pi, GG=GG, NN=NN, TT=TT, KK=KK, PP=PP, W=W,
                                       network_params=network_params,
                                       Smodel_params=Smodel_params,
                                       Ymodel_params=Ymodel_params,
                                       Ztensor=Ztensor,
                                       S0tensor=S0tensor,
                                       num_runs=num_runs,
                                       eps_sigma4S=eps_sigma4S,
                                       eps_sigma4Y=eps_sigma4Y,
                                       mode="", eco=True)
        avg_reward = np.mean(rewards)
        all_results.append((param_dict, avg_reward))

        print(f"参数 {param_dict} → 平均 reward = {avg_reward:.4f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = param_dict

    print(f"\n✅ 最优参数: {best_params}，对应平均 reward = {best_reward:.4f}")
    return best_params, best_reward, all_results


# extract data
data = np.load('economy_data.npz')

ur_tensor = data['UR']
gdp_growth_tensor = data['GDP_growth'][:, :, np.newaxis, np.newaxis]
cpi_tensor = data['CPI'][:, :, :, np.newaxis]
hpi_tensor = data['HPI'][:, :, :, np.newaxis]
pcpi_tensor = data['PCPI'][:, :, :, np.newaxis]
ipi_tensor = data['IPI'][:, :, :, np.newaxis]
population_growth_tensor = data["population_growth"][:, :, :, np.newaxis]
Action_tensor = data['Action'][:, :, :, np.newaxis]
W_mat = data['W']
state_names = data['state_names']


# NN TT PP 1

Stensor = np.concatenate((gdp_growth_tensor, cpi_tensor), axis=2)
Ztensor = np.concatenate((pcpi_tensor, ipi_tensor, hpi_tensor, population_growth_tensor), axis=2)

eps_sigma4S = np.var(Stensor[:,:,0,:])*5
eps_sigma4Y = np.var(ur_tensor)*5

# 指定 JSON 文件所在的文件夹路径
json_folder = "json"

# 获取文件夹中所有 JSON 文件的路径
json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith(".json")]

num_runs = 100
pi_list = [pi_const, pi_time_linear, pi_gdp_change, pi_gdp_threshold, pi_gdp_neighbor, pi_gdp_catchup]
# pi_list = [pi0, pi1, pirandom, pithreshold, pimix]
Pi_names = ["恒定", "随时间增长", "随GDP增速变化调整", "GDP增速超过阈值才调整"]

json_count = 0
# 读取每个 JSON 文件
json_file = "json//Eco_result.json"
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

res_dict4S = data["res_dict4S"]
res_dict4Y = data["res_dict4Y"]

NN = data["NN"]
TT = data["TT"]
PP = data["PP"]
KK = data["KK"]
W = data["W"]
GG = 3

#####################

# Comparing all policies.
all_histories = []  # 每个元素是一个策略对应的 MC 历史列表
labels = []  # 每个历史对应的策略标签

for i in range(len(pi_list)):
    pi = pi_list[i]
    Smodel_params_hat, Ymodel_params_hat, network_params_hat = decode_res_dict(res_dict4S, res_dict4Y)

    estimated_mc_history = monte_carlo_estimate(pi=pi, GG=GG, NN=NN, TT=TT, KK=KK, PP=PP, W=W,
                                                network_params=network_params_hat,
                                                Smodel_params=Smodel_params_hat,
                                                Ymodel_params=Ymodel_params_hat,
                                                Ztensor=Ztensor, S0tensor=Stensor[:, 0, :, :],
                                                num_runs=num_runs, eps_sigma4S=eps_sigma4S,
                                                eps_sigma4Y=eps_sigma4Y,
                                                mode="simu", eco=True)

    estimated_value_function = sum(estimated_mc_history)/len(estimated_mc_history)
    print(f"The estimated value function is {estimated_value_function}.")

    all_histories.append(estimated_mc_history)
    labels.append(f"Policy {i + 1} Estimated")  # 或者用 pi 本身命名

    # 将当前遍历的结果记录到临时 DataFrame 中
    temp_df = pd.DataFrame({
        "Pi": Pi_names[i],
        "NN": [NN], "TT": [TT], "PP": [PP], "KK": [KK],
        "Estimated Mean": np.mean(estimated_mc_history), "Estimated Std": np.std(estimated_mc_history)
    })

    # 使用 pd.concat() 来合并 DataFrame
    results_df = pd.concat([results_df, temp_df], ignore_index=True)

    ttest_results_df = compare_policies_ttest(all_histories, Pi_names)
    ttest_results_df.to_excel("经济面板数据//Eco_ttest_results.xlsx", index=False)

barplot(results_df, filename=f"经济面板数据//Eco_Policy_comparison_barplot.png", estimated_only=True)
results_df.to_excel(f"经济面板数据//Eco_mc_comparison_results.xlsx", index=False)
json_count += 1

#########################
#
# def plot_param_1d(results, param_name, filename=None, policy_name=None):
#     """
#     results: List[({param_name: value}, avg_reward)]
#     """
#     x = [param[param_name] for param, reward in results]
#     y = [reward for param, reward in results]
#
#     plt.figure(figsize=(6, 4))
#     plt.plot(x, y, marker='o', linewidth=5)
#     plt.xlabel(param_name, fontsize=20)
#     plt.ylabel("Average Reward", fontsize=20)
#     # plt.title(f"{policy_name} 策略参数调优 - {param_name}")
#     plt.grid(True)
#
#     if filename:
#         plt.savefig(filename, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#
# def plot_param_2d(results, param_x, param_y, filename=None, policy_name=None):
#     """
#     results: List[({param_x: val1, param_y: val2}, avg_reward)]
#     """
#     df = pd.DataFrame([
#         {param_x: param[param_x], param_y: param[param_y], "reward": reward}
#         for param, reward in results
#     ])
#     pivot = df.pivot(index=param_y, columns=param_x, values="reward")
#
#     plt.figure(figsize=(8, 6))
#     ax = sns.heatmap(pivot, annot=True, cmap='YlGnBu')
#
#     # 设置坐标轴标签
#     # ax.set_title('pi_gdp_growth 策略参数调优热力图 (A_base1 vs beta)')
#     ax.set_xlabel('A_base1')
#     ax.set_ylabel('beta')
#
#     # 格式化横轴坐标，使小数位最多保留两位
#     ax.set_xticklabels([f'{float(label.get_text()):.2f}' for label in ax.get_xticklabels()])
#     # plt.title(f"{policy_name} 策略参数调优热力图 ({param_x} vs {param_y})")
#     plt.xlabel(param_x, fontsize=20)
#     plt.ylabel(param_y, fontsize=20)
#     plt.tight_layout()
#
#     if filename:
#         plt.savefig(filename, bbox_inches='tight')
#     # plt.show()
#     plt.close()
#
# def auto_optimize_policies(policy_info_list, common_args, save_dir="plots"):
#     os.makedirs(save_dir, exist_ok=True)
#     results_summary = []
#
#     for policy_dict in policy_info_list:
#         policy_func = policy_dict["func"]
#         policy_name = policy_dict["name"]
#         param_grid = policy_dict["param_grid"]
#
#         print(f"\n正在优化策略: {policy_name} ...")
#
#         best_params, best_reward, all_results = optimize_policy_parameters(
#             policy_func=policy_func,
#             param_grid=param_grid,
#             **common_args
#         )
#
#         # 自动画图
#         if len(param_grid) == 1:
#             param_name = list(param_grid.keys())[0]
#             plot_param_1d(all_results, param_name=param_name,
#                           filename=os.path.join(save_dir, f"{policy_name}_{param_name}.png"), policy_name=policy_name)
#         elif len(param_grid) == 2:
#             param_names = list(param_grid.keys())
#             plot_param_2d(all_results, param_x=param_names[0], param_y=param_names[1],
#                           filename=os.path.join(save_dir, f"{policy_name}_heatmap.png"), policy_name=policy_name)
#         else:
#             print("⚠️ 目前仅支持 1-2 个参数的可视化。")
#
#         results_summary.append({
#             "Policy": policy_name,
#             "Best Params": best_params,
#             "Best Reward": best_reward
#         })
#
#     return results_summary
#
# policy_info_list = [
#     {
#         "name": "pi_const",
#         "func": pi_const,
#         "param_grid": {
#             "A_base1": np.linspace(0.1, 1.0, 10)
#         }
#     },
#     {
#         "name": "pi_time_linear",
#         "func": pi_time_linear,
#         "param_grid": {
#             "A_base1": np.linspace(0.1, 0.5, 5),
#             "gamma": np.linspace(0.02, 0.01, 5)}
#     },
#     {
#         "name": "pi_gdp_change",
#         "func": pi_gdp_change,
#         "param_grid": {"alpha": np.linspace(0.1, 1.0, 10)}
#     },
#     {
#         "name": "pi_gdp_threshold",
#         "func": pi_gdp_threshold,
#         "param_grid": {
#             "threshold": np.linspace(0.01, 0.05, 5),
#             "delta": np.linspace(0.01, 0.05, 5)
#         }
#     }
# ]
#
# Smodel_params_hat, Ymodel_params_hat, network_params_hat = decode_res_dict(res_dict4S, res_dict4Y)
#
# common_args = {
#     "GG": GG, "NN": NN, "TT": TT, "KK": KK, "PP": PP, "W": W,
#     "network_params": network_params_hat,
#     "Smodel_params": Smodel_params_hat,
#     "Ymodel_params": Ymodel_params_hat,
#     "S0tensor": Stensor[:, 0, :, :],
#     "Ztensor": Ztensor,
#     "num_runs": num_runs,
#     "eps_sigma4S": eps_sigma4S,
#     "eps_sigma4Y": eps_sigma4Y
# }
#
# summary = auto_optimize_policies(policy_info_list, common_args, save_dir="plots//auto_optim")
#
# # 保存 summary
# pd.DataFrame(summary).to_excel("plots//auto_optim//summary.xlsx", index=False)




