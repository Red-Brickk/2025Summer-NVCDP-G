import numpy as np
import itertools

def optimize_policy_parameters(
    policy_func,                     # 策略函数，如 pi_time_linear
    param_grid: dict,                # 参数网格，如 {"gamma": [0.0, 0.05, 0.1]}
    GG=3, NN=100, TT=20, KK=5, PP=3, W=None,
    network_params=None,
    Smodel_params=None,
    Ymodel_params=None,
    num_runs=10,
    eps_sigma=0.5,
    A_base=7.0
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
            return policy_func(S_current, S_past, A_prev=A_prev, A_base=A_base, t=t, **param_dict)

        # Monte Carlo 评估该参数组合
        rewards = monte_carlo_estimate(pi, GG=GG, NN=NN, TT=TT, KK=KK, PP=PP, W=W,
                                       network_params=network_params,
                                       Smodel_params=Smodel_params,
                                       Ymodel_params=Ymodel_params,
                                       num_runs=num_runs,
                                       eps_sigma=eps_sigma,
                                       mode="", eco=True)
        avg_reward = np.mean(rewards)
        all_results.append((param_dict, avg_reward))

        print(f"参数 {param_dict} → 平均 reward = {avg_reward:.4f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = param_dict

    print(f"\n✅ 最优参数: {best_params}，对应平均 reward = {best_reward:.4f}")
    return best_params, best_reward, all_results
