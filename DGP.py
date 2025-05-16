# Oct 14th
# Creating my own code.

import numpy as np
from graspologic.simulations import sbm
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

### J = J0 +d

def B(param_t, degree, control_point_id, knots):
    if degree == 0:
        return 1.0 if knots[control_point_id] <= param_t < knots[control_point_id + 1] else 0.0
    if knots[control_point_id + degree] == knots[control_point_id]:  # 如果节点相等
        c1 = 0.0
    else:  # x适合落在此区间内，向下追溯
        c1 = (param_t - knots[control_point_id]) / (knots[control_point_id + degree] - knots[control_point_id]) * B(
            param_t, degree - 1, control_point_id, knots)
    if knots[control_point_id + degree + 1] == knots[control_point_id + 1]:  # 另一个加和项
        c2 = 0.0
    else:
        c2 = (knots[control_point_id + degree + 1] - param_t) / (
                knots[control_point_id + degree + 1] - knots[control_point_id + 1]) * B(param_t, degree - 1,
                                                                                        control_point_id + 1, knots)
    return c1 + c2


def SieveBasis(Tvec, spline_type="b_spline", J0=None, visualize=False):
    d = 4
    if J0 is None:
        J0 = 2
    J = J0

    knots = np.ones((J + 2 * (d - 1), 1)).flatten()
    knots[0:(d - 1)] = np.zeros((d - 1))
    knots[(d - 1):(J + d - 1)] = np.percentile(Tvec, [100 * j / (J - 1) for j in range(0, J)])

    if spline_type == "b_spline":
        bt = np.zeros((len(Tvec), (J0 + d - 2)))
        for i in range((J0 + d - 2)):
            for t in range(len(Tvec)):
                bt[t, i] = B(param_t=Tvec[t], degree=d - 1, control_point_id=i, knots=knots)
    else:
        raise ValueError("Not implemented yet.")

    if visualize:
        plt.figure(figsize=(10, 6))
        xx = np.linspace(0, 1, 1000)
        yt = np.zeros((len(xx), (J0 + d - 2)))
        for i in range((J0 + d - 2)):
            for t in range(len(xx)):
                yt[t, i] = B(param_t=xx[t], degree=d - 1, control_point_id=i, knots=knots)
        for i in range((J0 + d - 2)):
            plt.plot(xx, yt[:, i], label=f"Basis {i + 1}")
        plt.title(f"{spline_type.capitalize()} Basis Functions")
        plt.xlabel("Tvec")
        plt.ylabel("Basis Value")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.close()

    return bt  # TT J0

def StateEvolve(NN, Gvec, W, D1g, Sold, Xi, Ztensor, Att, Abartt, epsmat4state, tt, pp, PP,KK):
    temp_inv = np.linalg.inv(
        np.eye(NN) - np.diag(D1g[:, tt, pp][Gvec].flatten()) @ W
    )
    temp_Sbar = np.einsum('ij,jklm->iklm', W, Sold[:, :, pp:(pp + 1), :])
    temp_mat_mult = Xi[:, tt, pp, :][Gvec][:, np.newaxis, np.newaxis, :] * np.concatenate(
        (np.ones(shape=(NN, 1, 1, 1))
         , np.reshape(Sold, newshape=(NN, 1, 1, PP))
         , temp_Sbar
         , Att
         , Abartt
         , np.reshape(Ztensor[:, tt, :, :], newshape=(NN, 1, 1, KK))), axis=-1
    )
    temp_mat_mult_noised = np.sum(temp_mat_mult, axis=-1, keepdims=True) + epsmat4state[:, tt:tt + 1,
                                                                           pp:pp + 1, :]

    ###############################
    # temp_inv NN
    # Phi Xi[:, tt, pp, 1:2]
    # test_mat = np.einsum("ij, jk->ik",temp_inv, np.diag(Xi[:, tt, pp, 1][Gvec]))
    # eig_val = np.linalg.eigvals(test_mat)
    # print(f"TEMP INV The largest eigenvalue:{np.max(np.linalg.eigvals(temp_inv))}, the smallest eigenvalue:{np.min(np.linalg.eigvals(temp_inv))}.")
    # print(f"TEMP MULT The largest eigenvalue:{np.max(eig_val)}, the smallest eigenvalue:{np.min(eig_val)}.")
    ###############################

    return np.einsum('ij,jklm->iklm', temp_inv, temp_mat_mult_noised)

class DGP:
    """
    Create a simulation dataset with desired properties
    PP: the dimension of the state vector.
    KK: the dimension of the exogenous covariates.
    """

    def __init__(self, NN=50, TT=40, GG=3, PP=1, KK=1, eps_sigma=1, net_name="sbm",
                 net_extra_param_dict=None, param_mode_dict=None, param_seed4group=None, seed=2024,
                 spline_type="b_spline"):
        self.Stensor = None
        self.NN = NN
        self.TT = TT
        self.GG = GG
        self.PP = PP
        self.KK = KK
        self.eps_sigma = eps_sigma
        self.Gvec = None
        self.net_name = net_name
        self.net_extra_param_dict = net_extra_param_dict
        self.seed = seed
        self.splinetype = spline_type

        if param_mode_dict is None:
            self.param_mode_dict = dict(zip(
                ['ag', 'D1g', 'D2g', 'Phi1g', 'Phi2g', 'bg', 'betag', "gamma1g", "gamma2g", "zeta1g", "zeta2g"],
                [1 for _ in range(11)]
            ))
        else:
            self.param_mode_dict = param_mode_dict

        if param_seed4group is None:
            self.param_seed4group = [x + 24 for x in range(GG)]
        else:
            self.param_seed4group = param_seed4group

    # This is connecting matrix. You can see this as a geographical map.
    def _generate_network(self):
        if self.net_name == "sbm":
            np.random.seed(self.seed)
            K = self.net_extra_param_dict["K"]
            n_list = [int(self.NN / K) for _ in range(K)]
            n_list[K - 1] = self.NN - (K - 1) * int(self.NN / K)
            adjacency_matrix = sbm(n=n_list, p=self.net_extra_param_dict["sbm_matrix"])
            row_sums = np.sum(adjacency_matrix, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零错误
            self.W = adjacency_matrix / row_sums

        elif self.net_name == "scale_free":
            # 使用 Barabási-Albert 模型生成幂律分布网络
            np.random.seed(self.seed)
            m = self.net_extra_param_dict.get("m", 0)  # 每个新节点连接的边数
            G = nx.barabasi_albert_graph(self.NN, m, seed=self.seed)
            adjacency_matrix = nx.to_numpy_array(G)
            row_sums = np.sum(adjacency_matrix, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零错误
            self.W = adjacency_matrix / row_sums

        elif self.net_name == "configuration":
            # 使用配置模型生成幂律分布网络
            np.random.seed(self.seed)
            degree_sequence = np.random.zipf(a=self.net_extra_param_dict["alpha"], size=self.NN)
            degree_sequence = np.clip(degree_sequence, 1, self.NN - 1)  # 防止出现过大的度
            G = nx.configuration_model(degree_sequence, seed=self.seed)
            G = nx.Graph(G)  # 移除多重边和自环
            G.remove_edges_from(nx.selfloop_edges(G))
            adjacency_matrix = nx.to_numpy_array(G)
            row_sums = np.sum(adjacency_matrix, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零错误
            self.W = adjacency_matrix / row_sums

        else:
            tempA = np.eye(self.NN)
            row_sums = np.sum(tempA, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零错误
            self.W = tempA / row_sums

            raise ValueError("Network type not implemented yet.")

    # This is time vector, used for generating time varying function.
    def _generate_Timevec(self):
        np.random.seed(self.seed + 400)
        self.Timevec = np.zeros((self.TT, 1))
        self.Timevec[1:(self.TT - 1), 0] = np.sort(np.random.uniform(size=(self.TT - 2)))
        self.Timevec[self.TT - 1, 0] = 1.0

    # Calculate time varying function.
    def _time_vary_func(self, fun_type="sin", power=2, power_cent=0, mu=0.0, s=1.0, period=2, const=0):
        if fun_type == "zero":
            return np.zeros((len(self.Timevec), 1))
        elif fun_type == "sin":
            return np.sin((2/period)*np.pi*self.Timevec)
        elif fun_type == "power":
            return np.power(self.Timevec-power_cent, power)
        elif fun_type == "F":
            return 1 / (1 + np.exp(-(self.Timevec - mu) / s))
        elif fun_type == "const":
            return np.array([const for tt in self.Timevec]).reshape((self.TT, 1))
        else:
            raise ValueError(f"Unsupported fun_type:{fun_type}")

    def assign_time_vary_params(self, coeff, params):
        return sum(c * self._time_vary_func(fun_type=ft, **args) for c, (ft, args) in zip(coeff, params))

    def _generate_time_vary_params(self):
        """
        This function is used to generate the following real parms:
        ag: GG TT PP 1 group specific trend
        D1g: GG TT PP 1 simultaneous network effect
        D2g: GG TT PP 1 lagged network effect
        Phi1g: GG TT PP PP momentum effect
        Phi2g: GG TT PP KK exogenous effect
        :return: update the params, no return.
        """

        global coeff_ag
        self.ag = np.zeros((self.GG, self.TT, self.PP, 1))
        self.zeta1g = np.zeros((self.GG, self.TT, self.PP, 1))
        self.zeta2g = np.zeros((self.GG, self.TT, self.PP, 1))
        self.D1g = np.zeros((self.GG, self.TT, self.PP, 1))
        self.D2g = np.zeros((self.GG, self.TT, self.PP, 1))
        self.Phi1g = np.zeros((self.GG, self.TT, self.PP, self.PP))
        self.Phi2g = np.zeros((self.GG, self.TT, self.PP, self.KK))

        self.bg = np.zeros((self.GG, self.TT, 1, 1))
        self.gamma1g = np.zeros((self.GG, self.TT, 1, 1))
        self.gamma2g = np.zeros((self.GG, self.TT, 1, 1))
        self.betag = np.zeros((self.GG, self.TT, self.PP, 1))

        self.sieve = SieveBasis(self.Timevec, spline_type=self.splinetype, J0=2, visualize=False)
        TT, JJ = self.sieve.shape

        # Assign values
        for gg in range(self.GG):
            for pp in range(self.PP):
                for param, param_mode in self.param_mode_dict.items():
                    if param == 'ag':
                        coeff_ag = [0, 0, 0, 0, 0, 0, 0]
                        params_ag = [
                            ("power", {"power": 0}),
                            ("power", {"power": 1}),
                            ("power", {"power": 2}),
                            ("power", {"power": 3}),
                            ("F", {"mu": 0.6, "s": 0.01}),
                            ("sin", {"period": 2}),
                            ("const", {"const": 1})
                        ]
                        if gg == 0:
                            coeff_ag = [0, 0, 0, 0, 1, 2, 1]
                            params_ag[4] = ("F", {"mu": 0.5, "s": 0.1})
                        elif gg == 1:
                            coeff_ag = [0, 2, -6, 4, 1, 2, -1]
                            params_ag[4] = ("F", {"mu": 0.7, "s": 0.05})
                        elif gg == 2:
                            coeff_ag = [0, 4, -8, 4, 1, 2, 0]
                            params_ag[4] = ("F", {"mu": 0.6, "s": 0.05})

                        self.ag[gg, :, pp] = self.assign_time_vary_params(coeff=coeff_ag, params=params_ag)

                    elif param == 'zeta1g':
                        coeff_zeta1g = [0, 0, 0, 0, 0]
                        params_zeta1g = [
                            ("power", {"power": 0}),
                            ("power", {"power": 1}),
                            ("power", {"power": 2}),
                            ("power", {"power": 3}),
                            ("F", {"mu": 0.6, "s": 0.01}),
                        ]
                        if gg == 0:
                            coeff_zeta1g = [0, 0, 0, 0, 0.1]
                            params_zeta1g[4] = ("F", {"mu": 0.5, "s": 0.1})
                        elif gg == 1:
                            coeff_zeta1g = [0, 2, -6, +4, 0.1]
                            params_zeta1g[4] = ("F", {"mu": 0.7, "s": 0.05})
                        elif gg == 2:
                            coeff_zeta1g = [0, 4, -8, +4, 0.1]
                            params_zeta1g[4] = ("F", {"mu": 0.6, "s": 0.05})

                        self.zeta1g[gg, :, pp] = self.assign_time_vary_params(coeff= coeff_zeta1g, params=params_zeta1g)
                        # self.zeta1g[gg, :, pp] = self.assign_time_vary_params(coeff= coeff_zeta1g, params=params_zeta1g)*0

                    elif param == 'zeta2g':
                        coeff_zeta2g = [0, 0, 0, 0, 0]
                        params_zeta2g = [
                            ("power", {"power": 0}),
                            ("power", {"power": 1}),
                            ("power", {"power": 2}),
                            ("power", {"power": 3}),
                            ("F", {"mu": 0.6, "s": 0.01}),
                        ]
                        if gg == 0:
                            coeff_zeta2g = [0, 0, 0, 0, 0.1]
                            params_zeta2g[4] = ("F", {"mu": 0.5, "s": 0.1})
                        elif gg == 1:
                            coeff_zeta2g = [0, 2, -6, +4, 0.1]
                            params_zeta2g[4] = ("F", {"mu": 0.7, "s": 0.05})
                        elif gg == 2:
                            coeff_zeta2g = [0, 4, -8, +4, 0.1]
                            params_zeta2g[4] = ("F", {"mu": 0.6, "s": 0.05})

                        self.zeta2g[gg, :, pp] = self.assign_time_vary_params(coeff= coeff_zeta2g, params=params_zeta2g)
                        # self.zeta2g[gg, :, pp] = self.assign_time_vary_params(coeff= coeff_zeta2g, params=params_zeta2g) * 0

                    elif param == 'D1g':
                        np.random.seed(self.param_seed4group[gg] + 600)
                        self.D1g[gg, :, pp] = self._time_vary_func(fun_type="zero")
                        coeff_D1g = [1,2,3,4,5,6]
                        params_D1g = [
                            ("power", {"power": 0}),
                            ("power", {"power": 1}),
                            ("power", {"power": 2}),
                            ("power", {"power": 3}),
                            ("F", {"mu": 0.6, "s": 0.01}),
                            ("const", {"const":0.9})
                        ]

                        if gg == 0:
                            coeff_D1g = [-0.5,0.2,-0.5,0.2,0.1, -0.3]
                            params_D1g[4] = ("F", {"mu": 0.6, "s": 0.03})
                        elif gg == 1:
                            coeff_D1g = [-0.5,0.1,-0.3,+0.2,0.1, -0.3]
                            params_D1g[4] = ("F", {"mu": 0.2, "s": 0.04})
                        elif gg == 2:
                            coeff_D1g = [-0.05,0.5,-0.05,0,0.1, -0.3]
                            params_D1g[4] = ("F", {"mu": 0.8, "s": 0.07})

                        coeff_D1g = [0,0,0,0,0, -0.8] # 本句用于对内生性进行测试
                        self.D1g[gg, :, pp] = self.assign_time_vary_params(coeff= coeff_D1g, params=params_D1g)
                        # self.D1g[gg, :, pp] = self.assign_time_vary_params(coeff= coeff_D1g, params=params_D1g) * 0

                    elif param == 'D2g':
                        np.random.seed(self.param_seed4group[gg] + 600)
                        self.D2g[gg, :, pp] = self._time_vary_func(fun_type="zero")
                        coeff_D2g = [0,0,0,0,0]
                        F_D2g = [1, 1]

                        params_D2g = [
                            ("power", {"power": 0}),
                            ("power", {"power": 1}),
                            ("power", {"power": 2}),
                            ("power", {"power": 3}),
                            ("F", {"mu": 0.6, "s": 0.01}),
                        ]

                        if gg == 0:
                            coeff_D2g = [0, 2, -1, 2, 0.01]
                            params_D2g[4] = ("F", {"mu": 0.6, "s": 0.03})
                        elif gg == 1:
                            coeff_D2g = [-0.5,1,-3,+2,0.01]
                            params_D2g[4] = ("F", {"mu": 0.2, "s": 0.04})
                        elif gg == 2:
                            coeff_D2g = [-0.5,0.5,-0.5,0,0.01]
                            params_D2g[4] = ("F", {"mu": 0.8, "s": 0.07})
                        self.D2g[gg, :, pp] = self.assign_time_vary_params(coeff= coeff_D2g, params=params_D2g)
                        # self.D2g[gg, :, pp] = self.assign_time_vary_params(coeff= coeff_D2g, params=params_D2g) * 0

                    elif param == 'Phi1g':
                        np.random.seed(self.param_seed4group[gg] + 800)
                        for pp2 in range(self.PP):
                            # self.Phi1g[gg, :, pp, pp2:(pp2 + 1)] = self._time_vary_func(fun_type="zero")
                            coeff_Phi1g = [0, 0, 0, 0, 0]
                            F_Phi1g = [1, 1]
                            params_Phi1g = [
                                ("power", {"power": 0}),
                                ("power", {"power": 1}),
                                ("power", {"power": 2}),
                                ("power", {"power": 3}),
                                ("F", {"mu": 0.6, "s": 0.01}),
                            ]

                            if gg == 0:
                                coeff_Phi1g = [0, 2, -4, 2, 0.1]
                                params_Phi1g[4] = ("F", {"mu": 0.6, "s": 0.81})
                            elif gg == 1:
                                coeff_Phi1g = [0, 1, -3, +2, 0.1]
                                params_Phi1g[4] = ("F", {"mu": 0.7, "s": 0.04})
                            elif gg == 2:
                                coeff_Phi1g = [0, 0.5, -0.5, 0, 0.1]
                                params_Phi1g[4] = ("F", {"mu": 0.4, "s": 0.07})

                            self.Phi1g[gg, :, pp, pp2:(pp2 + 1)] = self.assign_time_vary_params(coeff=coeff_Phi1g,
                                                                                              params=params_Phi1g)
                            # self.Phi1g[gg, :, pp, pp2:(pp2 + 1)] = self.assign_time_vary_params(coeff=coeff_Phi1g,
                            #                                                                   params=params_Phi1g) * 0



                    elif param == 'Phi2g':
                        np.random.seed(self.param_seed4group[gg] + 700)
                        for kk in range(self.KK):
                            self.Phi2g[gg, :, pp, kk:(kk + 1)] = self._time_vary_func(fun_type="zero")
                            coeff_Phi2g = [0, 0, 0, 0, 0]
                            F_Phi2g = [1, 1]
                            params_Phi2g = [
                                ("power", {"power": 0}),
                                ("power", {"power": 1}),
                                ("power", {"power": 2}),
                                ("power", {"power": 3}),
                                ("F", {"mu": 0.6, "s": 0.01}),
                            ]

                            if gg == 0:
                                coeff_Phi2g = [0, 2, -4, 2, 1]
                                params_Phi2g[4] = ("F", {"mu": 0.6, "s": 0.81})
                            elif gg == 1:
                                coeff_Phi2g = [0, 1, -3, +2, 1]
                                params_Phi2g[4] = ("F", {"mu": 0.7, "s": 0.04})
                            elif gg == 2:
                                coeff_Phi2g = [0, 0.5, -0.5, 0, 1]
                                params_Phi2g[4] = ("F", {"mu": 0.4, "s": 0.07})

                            self.Phi2g[gg, :, pp, kk:(kk + 1)] = self.assign_time_vary_params(coeff=coeff_Phi2g, params=params_Phi2g)
                            # self.Phi2g[gg, :, pp, kk:(kk + 1)] = self.assign_time_vary_params(coeff=coeff_Phi2g, params=params_Phi2g) * 0


                    elif param in ['bg', 'betag', "gamma1g", "gamma2g"]:
                        pass

                    else:
                        raise ValueError(f"Error in parameter name:{param}.")

        self.Xi = np.concatenate((self.ag, self.Phi1g,
                                  self.D2g, self.zeta1g, self.zeta2g, self.Phi2g), axis=-1)

        # Assign values
        for gg in range(self.GG):
            for param, param_mode in self.param_mode_dict.items():
                if param == 'bg':
                    coeff_bg = [0, 0, 0, 0, 0]
                    params_bg = [
                        ("power", {"power": 0}),
                        ("power", {"power": 1}),
                        ("power", {"power": 2}),
                        ("power", {"power": 3}),
                        ("F", {"mu": 0.6, "s": 0.01}),
                    ]
                    if gg == 0:
                        coeff_bg = [0, 0, 0, 0, 1]
                        params_bg[4] = ("F", {"mu": 0.5, "s": 0.1})
                    elif gg == 1:
                        coeff_bg = [0, 2, -6, +4, 0.1]
                        params_bg[4] = ("F", {"mu": 0.7, "s": 0.05})
                    elif gg == 2:
                        coeff_bg = [0, 4, -8, +4, 0.1]
                        params_bg[4] = ("F", {"mu": 0.6, "s": 0.05})

                    self.bg[gg, :, :, 0] = self.assign_time_vary_params(coeff= coeff_bg, params=params_bg)
                    # self.bg[gg, :, :, 0] = self.assign_time_vary_params(coeff= coeff_bg, params=params_bg) * 0

                elif param == 'gamma1g':
                    coeff_gamma1g = [0, 0, 0, 0, 0]
                    params_gamma1g = [
                        ("power", {"power": 0}),
                        ("power", {"power": 1}),
                        ("power", {"power": 2}),
                        ("power", {"power": 3}),
                        ("F", {"mu": 0.6, "s": 0.01}),
                    ]
                    if gg == 0:
                        coeff_gamma1g = [0, 0, 0, 0, 1]
                        params_gamma1g[4] = ("F", {"mu": 0.5, "s": 0.1})
                    elif gg == 1:
                        coeff_gamma1g = [0, 2, -6, +4, 0.1]
                        params_gamma1g[4] = ("F", {"mu": 0.7, "s": 0.05})
                    elif gg == 2:
                        coeff_gamma1g = [0, 4, -8, +4, 0.1]
                        params_gamma1g[4] = ("F", {"mu": 0.6, "s": 0.05})
                    self.gamma1g[gg, :, 0] = self.assign_time_vary_params(coeff= coeff_gamma1g, params=params_gamma1g)
                    # self.gamma1g[gg, :, 0] = self.assign_time_vary_params(coeff= coeff_gamma1g, params=params_gamma1g) * 0

                elif param == 'gamma2g':
                    coeff_gamma2g = [0, 0, 0, 0, 0]
                    params_gamma2g = [
                        ("power", {"power": 0}),
                        ("power", {"power": 1}),
                        ("power", {"power": 2}),
                        ("power", {"power": 3}),
                        ("F", {"mu": 0.6, "s": 0.01}),
                    ]
                    if gg == 0:
                        coeff_gamma2g = [0, 0, 0, 0, 0.1]
                        params_gamma2g[4] = ("F", {"mu": 0.5, "s": 0.1})
                    elif gg == 1:
                        coeff_gamma2g = [0, 2, -6, +4, 0.1]
                        params_gamma2g[4] = ("F", {"mu": 0.7, "s": 0.05})
                    elif gg == 2:
                        coeff_gamma2g = [0, 4, -8, +4, 0.1]
                        params_gamma2g[4] = ("F", {"mu": 0.6, "s": 0.05})
                    self.gamma2g[gg, :, 0] = self.assign_time_vary_params(coeff= coeff_gamma2g, params=params_gamma2g)
                    # self.gamma2g[gg, :, 0] = self.assign_time_vary_params(coeff= coeff_gamma2g, params=params_gamma2g) * 0

                elif param == 'betag':
                    coeff_betag = [0, 0, 0, 0, 0]
                    params_betag = [
                        ("power", {"power": 0}),
                        ("power", {"power": 1}),
                        ("power", {"power": 2}),
                        ("power", {"power": 3}),
                        ("F", {"mu": 0.6, "s": 0.01}),
                    ]
                    if gg == 0:
                        coeff_betag = [0, 0, 0, 0, 1]
                        params_betag[4] = ("F", {"mu": 0.5, "s": 0.1})
                    elif gg == 1:
                        coeff_betag = [0, 2, -6, +4, 1]
                        params_betag[4] = ("F", {"mu": 0.7, "s": 0.05})
                    elif gg == 2:
                        coeff_betag = [0, 4, -8, +4, 1]
                        params_betag[4] = ("F", {"mu": 0.6, "s": 0.05})

                    for pp in range(self.PP):
                        self.betag[gg, :, pp, :] = self.assign_time_vary_params(coeff=coeff_betag, params=params_betag)
                        # self.betag[gg, :, pp, :] = self.assign_time_vary_params(coeff=coeff_betag, params=params_betag) * 0

                elif param in ['ag', 'D1g', 'D2g', 'Phi1g', 'Phi2g', "zeta1g", "zeta2g"]:
                    pass

                else:
                    raise ValueError("Error in parameter name.")


    # Initial state
    def _generate_S0mat(self):
        np.random.seed(self.seed + 100)
        self.S0 = np.random.normal(loc=0, scale=self.eps_sigma, size=(self.NN, 1, self.PP, 1))

    # noise tensor
    def _generate_eps_mat4state(self):
        np.random.seed(self.seed + 200)
        self.epsmat4state = np.random.normal(loc=0, scale=self.eps_sigma, size=(self.NN, self.TT, self.PP, 1))
    # noise tensor
    def _generate_eps_mat4reward(self):
        np.random.seed(self.seed + 201)
        self.epsmat4reward = np.random.normal(loc=0, scale=self.eps_sigma, size=(self.NN, self.TT, 1, 1))

    # Initial state
    def _generate_exogenous_variates(self):
        np.random.seed(self.seed + 300)
        self.Ztensor = np.random.normal(loc=0, scale=0.1, size=(self.NN, self.TT, self.KK, 1))

    def _generate_actions(self):
        np.random.seed(self.seed + 400)
        self.A = np.random.binomial(n=1, p=0.5, size=(self.NN, self.TT, 1, 1))
        self.Abar = np.einsum("ij, jklm->iklm", self.W, self.A)

    def GenerateStateTensor(self, group_proportions=None):
        self._generate_network()
        self._generate_S0mat()
        self._generate_Timevec()
        self._generate_time_vary_params()
        self._generate_eps_mat4state()
        self._generate_exogenous_variates()
        self._generate_actions()

        if group_proportions is None:
            self.Gvec = np.random.choice([x for x in range(self.GG)], size=self.NN, replace=True)
        else:
            self.Gvec = np.random.choice([x for x in range(self.GG)], size=self.NN, replace=True,
                                         p=group_proportions)

        self.Stensor = np.zeros((self.NN, self.TT, self.PP, 1))

        Sold = self.S0
        for tt in range(self.TT):
            for pp in range(self.PP):
                self.Stensor[:, tt:tt + 1, pp:pp + 1, :] = StateEvolve(
                    NN=self.NN, Gvec=self.Gvec, W=self.W,
                    D1g=self.D1g, Sold=Sold, Xi=self.Xi,
                    Ztensor=self.Ztensor,
                    Att=self.A[:, tt:tt + 1, :, :], Abartt=self.Abar[:, tt:tt + 1, :, :],
                    epsmat4state=self.epsmat4state, tt=tt, pp=pp,PP=self.PP, KK=self.KK)

            Sold = self.Stensor[:, tt:tt + 1, :, :]

    def GenerateRewardTensor(self, group_proportions=None):
        self._generate_network()
        self._generate_S0mat()
        self._generate_Timevec()
        self._generate_time_vary_params()
        self._generate_eps_mat4reward()
        self._generate_exogenous_variates()
        self._generate_actions()

        if group_proportions is None:
            self.Gvec = np.random.choice([x for x in range(self.GG)], size=self.NN, replace=True)
        else:
            self.Gvec = np.random.choice([x for x in range(self.GG)], size=self.NN, replace=True,
                                         p=group_proportions)

        self.Ytensor = np.zeros((self.NN, self.TT, 1))
        temp_feature = np.concatenate((np.ones(shape=(self.NN, self.TT, 1, 1)),
            self.Stensor, self.A, self.Abar), axis=2
        )
        temp_coeffs = np.concatenate((self.bg[self.Gvec],
            self.betag[self.Gvec], self.gamma1g[self.Gvec], self.gamma2g[self.Gvec],
            ), axis=2
        )
        self.Ytensor[:, :, :] = (np.einsum("ijkl, ijkm->ijlm",
                                          temp_feature, temp_coeffs) + self.epsmat4reward)[:,:,:,0]

    def visualize_network(self):
        # 确保 self.W 是一个 numpy 数组
        if not isinstance(self.W, np.ndarray):
            raise ValueError("self.W must be a numpy array")

        # 获取节点数量
        num_nodes = self.W.shape[0]

        # 创建热力图
        plt.figure(figsize=(10, 8))

        # 使用 seaborn 的 heatmap 函数绘制热力图
        ax = sns.heatmap(self.W, annot=True, fmt=".1f", cmap="coolwarm",
                    linewidths=.5, cbar_kws={"label": "Connection Strength"})

        plt.title("Social Network Adjacency Matrix Heatmap", fontsize=16)
        plt.xlabel("Node Index", fontsize=14)
        plt.ylabel("Node Index", fontsize=14)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Connection Strength", fontsize=12)
        plt.tight_layout()
        plt.savefig("Adjacency matrix.png")
        # plt.show()
        plt.close()

    def plot_time_series(self, vis_list=None):
        if vis_list is None:
            vis_list = ["S", "Y", "Z"]
        if self.GG > 4:
            raise ValueError("Too much groups to visualize.")
        line_styles = ['-', '--', '-.',':']

        # 画状态曲线
        if "S" in vis_list:
            for pp_index in range(self.PP):
                fig, ax = plt.subplots(figsize=(12, 6))
                data_state = self.Stensor[:, :, pp_index, 0]
                for i in range(self.NN):
                    ax.plot(range(self.TT), data_state[i, :], alpha=0.6, linewidth=1.2,
                            linestyle=line_styles[self.Gvec[i] - 1],
                            label=f"State {i + 1}" if i < 5 else "_nolegend_")

                ax.set_title(f"State Curves (pp_index={pp_index})")
                ax.set_ylabel("State Value")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)

                plt.tight_layout()
                plt.savefig(f"state_curves_pp_{pp_index}.png")
                # plt.show()
                plt.close()

        # 画外部变量曲线
        if "Z" in vis_list:
            for kk_index in range(self.KK):
                fig, ax = plt.subplots(figsize=(12, 6))
                data_EV = self.Ztensor[:, :, kk_index, 0]
                for i in range(self.NN):
                    ax.plot(range(self.TT), data_EV[i, :], alpha=0.6, linewidth=1.2,
                            linestyle=line_styles[self.Gvec[i] - 1],
                            label=f"State {i + 1}" if i < 5 else "_nolegend_")

                ax.set_title(f"Exogenous Variable Curves (kk_index={kk_index})")
                ax.set_ylabel("Exogenous Variable Value")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)

                plt.tight_layout()
                plt.savefig(f"exogenous_variable_curves_kk_{kk_index}.png")
                # plt.show()
                plt.close()


        # 画奖励曲线
        if "Y" in vis_list:
            fig, ax = plt.subplots(figsize=(12, 6))
            data_reward = self.Ytensor[:, :, 0]
            for i in range(self.NN):
                ax.plot(range(self.TT), data_reward[i, :], alpha=0.6, linewidth=1.2,
                        linestyle=line_styles[self.Gvec[i] - 1],
                        label=f"Reward {i + 1}" if i < 5 else "_nolegend_")

            ax.set_title("Reward Curves")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Reward Value")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)

            plt.tight_layout()
            plt.savefig(f"reward_curves.png")
            # plt.show()
            plt.close()

    def dump_coeffs(self):
        self.Xi = np.concatenate((self.ag, self.Phi1g,
                                  self.D2g, self.zeta1g, self.zeta2g, self.Phi2g), axis=-1)
        Smodel_params = {"ag": self.ag, "zeta1g": self.zeta1g, "zeta2g": self.zeta2g,
                         "D1g": self.D1g, "D2g": self.D2g, "Phi1g": self.Phi1g, "Phi2g": self.Phi2g,
                         "Xig": self.Xi}
        Ymodel_params = {"bg": self.bg, "gamma1g": self.gamma1g, "gamma2g": self.gamma2g, "betag":self.betag}
        network_params = {"Gvec": self.Gvec, "W":self.W}

        Smodel_params = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for
                         key, value in Smodel_params.items()}
        Ymodel_params = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for
                         key, value in Ymodel_params.items()}
        network_params = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for
                          key, value in network_params.items()}

        return Smodel_params, Ymodel_params, network_params


