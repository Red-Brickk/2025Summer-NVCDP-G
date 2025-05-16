import numpy as np
import matplotlib.pyplot as plt
from DGP import StateEvolve


class Environment:
    def __init__(self, GG, NN, TT, KK, PP, W, network_params,
                 Smodel_params, Ymodel_params, Ztensor, S0tensor,
                 eps_sigma4S, eps_sigma4Y):
        """
        Smodel_params: 状态转移的线性模型参数
        Ymodel_params: 奖励函数的线性模型参数
        S0: 初始状态张量，形状为 (NN, 1, PP, 1)
        TT: 总时间步数
        """
        self.GG = GG
        self.NN = NN
        self.PP = PP
        self.TT = TT
        self.KK = KK
        self.t = 0

        self.ag = np.array(Smodel_params["ag"])
        self.zeta1g = np.array(Smodel_params["zeta1g"])
        self.zeta2g = np.array(Smodel_params["zeta2g"])
        self.D1g = np.array(Smodel_params["D1g"])
        self.D2g = np.array(Smodel_params["D2g"])
        self.Phi1g = np.array(Smodel_params["Phi1g"])
        self.Phi2g = np.array(Smodel_params["Phi2g"])

        self.bg = np.array(Ymodel_params["bg"])
        self.gamma1g = np.array(Ymodel_params["gamma1g"])
        self.gamma2g = np.array(Ymodel_params["gamma2g"])
        self.betag = np.array(Ymodel_params["betag"])

        self.Gvec = np.array(network_params["Gvec"])
        self.W = np.array(W)

        self.Xi = np.concatenate((self.ag, self.Phi1g,
                                  self.D2g, self.zeta1g, self.zeta2g, self.Phi2g), axis=-1)

        # 初始化状态张量和奖励张量
        self.Stensor = np.zeros((self.NN, TT, self.PP, 1))
        self.Ytensor = np.zeros((self.NN, TT, 1))
        self.Ztensor = Ztensor

        # np.random.seed(self.seed + 100)
        self.Stensor[:, 0, :, :] = S0tensor

        self._generate_eps_mat4state(eps_sigma4S)
        self._generate_eps_mat4reward(eps_sigma4Y)


    # Initial state
    def _generate_exogenous_variates(self):
        # np.random.seed(self.seed + 300)
        self.Ztensor = np.random.normal(loc=0, scale=0.1, size=(self.NN, self.TT, self.KK, 1))

    # noise tensor
    def _generate_eps_mat4state(self, eps_S=0.01):
        # np.random.seed(self.seed + 200)
        self.epsmat4state = np.random.normal(loc=0, scale=eps_S, size=(self.NN, self.TT, self.PP, 1))

    # noise tensor
    def _generate_eps_mat4reward(self, eps_Y=0.01):
        # np.random.seed(self.seed + 201)
        self.epsmat4reward = np.random.normal(loc=0, scale=eps_Y, size=(self.NN, self.TT, 1, 1))

    def observe(self):
        """返回当前和前一时刻的状态张量"""
        if self.t == 0:
            # t = 0 时，前一状态等于当前状态
            obs = {"S_past": self.Stensor[:, 0, :, :], "S_current": self.Stensor[:, 0, :, :]}
        else:
            obs = {
                "S_past": self.Stensor[:, 0, :, :], "S_current": self.Stensor[:, self.t, :, :]
            }
        return obs

    def interact(self, A):
        """
        输入：
        A: 动作张量，形状为 (NN, 1)，数值 0 或 1

        作用：
        - 更新状态张量：S[t+1] = S[t] + A * W （W是某种线性映射）
        - 更新奖励张量：Y[t] = S[t]^T * Ymodel_params + 噪声
        """
        if self.t >= self.TT:
            raise ValueError("已经达到最大时间步数")

        Sold = self.Stensor[:, self.t:self.t + 1, :, :]
        A = A.reshape(self.NN, 1, 1, 1)  # 扩展形状 (NN, 1, 1)
        Abartt = np.einsum("mn, nijk->mijk", self.W, A)

        for pp in range(self.PP):
            self.Stensor[:, self.t:self.t + 1, pp:pp + 1, :] = StateEvolve(
                NN=self.NN, Gvec=self.Gvec, W=self.W,
                D1g=self.D1g, Sold=Sold, Xi=self.Xi,
                Ztensor=self.Ztensor,
                Att=A, Abartt=Abartt,
                epsmat4state=self.epsmat4state, tt=self.t, pp=pp, PP=self.PP, KK=self.KK)

        temp_feature = np.concatenate((np.ones(shape=(self.NN, 1, 1, 1)),
                                       self.Stensor[:, self.t:(self.t+1),:,:], A, Abartt), axis=2
                                      )
        temp_coeffs = np.concatenate((self.bg[self.Gvec][:, self.t:(self.t+1),:,:],
                                      self.betag[self.Gvec][:, self.t:(self.t+1),:,:],
                                      self.gamma1g[self.Gvec][:, self.t:(self.t+1),:,:],
                                      self.gamma2g[self.Gvec][:, self.t:(self.t+1),:,:],
                                      ), axis=2
                                     )
        self.Ytensor[:, self.t:(self.t+1), :] = (np.einsum("ijkl, ijkm->ijlm",
                                           temp_feature, temp_coeffs) +
                                                 self.epsmat4reward[:,self.t:(self.t+1),:,:])[:, :, :, 0]

        # 时间步 +1
        self.t += 1

    def dump(self):
        return self.Stensor, self.Ytensor

    def plot_time_series(self, vis_list=None, folder="", mode=""):
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
                plt.savefig(folder+mode+f"state_curves_pp_{pp_index}.png")
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
                plt.savefig(folder+mode+f"exogenous_variable_curves_kk_{kk_index}.png")
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
            plt.savefig(folder+mode+f"reward_curves.png")
            # plt.show()
            plt.close()