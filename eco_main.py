import json
import numpy as np
from utils import plot_time_series, plot_coefficients, plot_residuals
from utils_Fit import Fit4S, Fit4Y
from utils_select_G import selectG
from Estimate import Iter_model


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
S0 = Stensor[:, 0:1, :, :]
Abar = np.einsum("ij, jklm->iklm", W_mat, Action_tensor)

NN, TT, PP, _ = Stensor.shape
J0 = 4

Timevec = [i for i in range(TT)]
Timevec = np.array(Timevec) / TT
splinetype = "b_spline"
IV_type = "1"
lambda_NT = 0.05
mu_G = 0.1
Glist = [1, 2, 3, 4, 5]

# plot_time_series(GDP_growth_tensor[:, :, :,np.newaxis], Gvec=Gvec, mode="Y")
# plot_time_series(State_tensor, Gvec=Gvec, mode="S")

SAVE_MODEL = False

# Ghat = selectG(
#     Timevec=Timevec,
#     S0=S0, Stensor=Stensor,
#     W=W_mat, Ztensor=Ztensor,
#     Ytensor=ur_tensor,
#     A=Action_tensor, Abar=Abar,
#     J0=J0, Glist=Glist, lambda_NT=lambda_NT, mu_G=mu_G
# )
Ghat = 3
res_dict4S, res_dict4Y = Iter_model(
    Timevec=Timevec,
    S0mat=S0, Stensor=Stensor,
    Wmat=W_mat, Ztensor= Ztensor,
    Ytensor=ur_tensor,
    Atensor=Action_tensor, Abar= Abar,
    Gnum=Ghat, J0=J0,
    Gvec_real=None,
    splinetype=splinetype,
    IV_type=IV_type,
    max_iter=100, tol=1e-6, oracle=False
)

Gvec = res_dict4S["Gvec"]

group1 = list(state_names[np.where(Gvec == 0)[0]])
group2 = list(state_names[np.where(Gvec == 1)[0]])
group3 = list(state_names[np.where(Gvec == 2)[0]])
group4 = list(state_names[np.where(Gvec == 3)[0]])

print("group1=", ", ".join(f'"{s}"' for s in state_names[np.where(Gvec == 0)[0]]))
print("group2=", ", ".join(f'"{s}"' for s in state_names[np.where(Gvec == 1)[0]]))
print("group3=:", ", ".join(f'"{s}"' for s in state_names[np.where(Gvec == 2)[0]]))

plot_coefficients(res_dict4S, dgp=None, mode="S", plot_true=False)
plot_coefficients(res_dict4Y, dgp=None, mode="Y", plot_true=False)

StensorFit = Fit4S(res_dict4S, S0mat=S0, Stensor=Stensor,
                   Ztensor=Ztensor, Atensor=Action_tensor, Wmat=W_mat)
YtensorFit = Fit4Y(res_dict4Y, Stensor=Stensor, Atensor=Action_tensor, Wmat=W_mat)

plot_residuals(Tensor=Stensor, Tensor_fit=StensorFit, mode="S")
plot_residuals(Tensor=ur_tensor, Tensor_fit=YtensorFit, mode="Y")

if SAVE_MODEL:
    res_dict4S = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for
                  key, value in res_dict4S.items()}
    res_dict4Y = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for
                  key, value in res_dict4Y.items()}
    W_mat = W_mat.tolist()

    data_to_save = {
        "NN": NN, "TT": TT, "PP": PP, "KK": 4, "W": W_mat,
        "res_dict4S": res_dict4S,
        "res_dict4Y": res_dict4Y
    }
    filename = f"json//Eco_result.json"

    # 将数据保存为 JSON 文件
    with open(filename, "w") as json_file:
        json.dump(data_to_save, json_file, indent=4)
    print(f"Saved data to {filename}.")