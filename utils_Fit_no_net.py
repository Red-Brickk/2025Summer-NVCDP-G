import numpy as np


def Fit4S(res_dict, S0mat, Stensor, Ztensor, Atensor, Wmat):
    theta_hat = res_dict["Theta_hat4S"]
    coeff_hat = res_dict["coeff_hat4S"]
    Gvec = res_dict["Gvec"]
    coeff_hat_by_ind = coeff_hat[Gvec]
    NN, TT, PP, QQ1 = coeff_hat_by_ind.shape
    _, _, KK, _ = Ztensor.shape
    # features (NN, TT, PP, QQ1)

    # S0mat (NN, 1, PP, 1)
    # Stensor (NN, TT, PP, 1)
    # ST1 (NN, TT, PP, 1)
    # ST1bar (NN, TT, PP, 1)
    # Ztensor (NN, TT, KK, 1)
    # Atensor (NN, TT, 1, 1)
    # feature (NN, TT, PP, QQ1)
    ST1 = np.concatenate((S0mat, Stensor[:, :TT-1, :, :]), axis=1)
    ST1bar = np.einsum('ij,jklm->iklm', Wmat, ST1)
    Abar = np.einsum("ij, jklm->iklm", Wmat, Atensor)
    features = np.concatenate(
        (np.ones((NN, TT, PP, 1)),
         np.tile(ST1, (1, 1, 1, PP)),
         ST1bar,
         np.tile(Atensor, reps=(1, 1, PP, 1)),
         np.tile(Abar, reps=(1, 1, PP, 1)),
         np.tile(np.reshape(Ztensor, newshape=(NN, TT, 1, KK)), reps=(1, 1, PP, 1)),
        ), axis=-1
    )

    Gvec_hat = res_dict["Gvec"]
    # temp_inv (NN, TT, PP, NN)
    # temp_mat_mult (NN TT PP 1)
    temp_inv = np.zeros(shape=(NN, TT, PP, NN))
    for tt in range(TT):
        for pp in range(PP):
            temp_inv[:, tt, pp, :] = np.linalg.inv(
                np.eye(NN)
            )
    temp_mat_mult = np.einsum("ntpq, ntpq -> ntp", features, coeff_hat_by_ind)[:,:,:,np.newaxis]
    StensorFit = np.einsum('ntpi,itpm->ntpm', temp_inv, temp_mat_mult)

    return StensorFit



def Fit4Y(res_dict, Stensor, Atensor, Wmat):
    theta_hat = res_dict["Theta_hat4Y"]
    coeff_hat = res_dict["coeff_hat4Y"]
    Gvec = res_dict["Gvec"]
    coeff_hat_by_ind = coeff_hat[Gvec]
    NN, TT, PP, _ = Stensor.shape
    # features (NN, TT, PP, QQ1)

    # Stensor (NN, TT, PP, 1)
    # Ztensor (NN, TT, KK, 1)
    # Atensor (NN, TT, 1, 1)
    # feature (NN, TT, PP3, 1)
    Abar = np.einsum("ij, jklm->iklm", Wmat, Atensor)
    features = np.concatenate((
         np.ones((NN, TT, 1, 1)),
         Stensor,
         Atensor,
         Abar), axis=-2
    )

    # coeff (NN, TT, 1, PP3)
    YtensorFit = np.einsum('ntip, ntip->ntp', features, coeff_hat_by_ind)

    return YtensorFit