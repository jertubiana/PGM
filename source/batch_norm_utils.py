
"""
 Copyright 2020 - by Jerome Tubiana (jertubiana@gmail.com)
     All rights reserved

     Permission is granted for anyone to copy, use, or modify this
     software for any uncommercial purposes, provided this copyright
     notice is retained, and note is made of any changes that have
     been made. This software is distributed without any warranty,
     express or implied. In no event shall the author or contributors be
     liable for any damage arising out of the use of this software.

     The publication of research using this software, modified or not, must include
     appropriate citations to:
"""

import numpy as np
from utilities import erf_times_gauss, average, average_product, covariance



def get_cross_derivatives_Gaussian(V_pos, psi_pos, hlayer, n_cv, weights=None):
    db_dw = average(V_pos, c=n_cv, weights=weights)
    da_db = np.zeros(hlayer.N)
    WChat = covariance(psi_pos, V_pos, weights=weights, c1=1, c2=n_cv)
    var_e = average(psi_pos**2, weights=weights) - \
        average(psi_pos, weights=weights)**2
    if n_cv > 1:
        da_dw = 2 / np.sqrt(1 + 4 * var_e)[:, np.newaxis, np.newaxis] * WChat
    else:
        da_dw = 2 / np.sqrt(1 + 4 * var_e)[:, np.newaxis] * WChat
    return db_dw, da_db, da_dw


def get_cross_derivatives_ReLU(V_pos, psi_pos, hlayer, n_cv, weights=None):
    db_dw = average(V_pos, c=n_cv, weights=weights)
    a = hlayer.gamma[np.newaxis, :]
    theta = hlayer.delta[np.newaxis, :]
    b = hlayer.theta[np.newaxis, :]

    psi = psi_pos

    psi_plus = (-(psi - b) + theta) / np.sqrt(a)
    psi_minus = ((psi - b) + theta) / np.sqrt(a)

    Phi_plus = erf_times_gauss(psi_plus)
    Phi_minus = erf_times_gauss(psi_minus)

    p_plus = 1 / (1 + Phi_minus / Phi_plus)
    p_minus = 1 - p_plus

    e = (psi - b) - theta * (p_plus - p_minus)
    v = p_plus * p_minus * (2 * theta / np.sqrt(a)) * \
        (2 * theta / np.sqrt(a) - 1 / Phi_plus - 1 / Phi_minus)

    dpsi_plus_dpsi = -1 / np.sqrt(a)
    dpsi_minus_dpsi = 1 / np.sqrt(a)
    dpsi_plus_dtheta = 1 / np.sqrt(a)
    dpsi_minus_dtheta = 1 / np.sqrt(a)
    dpsi_plus_da = -1.0 / (2 * a) * psi_plus
    dpsi_minus_da = -1.0 / (2 * a) * psi_minus

    d2psi_plus_dadpsi = 0.5 / np.sqrt(a**3)
    d2psi_plus_dthetadpsi = 0
    d2psi_minus_dadpsi = -0.5 / np.sqrt(a**3)
    d2psi_minus_dthetadpsi = 0

    dp_plus_dpsi = p_plus * p_minus * \
        ((psi_plus - 1 / Phi_plus) * dpsi_plus_dpsi -
         (psi_minus - 1 / Phi_minus) * dpsi_minus_dpsi)
    dp_plus_dtheta = p_plus * p_minus * \
        ((psi_plus - 1 / Phi_plus) * dpsi_plus_dtheta -
         (psi_minus - 1 / Phi_minus) * dpsi_minus_dtheta)
    dp_plus_da = p_plus * p_minus * \
        ((psi_plus - 1 / Phi_plus) * dpsi_plus_da -
         (psi_minus - 1 / Phi_minus) * dpsi_minus_da)

    d2p_plus_dpsi2 = -(p_plus - p_minus) * p_plus * p_minus * ((psi_plus - 1 / Phi_plus) * dpsi_plus_dpsi - (psi_minus - 1 / Phi_minus) * dpsi_minus_dpsi)**2 \
        + p_plus * p_minus * ((dpsi_plus_dpsi)**2 * (1 + (psi_plus - 1 / Phi_plus) / Phi_plus) - (
            dpsi_minus_dpsi)**2 * (1 + (psi_minus - 1 / Phi_minus) / Phi_minus))

    d2p_plus_dadpsi = -(p_plus - p_minus) * ((psi_plus - 1 / Phi_plus) * dpsi_plus_dpsi - (psi_minus - 1 / Phi_minus) * dpsi_minus_dpsi) * (dp_plus_da)\
        + p_plus * p_minus * ((dpsi_plus_dpsi * dpsi_plus_da) * (1 + (psi_plus - 1 / Phi_plus) / Phi_plus) - (dpsi_minus_dpsi * dpsi_minus_da) * (1 + (psi_minus - 1 / Phi_minus) / Phi_minus)
                              + (d2psi_plus_dadpsi) * (psi_plus - 1 / Phi_plus) - (d2psi_minus_dadpsi) * (psi_minus - 1 / Phi_minus))

    d2p_plus_dthetadpsi = -(p_plus - p_minus) * ((psi_plus - 1 / Phi_plus) * dpsi_plus_dpsi - (psi_minus - 1 / Phi_minus) * dpsi_minus_dpsi) * (dp_plus_dtheta)\
        + p_plus * p_minus * ((dpsi_plus_dpsi * dpsi_plus_dtheta) * (1 + (psi_plus - 1 / Phi_plus) / Phi_plus) - (dpsi_minus_dpsi * dpsi_minus_dtheta) * (1 + (psi_minus - 1 / Phi_minus) / Phi_minus)
                              + (d2psi_plus_dthetadpsi) * (psi_plus - 1 / Phi_plus) - (d2psi_minus_dthetadpsi) * (psi_minus - 1 / Phi_minus))

# dlogZ_dpsi = (p_plus * (psi_plus - 1 / Phi_plus) * dpsi_plus_dpsi +
#               p_minus * (psi_minus - 1 / Phi_minus) * dpsi_minus_dpsi)
# dlogZ_dtheta = (p_plus * (psi_plus - 1 / Phi_plus) * dpsi_plus_dtheta +
#                 p_minus * (psi_minus - 1 / Phi_minus) * dpsi_minus_dtheta)
# dlogZ_da = (p_plus * (psi_plus - 1 / Phi_plus) * dpsi_plus_da +
#             p_minus * (psi_minus - 1 / Phi_minus) * dpsi_minus_da)

    de_dpsi = (1 + v)
    de_db = -de_dpsi
    de_da = 2 * (- theta) * dp_plus_da
    de_dtheta = -(p_plus - p_minus) + 2 * (- theta) * dp_plus_dtheta

    dv_dpsi = 2 * (-theta) * d2p_plus_dpsi2
    dv_db = -dv_dpsi

    dv_da = + 2 * (- theta) * d2p_plus_dadpsi

    dv_dtheta = - 2 * dp_plus_dpsi \
        + 2 * (- theta) * d2p_plus_dthetadpsi

    var_e = average(e**2, weights=weights) - average(e, weights=weights)**2
    mean_v = average(v, weights=weights)

    dmean_v_da = average(dv_da, weights=weights)
    dmean_v_db = average(dv_db, weights=weights)
    dmean_v_dtheta = average(dv_dtheta, weights=weights)

    dvar_e_da = 2 * (average(e * de_da, weights=weights) -
                     average(e, weights=weights) * average(de_da, weights=weights))
    dvar_e_db = 2 * (average(e * de_db, weights=weights) -
                     average(e, weights=weights) * average(de_db, weights=weights))
    dvar_e_dtheta = 2 * (average(e * de_dtheta, weights=weights) -
                         average(e, weights=weights) * average(de_dtheta, weights=weights))

    tmp = np.sqrt((1 + mean_v)**2 + 4 * var_e)
    da_db = (dvar_e_db + 0.5 * dmean_v_db * (1 + mean_v + tmp)) / \
        (tmp - dvar_e_da - 0.5 * dmean_v_da * (1 + mean_v + tmp))
    da_dtheta = (dvar_e_dtheta + 0.5 * dmean_v_dtheta * (1 + mean_v + tmp)
                 ) / (tmp - dvar_e_da - 0.5 * dmean_v_da * (1 + mean_v + tmp))

    dmean_v_dw = average_product(
        dv_dpsi, V_pos, c1=1, c2=n_cv, weights=weights)

    if n_cv > 1:
        dvar_e_dw = 2 * (average_product(e * de_dpsi, V_pos, c1=1, c2=n_cv, weights=weights) - average(e, weights=weights)
                         [:, np.newaxis, np.newaxis] * average_product(de_dpsi, V_pos, c1=1, c2=n_cv, weights=weights))
        da_dw = (dvar_e_dw + 0.5 * dmean_v_dw * (1 + mean_v + tmp)[:, np.newaxis, np.newaxis]) / (
            tmp - dvar_e_da - 0.5 * dmean_v_da * (1 + mean_v + tmp))[:, np.newaxis, np.newaxis]

    else:
        dvar_e_dw = 2 * (average_product(e * de_dpsi, V_pos, c1=1, c2=1, weights=weights) - average(
            e, weights=weights)[:, np.newaxis] * average_product(de_dpsi, V_pos, c1=1, c2=1, weights=weights))
        da_dw = (dvar_e_dw + 0.5 * dmean_v_dw * (1 + mean_v + tmp)[:, np.newaxis]) / (
            tmp - dvar_e_da - 0.5 * dmean_v_da * (1 + mean_v + tmp))[:, np.newaxis]

    return db_dw, da_db, da_dtheta, da_dw


def get_cross_derivatives_ReLU_plus(V_pos, psi_pos, hlayer, n_cv, weights=None):
    db_dw = average(V_pos, c=n_cv, weights=weights)

    a = hlayer.gamma[np.newaxis, :]
    b = hlayer.theta[np.newaxis, :]

    psi = psi_pos
    psi_plus = -(psi - b) / np.sqrt(a)

    Phi_plus = erf_times_gauss(psi_plus)

    e = (psi - b) + np.sqrt(a) / Phi_plus
    v = (psi_plus - 1 / Phi_plus) / Phi_plus

    dpsi_plus_dpsi = -1 / np.sqrt(a)
    dpsi_plus_da = -1.0 / (2 * a) * psi_plus

    de_dpsi = 1 + v
    de_db = -de_dpsi
    de_da = np.sqrt(a) * (1.0 / (2 * a * Phi_plus) -
                          (psi_plus - 1 / Phi_plus) / Phi_plus * dpsi_plus_da)

    dv_dpsi = dpsi_plus_dpsi * \
        (1 + psi_plus / Phi_plus - 1 / Phi_plus **
         2 - (psi_plus - 1 / Phi_plus)**2) / Phi_plus
    dv_db = -dv_dpsi
    dv_da = dpsi_plus_da * (1 + psi_plus / Phi_plus - 1 /
                            Phi_plus**2 - (psi_plus - 1 / Phi_plus)**2) / Phi_plus

    var_e = average(e**2, weights=weights) - average(e, weights=weights)**2
    mean_v = average(v, weights=weights)

    dmean_v_da = average(dv_da, weights=weights)
    dmean_v_db = average(dv_db, weights=weights)

    dvar_e_da = 2 * (average(e * de_da, weights=weights) -
                     average(e, weights=weights) * average(de_da, weights=weights))
    dvar_e_db = 2 * (average(e * de_db, weights=weights) -
                     average(e, weights=weights) * average(de_db, weights=weights))

    tmp = np.sqrt((1 + mean_v)**2 + 4 * var_e)
    denominator = (tmp - dvar_e_da - 0.5 * dmean_v_da * (1 + mean_v + tmp))
    denominator = np.maximum(denominator, 0.5)  # For numerical stability.

    da_db = (dvar_e_db + 0.5 * dmean_v_db * (1 + mean_v + tmp)) / denominator

    dmean_v_dw = average_product(
        dv_dpsi, V_pos, c1=1, c2=n_cv, weights=weights)

    if n_cv > 1:
        dvar_e_dw = 2 * (average_product(e * de_dpsi, V_pos, c1=1, c2=n_cv, weights=weights) - average(e, weights=weights)
                         [:, np.newaxis, np.newaxis] * average_product(de_dpsi, V_pos, c1=1, c2=n_cv, weights=weights))
        da_dw = (dvar_e_dw + 0.5 * dmean_v_dw * (1 + mean_v + tmp)
                 [:, np.newaxis, np.newaxis]) / denominator[:, np.newaxis, np.newaxis]

    else:
        dvar_e_dw = 2 * (average_product(e * de_dpsi, V_pos, c1=1, c2=1, weights=weights) - average(
            e, weights=weights)[:, np.newaxis] * average_product(de_dpsi, V_pos, c1=1, c2=1, weights=weights))
        da_dw = (dvar_e_dw + 0.5 * dmean_v_dw * (1 + mean_v + tmp)
                 [:, np.newaxis]) / denominator[:, np.newaxis]

    return db_dw, da_db, da_dw


# def get_cross_derivatives_dReLU(V_pos,psi_pos, hlayer, n_cv,weights=None):
#     # a = 2.0/(1.0/hlayer.a_plus + 1.0/hlayer.a_minus)
#     # eta = 0.5* (a/hlayer.a_plus - a/hlayer.a_minus)
#     # theta = (1.-eta**2)/2. * (hlayer.theta_plus+hlayer.theta_minus)
#     # b = (1.+eta)/2. * hlayer.theta_plus - (1.-eta)/2. * hlayer.theta_minus
#     db_dw = average(V_pos,c=n_cv,weights=weights)
#     a = hlayer.a[np.newaxis,:]
#     eta = hlayer.eta[np.newaxis,:]
#     theta = hlayer.theta[np.newaxis,:]
#     b = hlayer.b[np.newaxis,:]

#     psi = psi_pos

#     psi_plus = ( -np.sqrt(1+eta) * (psi-b) + theta )/np.sqrt(a)
#     psi_minus = ( np.sqrt(1-eta) * (psi-b) + theta )/np.sqrt(a)

#     Phi_plus = erf_times_gauss(psi_plus)
#     Phi_minus = erf_times_gauss(psi_minus)

#     Z = Phi_plus * np.sqrt(1+eta) + Phi_minus * np.sqrt(1-eta)

#     p_plus = 1/(1 + (Phi_minus* np.sqrt(1-eta) )/(Phi_plus* np.sqrt(1+eta) ) )
#     nans  = np.isnan(p_plus)
#     p_plus[nans] = 1.0 * (np.abs(psi_plus[nans]) > np.abs(psi_minus[nans]) )
#     p_minus = 1-p_plus

#     e = (psi-b) * (1+eta * (p_plus-p_minus) ) - theta * (np.sqrt(1+eta) * p_plus- np.sqrt(1-eta) * p_minus) + 2*eta*np.sqrt(a)/Z
#     v = eta * (p_plus-p_minus) + p_plus * p_minus * ( (np.sqrt(1+eta) + np.sqrt(1-eta)) *theta/np.sqrt(a)- 2 * eta * (psi-b)/np.sqrt(a)) * ( (np.sqrt(1+eta) + np.sqrt(1-eta))*theta/np.sqrt(a)- 2 * eta * (psi-b)/np.sqrt(a) - np.sqrt(1+eta)/Phi_plus - np.sqrt(1-eta)/Phi_minus)- 2 * eta*e/(np.sqrt(a)*Z)


#     dpsi_plus_dpsi = -np.sqrt((1+eta)/a)
#     dpsi_minus_dpsi = np.sqrt((1-eta)/a)
#     dpsi_plus_dtheta = 1/np.sqrt(a)
#     dpsi_minus_dtheta = 1/np.sqrt(a)
#     dpsi_plus_da = -1.0/(2*a) * psi_plus
#     dpsi_minus_da = -1.0/(2*a) * psi_minus

#     dpsi_plus_deta = -1.0/(2*np.sqrt(a*(1+eta))) * (psi-b)
#     dpsi_minus_deta = -1.0/(2*np.sqrt(a*(1-eta))) * (psi-b)


#     d2psi_plus_dadpsi = 0.5 * np.sqrt((1+eta)/a**3 )
#     d2psi_plus_dthetadpsi = 0
#     d2psi_plus_detadpsi = -0.5/np.sqrt((1+eta)*a)
#     d2psi_minus_dadpsi = -0.5 * np.sqrt((1-eta)/a**3 )
#     d2psi_minus_dthetadpsi = 0
#     d2psi_minus_detadpsi = -0.5/np.sqrt((1-eta)*a)


#     dp_plus_dpsi = p_plus * p_minus * ( (psi_plus-1/Phi_plus) * dpsi_plus_dpsi  - (psi_minus-1/Phi_minus) * dpsi_minus_dpsi )
#     dp_plus_dtheta = p_plus * p_minus * ( (psi_plus-1/Phi_plus) * dpsi_plus_dtheta  - (psi_minus-1/Phi_minus) * dpsi_minus_dtheta )
#     dp_plus_da = p_plus * p_minus * ( (psi_plus-1/Phi_plus) * dpsi_plus_da  - (psi_minus-1/Phi_minus) * dpsi_minus_da )
#     dp_plus_deta = p_plus * p_minus * ( (psi_plus-1/Phi_plus) * dpsi_plus_deta  - (psi_minus-1/Phi_minus) * dpsi_minus_deta + 1/(1-eta**2) )


#     d2p_plus_dpsi2 = -(p_plus-p_minus) * p_plus * p_minus * ( (psi_plus-1/Phi_plus) * dpsi_plus_dpsi  - (psi_minus-1/Phi_minus) * dpsi_minus_dpsi )**2 \
#     + p_plus * p_minus * ( (dpsi_plus_dpsi)**2 *  (1+ (psi_plus-1/Phi_plus)/Phi_plus) - (dpsi_minus_dpsi)**2 * (1+ (psi_minus-1/Phi_minus)/Phi_minus) )


#     d2p_plus_dadpsi = -(p_plus-p_minus) * ( (psi_plus-1/Phi_plus) * dpsi_plus_dpsi  - (psi_minus-1/Phi_minus) * dpsi_minus_dpsi ) * (dp_plus_da)\
#     + p_plus * p_minus * ( (dpsi_plus_dpsi* dpsi_plus_da) *  (1+ (psi_plus-1/Phi_plus)/Phi_plus) - (dpsi_minus_dpsi *dpsi_minus_da) * (1+ (psi_minus-1/Phi_minus)/Phi_minus) \
#     + (d2psi_plus_dadpsi) * (psi_plus-1/Phi_plus) - (d2psi_minus_dadpsi) * (psi_minus-1/Phi_minus) )

#     d2p_plus_dthetadpsi = -(p_plus-p_minus) * ( (psi_plus-1/Phi_plus) * dpsi_plus_dpsi  - (psi_minus-1/Phi_minus) * dpsi_minus_dpsi ) * (dp_plus_dtheta)\
#     + p_plus * p_minus * ( (dpsi_plus_dpsi* dpsi_plus_dtheta) *  (1+ (psi_plus-1/Phi_plus)/Phi_plus) - (dpsi_minus_dpsi *dpsi_minus_dtheta) * (1+ (psi_minus-1/Phi_minus)/Phi_minus) \
#     + (d2psi_plus_dthetadpsi) * (psi_plus-1/Phi_plus) - (d2psi_minus_dthetadpsi) * (psi_minus-1/Phi_minus) )

#     d2p_plus_detadpsi = -(p_plus-p_minus) * ( (psi_plus-1/Phi_plus) * dpsi_plus_dpsi  - (psi_minus-1/Phi_minus) * dpsi_minus_dpsi ) * (dp_plus_deta)\
#     + p_plus * p_minus * ( (dpsi_plus_dpsi* dpsi_plus_deta) *  (1+ (psi_plus-1/Phi_plus)/Phi_plus) - (dpsi_minus_dpsi *dpsi_minus_deta) * (1+ (psi_minus-1/Phi_minus)/Phi_minus) \
#     + (d2psi_plus_detadpsi) * (psi_plus-1/Phi_plus) - (d2psi_minus_detadpsi) * (psi_minus-1/Phi_minus) )


#     dlogZ_dpsi = (p_plus * (psi_plus-1/Phi_plus)* dpsi_plus_dpsi + p_minus * (psi_minus-1/Phi_minus) * dpsi_minus_dpsi )
#     dlogZ_dtheta = (p_plus * (psi_plus-1/Phi_plus)* dpsi_plus_dtheta + p_minus * (psi_minus-1/Phi_minus) * dpsi_minus_dtheta )
#     dlogZ_da = (p_plus * (psi_plus-1/Phi_plus)* dpsi_plus_da + p_minus * (psi_minus-1/Phi_minus) * dpsi_minus_da )
#     dlogZ_deta = (p_plus * (psi_plus-1/Phi_plus)* dpsi_plus_deta + p_minus * (psi_minus-1/Phi_minus) * dpsi_minus_deta  + 0.5 * (p_plus/(1+eta) - p_minus/(1-eta)) )


#     de_dpsi = (1+v)
#     de_db = -de_dpsi
#     de_da = (2*(psi-b) * eta - (np.sqrt(1+eta) + np.sqrt(1-eta)) * theta) * dp_plus_da + eta/(Z*np.sqrt(a)) - 2*eta*np.sqrt(a)/Z * dlogZ_da
#     de_dtheta = -(np.sqrt(1+eta) * p_plus-np.sqrt(1-eta) * p_minus)  + (2*(psi-b) * eta - (np.sqrt(1+eta) + np.sqrt(1-eta)) * theta) * dp_plus_dtheta - 2*eta*np.sqrt(a)/Z * dlogZ_dtheta
#     de_deta = (psi-b) * (p_plus-p_minus) + (2*(psi-b) * eta - (np.sqrt(1+eta) + np.sqrt(1-eta)) * theta) * dp_plus_deta - theta * (p_plus/(2*np.sqrt(1+eta)) + p_minus/(2*np.sqrt(1-eta))) + 2*np.sqrt(a)/Z - 2*eta*np.sqrt(a)/Z * dlogZ_deta


#     dv_dpsi = 4 * eta * dp_plus_dpsi\
#     + ( 2*(psi-b)*eta- (np.sqrt(1+eta) + np.sqrt(1-eta)) * theta) * d2p_plus_dpsi2 \
#     - 2* eta/(np.sqrt(a)*Z) * ( de_dpsi - e*dlogZ_dpsi )

#     dv_db = -dv_dpsi

#     dv_da = eta * 2 * dp_plus_da \
#     + ( 2*(psi-b)*eta- (np.sqrt(1+eta) + np.sqrt(1-eta)) * theta) * d2p_plus_dadpsi \
#     -2 * eta/(Z * np.sqrt(a)) * ( -e/(2*a) - e*dlogZ_da + de_da )

#     dv_dtheta =  2 * eta *  dp_plus_dtheta \
#     - (np.sqrt(1+eta) + np.sqrt(1-eta)) * dp_plus_dpsi \
#     + ( 2*(psi-b)*eta- (np.sqrt(1+eta) + np.sqrt(1-eta)) * theta) * d2p_plus_dthetadpsi \
#     -2 * eta/(Z * np.sqrt(a)) * ( - e*dlogZ_dtheta + de_dtheta )

#     dv_deta = (p_plus-p_minus) \
#     + 2 * eta * dp_plus_deta \
#     + (2 * (psi-b)- theta/2*(1/np.sqrt(1+eta) - 1/np.sqrt(1-eta)) ) * dp_plus_dpsi \
#     + ( 2*(psi-b)*eta- (np.sqrt(1+eta) + np.sqrt(1-eta)) * theta) * d2p_plus_detadpsi \
#     -2 * 1/(Z * np.sqrt(a)) * (e - e*eta*dlogZ_deta + eta*de_deta )


#     var_e = average(e**2,weights=weights) - average(e,weights=weights)**2
#     mean_v = average(v,weights=weights)


#     dmean_v_da = average(dv_da,weights=weights)
#     dmean_v_db = average(dv_db,weights=weights)
#     dmean_v_dtheta = average(dv_dtheta,weights=weights)
#     dmean_v_deta = average(dv_deta,weights=weights)

#     dvar_e_da = 2* (average(e*de_da,weights=weights) -average(e,weights=weights) * average(de_da,weights=weights) )
#     dvar_e_db =  2* (average(e*de_db,weights=weights) -average(e,weights=weights) * average(de_db,weights=weights) )
#     dvar_e_dtheta = 2* (average(e*de_dtheta,weights=weights) -average(e,weights=weights) * average(de_dtheta,weights=weights) )
#     dvar_e_deta = 2* (average(e*de_deta,weights=weights) -average(e,weights=weights) * average(de_deta,weights=weights) )


#     tmp = np.sqrt( (1+mean_v)**2 + 4 * var_e )
#     # denominator = tmp
#     denominator = (tmp - dvar_e_da- 0.5 * dmean_v_da * (1+mean_v+tmp))
#     # denominator = np.maximum( denominator, 0.5) # For numerical stability.


#     da_db = (dvar_e_db + 0.5 * dmean_v_db * (1+ mean_v + tmp) )/denominator
#     da_dtheta = (dvar_e_dtheta + 0.5 * dmean_v_dtheta * (1+ mean_v + tmp) )/denominator
#     da_deta = (dvar_e_deta + 0.5 * dmean_v_deta * (1+ mean_v + tmp) )/denominator


#     dmean_v_dw = average_product(dv_dpsi, V_pos,c1=1,c2=n_cv,weights=weights)

#     if n_cv >1:
#         dvar_e_dw =  2* (average_product(e * de_dpsi ,V_pos, c1=1, c2 = n_cv,weights=weights) - average(e,weights=weights)[:,np.newaxis,np.newaxis] * average_product(de_dpsi,V_pos,c1=1,c2=n_cv,weights=weights) )
#         da_dw = (dvar_e_dw + 0.5 *dmean_v_dw * (1+ mean_v + tmp)[:,np.newaxis,np.newaxis] )/denominator[:,np.newaxis,np.newaxis]

#     else:
#         dvar_e_dw =  2* (average_product(e * de_dpsi ,V_pos, c1=1, c2 = 1,weights=weights) - average(e,weights=weights)[:,np.newaxis] * average_product(de_dpsi,V_pos,c1=1,c2=1,weights=weights) )
#         da_dw = (dvar_e_dw + 0.5 *dmean_v_dw * (1+ mean_v + tmp)[:,np.newaxis] )/denominator[:,np.newaxis]


#     return db_dw,da_db, da_dtheta, da_deta, da_dw
