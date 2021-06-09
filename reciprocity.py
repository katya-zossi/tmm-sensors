"""
Additional scripts required to reproduce the far-field 
radiation patters of polarizable molecules on the 
surface of multilayer complex materials.
"""
from __future__ import division, print_function, absolute_import

from tmm import (coh_tmm, position_resolved)
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import numpy as np


def find_max_field_enhancement(n_list, d_list, lam_vac):
    
    # calculate the electric field enhancent at the surface of the final layer
    layer = len(n_list)-1
    d_in_layer = 0
    
    theta_list = np.linspace(0, np.pi/2, num=901)
    # calculate angle of incidence for maximum field enhancement
    # only consider p-polarized light for excitation
    field_enhancement=[]        
    for theta in theta_list:
        coh_tmm_data = coh_tmm('p', n_list, d_list, theta, lam_vac)
        data = position_resolved(layer, d_in_layer, coh_tmm_data)
        Ep = array([data['Ex'], data['Ey'], data['Ez']])
        magnitude = np.linalg.norm(Ep)    
        field_enhancement.append(magnitude)
      
    theta_opt = theta_list[find_peaks(field_enhancement)[0]]
    enhancement = max(field_enhancement)
    
    return float(theta_opt), enhancement


def dipole_moment(alpha, n_list, d_list, theta_opt, lam_vac):
    
    layer = len(n_list)-1
    d_in_layer = 0 
    
    coh_tmm_data = coh_tmm('p', n_list, d_list, theta_opt, lam_vac)
    data = position_resolved(layer,d_in_layer,coh_tmm_data)

    Ex = data['Ex']
    Ey = data['Ey']
    Ez = data['Ez']

    dipole = alpha * np.array([Ex, Ey, Ez])
    
    return dipole


def radiation_pattern(dipole, n_list, d_list, lam_vac):
    
    # reverse the orders of the layers to calculate the inverse incidence
    # with Lorentz reciprocity theory
    n_list.reverse()
    d_list.reverse() 
    # insert a fictitious water layer to access the exact position of the dipole
    n_list.insert(1, n_list[0])
    d_list.insert(1, 100)

    theta_list = np.linspace(0, np.pi/2, num=401)
    poynting = []

    for theta in theta_list:
        coh_tmm_pdata = coh_tmm('p', n_list, d_list, theta, lam_vac)
        pdata = position_resolved(1, d_list[1], coh_tmm_pdata)
        Ep = np.array([pdata['Ex'], pdata['Ey'], pdata['Ez']])
        
        coh_tmm_sdata = coh_tmm('s', n_list, d_list, theta, lam_vac)
        sdata = position_resolved(1, d_list[1], coh_tmm_sdata)
        Es = np.array([sdata['Ex'], sdata['Ey'], sdata['Ez']])
        
        intensity =  abs(Es.dot(dipole)) ** 2 + abs(Ep.dot(dipole)) ** 2
        
        scale = n_list[0]/(2*376.7)
        poynting.append(intensity * scale)
        
    # Return to original arrays 
    n_list.pop(1)
    d_list.pop(1)
    n_list.reverse()
    d_list.reverse()
    
    return theta_list, poynting


def theta_integral(theta, dipole, n_list, d_list, lam_vac):

    n_list.reverse()
    d_list.reverse() 

    n_list.insert(1, n_list[0])
    d_list.insert(1, 100)

    coh_tmm_pdata = coh_tmm('p', n_list, d_list, theta, lam_vac)
    pdata = position_resolved(1, d_list[1], coh_tmm_pdata)
    Ep = np.array([pdata['Ex'], pdata['Ey'], pdata['Ez']])
    
    coh_tmm_sdata = coh_tmm('s', n_list, d_list, theta, lam_vac)
    sdata = position_resolved(1, d_list[1], coh_tmm_sdata)
    Es = np.array([sdata['Ex'], sdata['Ey'], sdata['Ez']])

    intensity = abs(Es.dot(dipole)) ** 2 + abs(Ep.dot(dipole)) ** 2
    
    # refractive index of the sensing medium/ vacuum impedance 
    scale = n_list[0]/(2*376.7) 
    mag = intensity * scale
    
    # Return to original arrays 
    n_list.pop(1)
    d_list.pop(1)
    n_list.reverse()
    d_list.reverse()
    
    return 2 * np.pi * mag * np.cos(theta)

