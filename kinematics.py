"""
analogy.kinematics
==================
Plate-velocity fitting from displacement data and strain-rate / vorticity
calculation from PIV velocity fields.
"""
import numpy as np
import pandas as pd

from .utils import line_fit

__all__ = [
    'plate_velocity_calc',
    'strain_component_calc',
    'vorticity',
]


def plate_velocity_calc(xmin,xmax,ymin,ymax,xleft,ybottom,selected_frames,
                          files,numskippedrows=3):
    '''
    Calculate the mean velocity of a selected region of PIV velocity data.
    Used to calcuate the velocity of a region of a sandbox model moving at a
        constant velocity, usually the non-deforming/purely-translating
        incoming plate material.
    Also calculate slope-intercept form of fit
    Spatial units are in pixels.
    '''
    # load one file to get index of rows within specified area
    data = pd.read_csv(files[0],skiprows=numskippedrows,usecols = [0,1,2,3],\
                                   names=['x','y','u','v'])

    # shift data to image coordinates
    data.x = data.x - xleft
    data.y = ybottom - data.y
    data.v = -data.v

    # get index of lines within defined area
    goodrows=(data.x>=xmin)&(data.x<=xmax)&(data.y>=ymin)&(data.y<=ymax)

    means=[]
    for i,sfr in enumerate(selected_frames):
        data = pd.read_csv(pfiles[i],skiprows=numskippedrows,
                           usecols = [0,1,2,3],
                           names=['x','y','u','v']).dropna(how='any')
        window = data.loc[goodrows]
        means.append([sfr*step,np.nanmean(window.u),np.nanmean(window.v)])
    means=pd.DataFrame(means, columns=['frame','mean_u','mean_v'])
    return means, line_fit(means.frame, means.mean_u)


def strain_component_calc(x,y,u,v,spacing=8):
    '''
    Calculate strain and its variants, as well as vorticity from PIVlab
        x- and y-velocity fields.
    Vorticity calcuation based on Meynart [1983], Omega equation, and other
        continuum mechanics texts (Fung, Eringen, etc...)
    '''
    if x.shape[1] >= x.shape[0]:
        # normal displacement gradients
        dUdX = np.gradient(u,spacing)[1]
        dVdY = np.gradient(v,spacing)[0]
        # shear displacement gradients
        dUdY = np.gradient(u,spacing)[0]
        dVdX = np.gradient(v,spacing)[1]
        # vorticity
        vor = dVdX - dUdY
        # shear strain
        exy = 0.5 * (dUdY + dVdX)
        # first invariant of strain [volumetric strain]
        firInv = dUdX + dVdY
        # second invariant of strain ['strain magnitude']
        secInv = exy**2 - (dUdX * dVdY)
        # maximum shear strain
        maxss = np.sqrt((0.5*(dUdX+dVdY))**2 + exy**2)
        # max compressive strain orientation
        theta_p = 0.5 * np.arctan2(2 * exy, dUdX - dVdY)
        return vor, exy, firInv, secInv, maxss, theta_p
    else:
        raise ValueError('Array input x dimension must be equal to or ' + \
                         'larger than y dimension, transpose arrays into ' +\
                         'image coordinates.')


def vorticity(x,y,u,v,dx=[],dy=[],method='leastsq',pad=False):
    '''
    Calcuate vorticity using differing methods.
    Expects inputs in [M,N] arrays.
    Expects these arrays to be in image coordinates.

    Originally written by Kristian Sveen [jks@math.uio.no] in 2001, modified
        for Python by Phiala Thouvenin [phialathouvenin@gmail.com] in 2018
    '''

    if pad:
        vor = np.zeros(x.shape)
        x = np.pad(x, pad_width=2, mode='constant', constant_values=0)
        y = np.pad(y, pad_width=2, mode='constant', constant_values=0)
        u = np.pad(u, pad_width=2, mode='constant', constant_values=0)
        v = np.pad(v, pad_width=2, mode='constant', constant_values=0)
    else:
        vor = np.zeros(x.shape)
    if method == 'gradient': # Meynart [1983], Middleton and Wilcock [1994,pg315], etc...
        dUdY = np.gradient(u)[0]
        dVdX = np.gradient(v)[1]
        vor = dVdX - dUdY
    if method == 'leastsq':
        for i in np.arange(3,x.shape[1] - 2,1).reshape(-1):
            for j in np.arange(3,x.shape[0]  - 2,1).reshape(-1):
                vor[j - 2,i - 2] = -(np.dot(2,v[j,i + 2]) + v[j,i + 1] \
                   - v[j,i - 1] - np.dot(2,v[j,i - 2])) / (np.dot(10,dx)) \
                   + (np.dot(2,u[j + 2,i]) + u[j + 1,i] - u[j - 1,i] - \
                      np.dot(2,u[j - 2,i])) / (np.dot(10,dy))
    if method == 'richardson':
        for i in np.arange(3,x.shape[1] - 2,1).reshape(-1):
            for j in np.arange(3,x.shape[0]  - 2,1).reshape(-1):
                vor[j - 2,i - 2] = -(- v[j,i + 2] + np.dot(8,v[j,i + 1]) \
                   - np.dot(8,v[j,i - 1]) + v[j,i - 2]) / (np.dot(12,dx)) + \
                   (-u[j + 2,i] + np.dot(8,u[j + 1,i]) - np.dot(8,u[j - 1,i]) \
                    + u[j - 2,i]) / (np.dot(12,dy))
    if pad:
        return vor[3:x.shape[0]-2,3:x.shape[1]-2]
    else:
        return vor
