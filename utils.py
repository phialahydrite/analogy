"""
analogy.utils
=============
General-purpose numeric/array helpers shared across the package: filling
NaNs by nearest-neighbor, gridding/degridding spatial column data,
nearest-neighbor lookups, curve/line fitting, discretized colormaps,
monotonicity checks, and time-axis resets.
"""
import itertools
import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from scipy.optimize import curve_fit
from scipy.spatial import KDTree

__all__ = [
    'fill',
    'gridize',
    'degridize',
    'find_comp',
    'do_kdtree',
    'first_closest',
    'last_closest',
    'first_last',
    'k_closest',
    'reject_outliers',
    'line_fit',
    'cmap_discretize',
    'div_proportional_colormap',
    'masked_array_shift_zero',
    'to_slices',
    'subtract_staggered',
    'strictly_increasing',
    'strictly_decreasing',
    'non_increasing',
    'non_decreasing',
    'angle',
    'find_roots',
    'reset_time',
    'reset_time_2d',
]


def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where
                 data value shoud be replaced.
                 If None (defaut), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.

    Modified From:
        https://stackoverflow.com/questions/3662361/fill-in-missing-values-
                    with-nearest-neighbour-in-python-numpy-masked-arrays
    """
    if invalid is None:
        invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]


def gridize(x,y,data):
    '''
    Take spatially-referenced column data and turn it into 2D array
    '''
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    new = np.empty(x_vals.shape + y_vals.shape)
    new.fill(np.nan) # or whatever yor desired missing data flag is
    new[x_idx, y_idx] = data
    return new


def degridize(data):
    '''
    Take 2D array and then turn it into spatially-referenced column data
    '''
    x_size, y_size = data.shape[1], data.shape[0]
    columns=[(x,y,data[y,x]) for x in range(x_size) for y in range(y_size)]
    return np.array(columns)


def find_comp(array,index,replace=0.):
    '''
    Find [index] in [array], returning replace instead of erroring, used with
        thickness data to return zero outside of eroded area

    Parameters
    ----------
    array : any
        Any array you want to find [index] in.
    index : int
        desired index.
    replace : any, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    sel : any
        returned value from [array], or [replace] if not present.

    '''
    try:
        sel = array[int(index)]
    except:
        sel = replace
    return sel


def do_kdtree(target_points,orig_points,find_num):
    '''
    Simple wraparound to compute k-d Tree for k-dimensional data

    Returns
    -------
    dist
        -Euclidean distance between points
    indicies
        -indicies of matched closest points
    '''
    mytree = KDTree(target_points,balanced_tree=False,compact_nodes=False)
    dist, indicies = mytree.query(list(orig_points),k=find_num)
    return dist, indicies


def first_closest(sample, pivot, k):
    '''
    Find k-nearest points from pivot in sample, pick first-ocurring instance
    '''
    nearest = sorted(enumerate(sample), key=lambda nv: abs(nv[1] - pivot))[:k]
    return min([n[0] for n in nearest])


def last_closest(sample, pivot, k):
    '''
    Find k-nearest points from pivot in sample, pick last-ocurring instance
    '''
    nearest = sorted(enumerate(sample), key=lambda nv: abs(nv[1] - pivot))[:k]
    return max([n[0] for n in nearest])


def first_last(sample,pivot):
    '''


    Parameters
    ----------
    sample : array
        array of values to examine.
    pivot : int or float
        value to determine where sample exceeds pivot.

    Returns
    -------
    first_value, int
        first instance of exceeding pivot in sample.
    last_value, int
        last instance of exceeding pivot in sample.
    '''
    exceed = np.argwhere(sample>=pivot)
    if exceed.size > 1:#need two intercepts / concavity
        return exceed[0], exceed[-1]
    else:
        return [],[]


def k_closest(sample, pivot, k):
    '''
    Find k-nearest points from pivot in sample
    '''
    return sorted(enumerate(sample), key=lambda nv: abs(nv[1] - pivot))[:k]


def reject_outliers(data, m=2):
    ind = abs(data - np.mean(data)) < m * np.std(data)
    return ind, data[ind]


def line_fit(x,y):
    '''
    Return slope [m] and y-intercept [b] of a least-squares fit line to [x,y],
        using least squares
    '''
    # straight line, y=f(x)
    def f(x, A, B):
        return A*x + B
    # only use non NaNs
    idx = np.isfinite(x) & np.isfinite(y)
    # fit curve
    m,b = curve_fit(f, x[idx], y[idx])[0]
    return m,b


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    # Return colormap object.
    return colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


def div_proportional_colormap(cmap='coolwarm', vmin=-1., vmax=1.,
                              vcenter=0., nsteps=512, cmap_name='my_cmap'):
    """ Given a diverging colormap, this returns a diverging
        colormap with color ranges on either side proportional to
        the extent of vmax, vmin w.r.t. vcenter i.e. vmax-vcenter, vcenter-vmin
    """
    if isinstance(cmap, str):
	    cmap = plt.cm.get_cmap(cmap)

    max_delta_v = max(vcenter-vmin, vmax-vcenter)
    cmap_min = 0.5-0.5*(vcenter-vmin)/max_delta_v
    cmap_max = 0.5+0.5*(vmax-vcenter)/max_delta_v
    my_colors = cmap(np.linspace(cmap_min, cmap_max, nsteps))
    my_cmap = colors.LinearSegmentedColormap.from_list(cmap_name, my_colors)
    return my_cmap


def masked_array_shift_zero(array):
    '''
    Take masked (i.e., background replaced with NaN) wedge data of any type,
        and shift it so that the surface lines up with the abscissa.
    Used with extracting geothermal gradients.
    '''
    def shift(l,n):
        return itertools.islice(itertools.cycle(l),n,n+len(l))
    array=array.copy()
    for i in np.arange(array.shape[1]):
        len_nan = len(array[:,i][np.isnan(array[:,i])])
        if len_nan > 0:
            array[:,i] = list(shift(array[:,i],len_nan))
    return array


def to_slices(slice_list):
    array = np.zeros((len(slice_list),
                      slice_list[0].shape[0],
                      slice_list[0].shape[1]))
    for i,s in enumerate(slice_list):
        array[i,:,:] = s
    return array


def subtract_staggered(A, B):
    '''
    Take two arrays that share the same origin and subtract them, even if
        dimensions don't match

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    Returns
    -------
    sub : TYPE
        DESCRIPTION.

    '''
    shape_A = A.shape
    shape_B = B.shape
    C = np.vstack((shape_A,shape_B)).min(axis=0)
    sub = A[:C[0],:C[1]] - B[:C[0],:C[1]]
    return sub


def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))


def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))


def angle(directions):
    """
    Return the angle between vectors
    """
    vec2 = directions[1:]
    vec1 = directions[:-1]

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)
    return np.arccos(cos)


def find_roots(x, y):
    '''
    Zero-crossings in a function f(x,y), used in structural_divide
    https://stackoverflow.com/questions/46909373/
        how-to-find-the-exact-intersection-of-a-curve-as-np-array-with-y-0
    '''
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)


def reset_time(time_array,index_array,zero_start=False):
    '''

    Parameters
    ----------
    time_array : array containing the time at every frame
    index_array : locations at which time is reset, starting at zero
    zero_start : optional zeroing out of time before first index_array event

    Returns
    -------
    time_reset : array containing new times with increase between index_array
        events

    '''
    time_reset = np.zeros(len(time_array))
    for i in index_array:
        time_reset[i:] = time_array[0:len(time_array)-i]
    if zero_start:
        time_reset[:index_array[1]] = 0
    return time_reset


def reset_time_2d(time_array,index_array,zero_start=False):
    '''

    Parameters
    ----------
    time_array : array containing the time at every frame
    index_array : locations at which time is reset, starting at zero
    zero_start : optional zeroing out of time before first index_array event

    Returns
    -------
    time_reset : array containing new times with increase between index_array
        events

    '''
    time_reset = np.zeros(time_array.size)
    for i in index_array:
        time_reset[i:] = time_array[:,0:len(time_array)-i]
    if zero_start:
        time_reset[:,:index_array[1]] = 0
    return time_reset
