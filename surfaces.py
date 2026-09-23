"""
analogy.surfaces
================
Tools that operate on the model's evolving topographic surface: masking
PIV/particle data above or below the surface, extracting the deformation
front (mode, mean, and dv/velocity variants), structural and topographic
divides, surface curvature, and volume change through time.
"""
import numpy as np
import pandas as pd
import peakutils
from scipy.ndimage import filters, binary_dilation, gaussian_filter1d
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

from .utils import line_fit, find_roots

__all__ = [
    'mask_from_surface',
    'mask_from_surface_old',
    'mask_only_surface',
    'extract_near_surf_dilation',
    'find_below',
    'find_below_all',
    'foreland_mask_surf',
    'topographic_divide',
    'structural_divide',
    'curvature_calc',
    'slope_deffront',
    'deformation_front',
    'deformation_front_dv',
    'deformation_front_mode',
    'volume_calc',
]

def mask_from_surface(surf,array,im_w,im_h):
    '''
    Use a surface to remove exhumation/burial/PIV data above surface histograms.
    '''
    array = np.copy(array)
    # dilation window
    pcs,xes,yes = np.histogram2d(surf.y,surf.x,
                                 bins=(array.shape[0],array.shape[1]),
                                 range=[[0,im_h],[0,im_w]])
    for i in range(pcs.shape[1]):
        if any(pcs[:,i])>0:
            line = np.where(pcs[:,i]>0)
            pcs[:,i][line[0][-1]:]=1
    pcs[pcs>0]=1
    pcs = np.array(abs(pcs-1),dtype='int')
    array[pcs==0] = np.nan
    return array


def mask_from_surface_old(surf,array,im_w,im_h,square_size=8):
    '''
    Use a surface to remove exhumation/burial data above surface histograms.
    '''
    r_im_h = round(im_h/square_size)*square_size
    r_im_w = round(im_w/square_size)*square_size
    num_x = len(np.linspace(0,r_im_w,round(im_w/square_size)))
    num_y = len(np.linspace(0,r_im_h,round(im_h/square_size)))
    # dilation window
    pcs,xes,yes = np.histogram2d(surf.y,surf.x,bins=(num_y,num_x),
                                  range=[[0,im_h],[0,im_w]])
    for i in range(pcs.shape[1]):
        if any(pcs[:,i])>0:
            line = np.where(pcs[:,i]>0)
            pcs[:,i][line[0][-1]:]=1
    pcs[pcs>0]=1
    pcs = np.array(abs(pcs-1),dtype='int')
    array[pcs==0] = np.nan
    return array


def mask_only_surface(surf,array,im_w,im_h):
    '''
    Use a surface to remove exhumation/burial/PIV data above surface histograms.
    returns only binary
    '''
    array = np.copy(array)
    # dilation window
    pcs,_,_ = np.histogram2d(surf.y,surf.x,
                                 bins=(array.shape[0],array.shape[1]),
                                 range=[[0,im_h],[0,im_w]])
    for i in range(pcs.shape[1]):
        if any(pcs[:,i])>0:
            line = np.where(pcs[:,i]>0)
            pcs[:,i][line[0][-1]:]=1
    pcs[pcs>0]=1
    pcs = np.array(abs(pcs-1),dtype='uint8')
    return pcs


def extract_near_surf_dilation(surf,array,im_w,im_h,window=3,square_size=8):
    '''
    Extract the values of an array around a line using image pattern dilation.
    From the extracted line, take mean along x-axis to get scalar value.
    Use on individual surfaces
    '''
    r_im_h = round(im_h/square_size)*square_size
    r_im_w = round(im_w/square_size)*square_size
    num_x = len(np.linspace(0,r_im_w,round(im_w/square_size)))
    num_y = len(np.linspace(0,r_im_h,round(im_h/square_size)))
    # dilation window
    kernel = np.ones((window,window),np.uint8)
    pcs,_,_ = np.histogram2d(surf.y,surf.x,bins=(num_y,num_x),
                                 range=[[0,im_h],[0,im_w]])
    # set any bin with count>1 to 1 to make binary mask
    pcs[pcs>0]=1
    # dilate
    dilation = binary_dilation(pcs,kernel,iterations = 1)
    # find array area near line
    line = array*dilation
    # set all 0s to nans
    line[line==0]=np.nan
    return np.nanmean(line,axis=0)


def find_below(surface,points):
    '''
    Use surface data (in pandas DataFrame format) to determine if points
        (n by 2 array format) are below surface

    Returns boolean index array for masking PIvab column data

    ~1000x faster than surface_masking_points
    '''
    surface.dropna(inplace=True)
    sf = UnivariateSpline(surface.x, surface.y)
    x, y = points[:,0], points[:,1]
    new_y = y - sf(x)
    return new_y<0


def find_below_all(surf_file,df,im_w,im_h,xmin,ymax):
    '''
    Use surface data (in pandas DataFrame format) to determine if points
        (n by 2 array format) are below surface

    Returns boolean index array for masking PIvab column data

    ~1000x faster than surface_masking_points
    '''
    filtd = []
    for fr in df.frame.unique():
        df_sel = df[df.frame==fr]
        surface = pd.read_hdf(surfs,'wedgetop_%05.0f'%(fr))
        surface.dropna(how='any',inplace=True)
        surface.x = surface.x - xmin
        surface.y = surface.y - (im_h-ymax)
        if len(surface) > 0:
            sf = UnivariateSpline(surface.x, surface.y)
            x, y = df_sel.x.values, df_sel.y.values
            new_y = y - sf(x)
            filtd.append(df_sel[new_y<0])
        else:
            filtd.append(df_sel)
    return pd.concat(filtd).reset_index(drop=True)


def foreland_mask_surf(surface,far_edge_count=1000,threshold=10):
    '''
    Identify foreland to negate errors in temperature gradient / heat flow
        calculations

    Parameters
    ----------
    surface : array
        Array containing surface elevation data.
    far_edge_count : int, optional
        Distance from the foreland edge in pixels to make mask.
        The default is 1000.
    threshold : int, optional
        Number of pixels above and below mean to count as foreland.
        The default is 10.

    Returns
    -------
    mask : bool
        False where wedge, True where foreland.
        Can invert if wedge is desired.
    mask : pd.DataFrame
        Location (frame, x) of the surface expression of the deformation front
    '''
    # determine mean height of foreland material on far edge of surface array
    far_edge_mean = surface[:,-far_edge_count:].mean()
    # mask to find area that is within a given elevation of this mean
    mask = (surface > far_edge_mean - threshold) & \
                (surface < far_edge_mean + threshold)
    # set boolean mask in area analyzed to True to remove edge effects
    mask[:,-far_edge_count:] = True

    # extract surface from detected edge going down array with convergence
    edge = filters.sobel(mask.astype(float))
    front = []
    for i in range(edge.shape[0]):
        if len(edge[i,:][edge[i,:] > 0]):
            # location is the last nonzero element of edge detection array
            front_loc = np.nonzero(edge[i,:])[0][-1]
            front.append([i,front_loc])
        else:
            front.append([i,np.nan])
    front = pd.DataFrame(front,columns=('frame','x'))
    return mask, front


def topographic_divide(surffile,num_imgs,
                        end_buffer=300.,topo_cutoff=70.,
                        im_h=0.,xmin=0.,ymax=0.
                        medfilt_out=True,from_larger=False):
    '''
    Track topographic divide in doubly-vergent models.
    Ignore surface near backwall, and ignore model images where the topography
        is insignificant, while median filtering the output [optional]
        to remove localized irreguarities from surface calculation errors.

    Returns DataFrame with [frame,xpos] format.
    '''
    t_d = []
    for i in range(num_imgs):
        # read surfaces
        surf = pd.read_hdf(surffile,'wedgetop_%05.0f'%i).dropna()
        if from_larger:
            surf.x = surf.x - xmin
            surf.y = surf.y - (im_h-ymax)
        # remove [end_buffer] pixels from backstop end
        sel_surf = surf[surf.x > end_buffer]
        # calcuate divide for surfaces with topo. taller than [topo_cutoff]
        if sel_surf.y.max() - sel_surf.y.min() > topo_cutoff:
            t_d.append([i,sel_surf[sel_surf.y == sel_surf.y.max()].x.values[0]])
        else:
            t_d.append([i,np.nan])
    if medfilt_out:
        t_d = np.array(t_d)
        td_mf = medfilt(t_d[:,1])
        t_d = pd.DataFrame(np.vstack((t_d[:,0],td_mf)).T,
                          columns=['frame','x_td'])
        return t_d
    else:
        return pd.DataFrame(np.array(t_d),columns=['frame','x_td'])


def structural_divide(shear, elevation_px=4, sigma=25):
    '''
    Determine the location of the structural divide in a doubly vergent wedge
        by finding zero-crossings in the 0-axis mean in an
        blurred shear strain array


    Parameters
    ----------
    cexy : float array
        Cumulative shear strain array.
    elevation_px : int, optional
        elevation above base in pixels to begin analysis. The default is 4.
    sigma : int, optional
        Sigma value to blur array. The default is 25.

    Returns
    -------
    DataFrame
        DataFrame of file numbers and location of structural divide in pixels

    '''
    sd = []
    for i in range(shear.shape[0]):
        # copy array and crop off bottom noisy pixels
        array = np.copy(shear[i, :, :])
        array = array[elevation_px:, :]
        # use sign function to determine where shear strains are
        #   positive (retro) and negative (pro)
        #   and heavily filter array (play with sigma)
        sign_filt = filters.gaussian_filter(np.sign(array), sigma=sigma)
        sign_filt[array == 0] = np.nan
        # take mean to make function to find zero crossing(s)
        mean_sf = np.nansum(sign_filt, axis=0)
        x = np.linspace(0, len(mean_sf), len(mean_sf))
        z = find_roots(x, mean_sf)
        if len(z) > 0:
            sd.append([i, z[-1]*sqr_sz])
        else:
            sd.append([i, np.nan])
    return pd.DataFrame(sd, columns=('frame', 'x_sd'))

def curvature_calc(line):
    '''
    Line vector curvature calcuations from (x,y) data in pandas format, from:
        http://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
    '''
    # x,y change, first derivative
    dx_dt = np.gradient(line.x)
    dy_dt = np.gradient(line.y)
    velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])

    # arc change
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

    # second derivative
    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    #tangent vector
    tangent = np.array([1/ds_dt] * 2).transpose() * velocity
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]
    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)

    normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    t_component = np.array([d2s_dt2] * 2).transpose()
    n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()

    acceleration = t_component * tangent + n_component * normal

    return curvature, acceleration


def slope_deffront(surfs,start_frame,num_frames,window=81,peak_thres=0.4,
                      scale=70.,plotting=False,sgol=False):
    '''
    FOR SINGLY-VERGENT WEDGES ONLY

    Examine calcuated surfaces, and determine the surface expression of their
        deformation fronts over the length of an experiment.

    Perform calcuation on Gaussian-filtered surface, as raw surface is noisy
        enough to interfere with properly identifying the deformation front,
        as the last "turn" in the surface, topography decreasing to the left.

    Calcuated value may NOT be actual deformation front, as deformation may
        jump forelandward for a short time before surface expression.

    THEN, calcuate the slope of the surface BEFORE the deformation front,


    TODO:
        - modify to calcuate last peak of curvature, to detect the retrowedge
            boundary in doubly-vergent analog models, and calcuate both
            pro- and retro-wedge slopes
    '''
    slope = []
    for i in np.arange(start_frame,num_frames):
        # read each surface
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        surf = surf.y.dropna().reindex(surf.x, method='nearest').reset_index()
        if sgol:
            smoothed_y = savgol_filter(surf.y, window, 3)
        else:
            # smooth surface with 1-D Gaussian filter of size [window]
            smoothed_y = gaussian_filter1d(surf.y, window)
        # recompile into DataFrame
        smoothed_surf = pd.DataFrame(np.vstack((surf.x,smoothed_y)).T,
                                     columns=['x','y'])
        # calcuate curvature, ignoring 'acceleration' of surface
        cc,_ = curvature_calc(smoothed_surf)
        # calcuate peaks in curvature
        indexes = peakutils.indexes(cc,thres=peak_thres)
        # calcuate slope of topography BEFORE the deformation front [last peak]
        max_surf = surf[surf.y == surf.y.max()].index.values[0]
        if len(surf[(surf.x <= surf.x[indexes[-1]]) & (surf.x >= surf.x[max_surf])]) >= 3:
            wedge_topo = surf[(surf.x <= surf.x[indexes[-1]]) & (surf.x >= surf.x[max_surf])]
        else:
            wedge_topo = surf[(surf.x <= surf.x[indexes[-1]])]
        m,b = line_fit(wedge_topo.x,wedge_topo.y)
        slope.append(m)
    return slope

def deformation_front(surfs,num_frames,skip_x=100,
                        im_h=0.,xmin=0.,ymax=0.,
                        sigma=5,
                        from_larger=False
                        ):
    '''
    FOR SINGLY-VERGENT WEDGES ONLY

    Examine calcuated surfaces, and determine the surface expression of their
        deformation fronts over the length of an experiment.

    Perform calcuation on filtered surface, with the deformation front defined
        as the first area that exceeds elevation outside of a specifed
        range of [sand_elev +/- surfvar].

    Calcuated value is only the physical expression of the deformation front,
        as actual deformation may jump forelandward for a short time
        before surface expression.
    '''
    # determine maxlimum width of profile
    len_x = []
    for i in range(num_frames):
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        if from_larger:
            surf.x = surf.x - xmin
            surf.y = surf.y - (im_h-ymax)
        len_x.append(surf.x.max())
    width = int(max(len_x))

    # place surface in preallocated array, smooth array
    surfarr = np.zeros((num_frames,width))
    for i in range(num_frames):
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        surf = surf.y.dropna().reindex(surf.x, method='nearest').reset_index()
        if from_larger:
            surf.x = surf.x - x_min
            surf.y = surf.y - (im_h-ymax)
        surfarr[i,np.array(surf.x.values,dtype='int')-1] = surf.y.values
    # complete sobel filter on topography and find first and last peaks
    #   these ideally are the retrowedge and prowedge deformation fronts
    ridges = filters.gaussian_filter(filters.sobel(surfarr[:,skip_x:]),sigma=sigma)
    # identify deformation front from [foreland], shift into image coordinates
    front_df= []
    for i in range(ridges.shape[0]):
        indexes = peakutils.indexes(np.abs(ridges[i,:]),min_dist=100)
        front_df.append([i, indexes[-1]])
    front_df = pd.DataFrame(front_df,columns=('frame','x_df'))
    return front_df


def deformation_front_dv(surfs,num_frames,
                            im_h=0.,xmin=0.,ymax=0.,
                            sigma=5,
                            from_larger=False):
    '''
    FOR DOUBLY-VERGENT WEDGES ONLY

    Examine calcuated surfaces, and determine the surface expression of their
        deformation fronts over the length of an experiment.

    Perform calcuation on filtered surface, with the deformation front defined
        as the first area that exceeds elevation outside of a specifed
        range of [sand_elev +/- surfvar].

    Calcuated value is only the physical expression of the deformation front,
        as actual deformation may jump forelandward for a short time
        before surface expression.
    '''
    # determine maximum width of profile
    len_x = []
    for i in range(num_frames):
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        if from_larger:
            surf.x = surf.x - xmin
            surf.y = surf.y - (im_h-ymax)
        len_x.append(surf.x.max())
    width = int(max(len_x))

    # place surface in preallocated array, smooth array
    surfarr = np.zeros((num_frames,width))
    for i in range(num_frames):
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        surf = surf.y.dropna().reindex(surf.x, method='nearest').reset_index()
        if from_larger:
            surf.x = surf.x - xmin
            surf.y = surf.y - (im_h-ymax)
        surfarr[i,np.array(surf.x.values,dtype='int')-1] = surf.y.values
    # complete sobel filter on topography and find first and last peaks
    #   these ideally are the retrowedge and prowedge deformation fronts
    ridges = filters.gaussian_filter(filters.sobel(surfarr),sigma=sigma)
    # identify deformation front from [foreland], shift into image coordinates
    front_r_df= []
    for i in range(ridges.shape[0]):
        indexes = peakutils.indexes(np.abs(ridges[i,:]),min_dist=100)
        front_r_df.append([i, indexes[0], indexes[-1]])
    front_r_df = pd.DataFrame(front_r_df,columns=('frame','x_rf','x_df'))
    return front_r_df


def deformation_front_mode(surfs,num_frames,far_edge_count=1000,threshold=10,
                            first_frames=1000,horiz_width=2000,
                            im_h=0.,xmin=0.,ymax=0.,
                            from_larger=True, grow_horiz=True,
                            vergence='s'):
    '''
    Examine calcuated surfaces, and determine the surface expression of their
        deformation fronts over the length of an experiment.

    Perform calcuation on surface with the retrowedge as the first area that
        exceeds the average flat topography elevation, and the deformation
        front defined as the last area that exceeds elevation outside of
        a specifed range of [sand_elev +/- surfvar].

    Calcuated value is only the physical expression of the deformation front,
        as actual deformation may jump forelandward for a short time
        before surface expression.
    '''
    # determine maxlimum width of profile
    len_x = []
    for i in range(num_frames):
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        if from_larger:
            surf.x = surf.x - xmin
            surf.y = surf.y - (im_h-ymax)
        len_x.append(surf.x.max())
    width = int(max(len_x))

    # place surface in preallocated array
    surface = np.zeros((num_frames,width))
    for i in range(num_frames):
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        surf = surf.y.dropna().reindex(surf.x, method='nearest').reset_index()
        if from_larger:
            surf.x = surf.x - xmin
            surf.y = surf.y - (im_h-ymax)
        surface[i,np.array(surf.x.values,dtype='int')-1] = surf.y.values

    # determine mean height of foreland material on far edge of surface array
    far_edge_mean = surface[:first_frames,-far_edge_count:].mean()
    # mask to find area that is within a given elevation of this mean
    mask = (surface > far_edge_mean - threshold) & \
                (surface < far_edge_mean + threshold)
    # set boolean mask on edges to remove edge effects
    if grow_horiz:
        mask = np.hstack((mask,np.ones((mask.shape[0],horiz_width)))).astype(bool)
    mask[:,-far_edge_count:] = True

    # different modes for either style of model
    if vergence == 's':
        edge = filters.sobel(mask.astype(float))
        front = []
        for i in range(edge.shape[0]):
            if len(edge[i,:][edge[i,:] > 0]):
                front_loc = np.nonzero(edge[i,:])[0][-1]
                front.append([i,front_loc])
            else:
                front.append([i,np.nan])
        front = pd.DataFrame(front,columns=('frame','x_df'))
    if vergence == 'd':
        edge = filters.sobel(mask.astype(float))
        front = []
        for i in range(edge.shape[0]):
            if len(edge[i,:][edge[i,:] > 0]):
                retro_loc = np.nonzero(edge[i,:])[0][0]
                front_loc = np.nonzero(edge[i,:])[0][-1]
                topo_div = round(np.where(surface[i,:] == surface[i,:].max())[0].mean())
                front.append([i,retro_loc,front_loc,topo_div])
            else:
                front.append([i,np.nan,np.nan])
        front = pd.DataFrame(front,columns=('frame','x_rf','x_df','x_td'))
    elif vergence != 's' and vergence != 'd':
        raise ValueError('Choose either singly (s) or doubly (d) vergent mode')
    return front


def volume_calc(surfaces,num_frames,scale=61.,
                    im_h=0.,xmin=0.,ymax=0.,
                    from_larger=True):
    '''
    Calcuate wedge area using interpolated spline fit and built-inintegration
        on surface data

    Use to determine erosional volume by analyzing areas over convergence

    Parameters
    ----------
    surfaces : hdf5 store that contains all of the surfaces for a mode.
    num_frames : number of frames in store to use

    Returns
    -------
    integrated_area: area [cm^2] of each wedge, calcuated by definite integral
    integrated_area_diff : difference in area [cm^2] over convergence
    '''
    integrated_area = []
    for i in range(num_frames):
        # load surface and shift dimensions if from a bigger image
        surf = pd.read_hdf(surfaces,'wedgetop_%05.0f'%i)
        if from_larger:
            surf.x = surf.x - xmin
            surf.y = surf.y - (im_h-ymax)
        # spline fit to scaled surface
        f = InterpolatedUnivariateSpline(surf.x/scale, surf.y/scale)
        # definite integral of surface between origin and end of model image
        integrated_area.append(f.integral(0,surf.x.max()/scale))
    areas = np.array(integrated_area)
    return areas, np.diff(areas)
