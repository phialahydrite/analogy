"""
analogy.histograms
===================
Gridded (image-shaped) histogram summaries built from particle-tracking
DataFrames: synthetic depth histograms, cumulative displacement fields,
and exhumation/burial fields.
"""
import numpy as np

__all__ = [
    'depth_synthetic_hist',
    'displace_hist',
    'exhum_burial_hist',
    'depth_burial_hist',
]


def depth_synthetic_hist(tp_df,im_w,im_h,num_frames,square_size=8):
    '''
    Calcuate a histogram of depth for large collections of synthetic particles
    '''
    # Set divide-by-zero warning to off to ignore warning.
    np.seterr(divide='ignore', invalid='ignore')
    # set the range of histograms, number of frames by image (y,x) dimensions
    extent = [[0,num_frames],[0,im_h],[0,im_w]]
    # calcuate the correct number of bins in each spatial dimension to match
    #   ideal square size
    r_im_h = round(im_h/square_size)*square_size
    r_im_w = round(im_w/square_size)*square_size
    num_x = len(np.linspace(0,r_im_w,round(im_w/square_size)))
    num_y = len(np.linspace(0,r_im_h,round(im_h/square_size)))
    # calcuate burial, exhumation, and minimum depth values
    #   additionally calcuate counts within each bin to get mean
    #   values of each quantity
    particle_pos = tp_df[['frame','y','x']].values
    depth, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.depth,
                                          density=False, range=extent)
    starting_depth, _ = np.histogramdd(particle_pos, bins=(num_frames,
                                                           num_y,num_x),
                                          weights=tp_df.starting_depth,
                                          density=False, range=extent)
    max_depth, _ = np.histogramdd(particle_pos, bins=(num_frames,
                                                      num_y,num_x),
                                          weights=tp_df.max_depth,
                                          density=False, range=extent)
    cum_max_depth, _ = np.histogramdd(particle_pos, bins=(num_frames,
                                                          num_y,num_x),
                                          weights=tp_df.cum_max_depth,
                                          density=False, range=extent)
    geot_e, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.e_temp,
                                          density=False, range=extent)
    geot_b, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.b_temp,
                                          density=False, range=extent)
    part_counts, bin_edges, = np.histogramdd(particle_pos,
                                             bins=(num_frames,num_y,num_x),
                                             range=extent)
    # if nan, mask array for cleanliness of data
    depth = np.ma.masked_invalid(depth / part_counts)
    starting_depth = np.ma.masked_invalid(starting_depth / part_counts)
    max_depth = np.ma.masked_invalid(max_depth / part_counts)
    cum_max_depth = np.ma.masked_invalid(cum_max_depth / part_counts)
    geot_e = np.ma.masked_invalid(geot_e / part_counts)
    geot_b = np.ma.masked_invalid(geot_b / part_counts)
    return depth,starting_depth,max_depth,cum_max_depth,geot_e,geot_b


def displace_hist(tp_df,im_w,im_h,num_frames,square_size=8,mask=False):
    '''
    Calcuate a histogram of current and cumulative displacements
    for large collections of synthetic particles

    To plot:
    (bin_edges[2] is the y-dimension, bin_edges[1] is the x-dimension.)
    plt.pcolormesh(bin_edges[2], bin_edges[1], values[frame,:,:],
                    edgecolors='none')

    '''
    # Set divide-by-zero warning to off to ignore warning.
    np.seterr(divide='ignore', invalid='ignore')
    # set the range of histograms, number of frames by image (y,x) dimensions
    extent = [[0,num_frames],[0,im_h],[0,im_w]]
    # calcuate the correct number of bins in each spatial dimension to match
    #   ideal square size
    r_im_h = round(im_h/square_size)*square_size
    r_im_w = round(im_w/square_size)*square_size
    num_x = len(np.linspace(0,r_im_w,round(im_w/square_size)))
    num_y = len(np.linspace(0,r_im_h,round(im_h/square_size)))
    # calcuate burial, exhumation, and minimum depth values
    #   additionally calcuate counts within each bin to get mean
    #   values of each quantity h
    particle_pos = tp_df[['frame','y','x']].values
    exy, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.exy,
                                          density=False, range=extent)
    vort, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.vort,
                                          density=False, range=extent)
    cumexy, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.cumexy,
                                          density=False, range=extent)
    cumvort, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.cumvort,
                                          density=False, range=extent)
    part_counts, bin_edges, = np.histogramdd(particle_pos,
                                             bins=(num_frames,num_y,num_x),
                                             range=extent)
    # if nan, mask array for cleanliness of data
    if mask:
        exy = np.ma.masked_invalid(exy / part_counts)
        vort = np.ma.masked_invalid(vort / part_counts)
        cumexy = np.ma.masked_invalid(cumexy / part_counts)
        cumvort = np.ma.masked_invalid(cumvort / part_counts)
    return exy, vort, cumexy, cumvort, part_counts, bin_edges


def exhum_burial_hist(tp_df,im_w,im_h,num_frames,calc_conv=False,square_size=8):
    '''
    Calcuate a histogram of depth, burial and exhumation over the history
        of an experiment.
    Resuts are 3D arrays with a [framenum,y,x] format.
    Y and X are swapped to fit original image and python array coordinates.

    MUCH, MUCH  faster than any of the depth_history functions, assuming that
        [cumuative_depth_stats] is used to calcuate max depths, burial and
        exhumation amounts prior.

    To plot:
    (bin_edges[2] is the y-dimension, bin_edges[1] is the x-dimension.)
    plt.pcolormesh(bin_edges[2], bin_edges[1], values[frame,:,:],
                   edgecolors='none')
    '''
    # Set divide-by-zero warning to off to ignore warning.
    np.seterr(divide='ignore', invalid='ignore')
    # set the range of histograms, number of frames by image (y,x) dimensions
    extent = [[0,num_frames],[0,im_h],[0,im_w]]
    # calcuate the correct number of bins in each spatial dimension to match
    #   ideal square size
    r_im_h = round(im_h/square_size)*square_size
    r_im_w = round(im_w/square_size)*square_size
    num_x = len(np.linspace(0,r_im_w,round(im_w/square_size)))
    num_y = len(np.linspace(0,r_im_h,round(im_h/square_size)))
    # calcuate burial, exhumation, and minimum depth values
    #   additionally calcuate counts within each bin to get mean
    #   values of each quantity h
    particle_pos = tp_df[['frame','y','x']].values
    burial, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.burial,
                                          density=False, range=extent)
    exhumation, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.exhumation,
                                          density=False, range=extent)
    part_counts, bin_edges, = np.histogramdd(particle_pos,
                                             bins=(num_frames,num_y,num_x),
                                             range=extent)
    # if nan, mask array for cleanliness of data
    burial = np.ma.masked_invalid(burial / part_counts)
    exhumation = np.ma.masked_invalid(exhumation / part_counts)
    return burial, exhumation, part_counts, bin_edges


def depth_burial_hist(tp_df,im_w,im_h,num_frames,calc_conv=False,square_size=8):
    '''
    Calcuate a histogram of depth, burial and exhumation over the history
        of an experiment.
    Resuts are 3D arrays with a [framenum,y,x] format.
    Y and X are swapped to fit original image and python array coordinates.

    MUCH, MUCH  faster than any of the depth_history functions, assuming that
        [cumuative_depth_stats] is used to calcuate max depths, burial and
        exhumation amounts prior.

    To plot:
    (bin_edges[2] is the y-dimension, bin_edges[1] is the x-dimension.)
    plt.pcolormesh(bin_edges[2], bin_edges[1], values[frame,:,:],
                   edgecolors='none')
    '''
    # Set divide-by-zero warning to off to ignore warning.
    np.seterr(divide='ignore', invalid='ignore')
    # set the range of histograms, number of frames by image (y,x) dimensions
    extent = [[0,num_frames],[0,im_h],[0,im_w]]
    # calcuate the correct number of bins in each spatial dimension to match
    #   ideal square size
    r_im_h = round(im_h/square_size)*square_size
    r_im_w = round(im_w/square_size)*square_size
    num_x = len(np.linspace(0,r_im_w,round(im_w/square_size)))
    num_y = len(np.linspace(0,r_im_h,round(im_h/square_size)))
    # calcuate burial, exhumation, and minimum depth values
    #   additionally calcuate counts within each bin to get mean
    #   values of each quantity h
    particle_pos = tp_df[['frame','y','x']].values
    burial, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.burial,
                                          density=False, range=extent)
    exhumation, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.exhumation,
                                          density=False, range=extent)
    cummindepth, _ = np.histogramdd(particle_pos,bins=(num_frames,num_y,num_x),
                                          weights=tp_df.cumuative_min_depth,
                                          density=False, range=extent)
    cumdD, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x),
                                          weights=tp_df.dD_cum,
                                          density=False, range=extent)
    part_counts, bin_edges, = np.histogramdd(particle_pos,
                                             bins=(num_frames,num_y,num_x),
                                             range=extent)
    # if nan, mask array for cleanliness of data
    burial = np.ma.masked_invalid(burial / part_counts)
    exhumation = np.ma.masked_invalid(exhumation / part_counts)
    cumdD = np.ma.masked_invalid(cumdD / part_counts)
    cummindepth = np.ma.masked_invalid(cummindepth / part_counts)
    if calc_conv:
        conv_bur, _ =  np.histogramdd(particle_pos,bins=(num_frames,num_y,num_x),
                                              weights=tp_df.conv_during_burial,
                                              density=False, range=extent)
        conv_ex, _ =  np.histogramdd(particle_pos,bins=(num_frames,num_y,num_x),
                                         weights=tp_df.conv_during_exhumation,
                                         density=False, range=extent)
        conv_bur = np.ma.masked_invalid(conv_bur / part_counts)
        conv_ex = np.ma.masked_invalid(conv_ex / part_counts)
        return burial, exhumation, cummindepth, cumdD, conv_bur,conv_ex, \
                part_counts, bin_edges
    else:
        return burial, exhumation, cummindepth, cumdD, part_counts, bin_edges
