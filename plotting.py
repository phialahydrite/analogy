"""
analogy.plotting
=================
Visualization helpers: plotting linked particle trajectories over a
background image, and checking PIV sub-pixel bias for a set of particle
diameters.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

__all__ = [
    'tp_plot_traj',
    'test_subpix_bias',
]


def tp_plot_traj(trajs,sample_int=1,
                    scale=61.,im_w=0.,im_h=0.,
                    prefix='',
                    scaled=False,save=False,
                    cmap=plt.cm.viridis):
    '''
    Convenience function to plot all trajs in a given dataframe
        with scaled axes.
    Optional settings for plotting every [sample_int] particle location,
        useful for extremely large datasets.
    '''
    f, ax = plt.subplots(figsize=(10,10))
    particle_it = np.sort(trajs['particle'].unique())
    trajs = trajs[trajs.particle.isin(particle_it)]
    # initialize plot
    #   place ticks outside of plot to avoid covering image
    #   remove right and upper axes to simplify plot
    #   only plot ticks on the left and bottom of plot
    ax.tick_params(axis='y', direction='out')
    ax.tick_params(axis='x', direction='out')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot particle trajs leading up to current fram
    if scaled:
        x = trajs['x']/scale
        y = trajs['y']/scale
    else:
        x = trajs['x']
        y = trajs['y']
    ax.scatter(x[::sample_int],y[::sample_int],
               c=trajs['frame'][::sample_int],marker='o',
               s=2, alpha=0.25, cmap=cmap,lw=0,
               vmin=trajs.frame.min(),vmax=trajs.frame.max())
    plt.axis('scaled')
    if scaled:
        ax.set_xlim([0,im_w/scale])
        ax.set_ylim([0,im_h/scale])
        ax.set_xlabel('Width [cm]')
        ax.set_ylabel('Height [cm]')
    else:
        ax.set_xlim([0,im_w])
        ax.set_ylim([0,im_h])
        ax.set_xlabel('Width [px]')
        ax.set_ylabel('Height [px]')
    if save:
        plt.savefig('%s_f%05ito%05i.png'%(prefix,trajs.frame.min(),
                                          trajs.frame.max()),dpi=200,
                                          bbox_inches='tight')
        plt.close('all')


def test_subpix_bias(diameters,image):
    '''
    Test uniformity of the subpixel bias for particles found within an
        image. A lower result represents even sampling from .0 to .9,
        with larger numbers signifying that there are peak(s) in
        subpixel values, and thus definite pixel bias in the analysis
    Parameters
    ----------
    diameters: list
        list of odd integers representing identified particle diameters
    image: array
        grayscale image to identify particles within
    Returns
    -------
    diameter : int
        input diameter for locating particle
    x_bias :  float
        uniformity of subpixel measurements from .0 to .9, x-direction,
        lower = more uniform
    y_bias :  float
        uniformity of subpixel measurements from .0 to .9, y-direction,
        lower = more uniform
    mean_signal :  float
        mean signal strength of all particles measured at [diameter]
    mean_epsilon :  float
        mean uncertainty of particle centroid position, 0 = correctly identified
    mean_eccentricity :  float
        mean eccentricity of particle centroid, 0 = circuar
    num_particles :  int
        number of particles identified at each [diameter]
    '''
    px_stats = []
    for diam in diameters:
        f = tp.locate(image,diam,minmass=100,invert=True)
        x_sub = f[['x','y']].applymap(lambda x: x % 1).x
        y_sub = f[['x','y']].applymap(lambda x: x % 1).y
        xst = stats.kstest(x_sub, stats.uniform(loc=0.0, scale=1.0).cdf)
        yst = stats.kstest(y_sub, stats.uniform(loc=0.0, scale=1.0).cdf)
        px_stats.append([diam,xst[0],yst[0],f.signal.mean(),abs(f.ep.mean()),
                         f.ecc.mean(),f.mass.mean(),len(f)])
    return pd.DataFrame(px_stats,
                        columns=['diameter','x_bias','y_bias','mean_signal',
                                 'mean_epsilon','mean_eccentricity','mean_mass',
                                 'num_particles'])
