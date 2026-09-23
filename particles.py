"""
analogy.particles
=================
Single/multi-particle-tracking (SPTV) analysis: depth-below-surface
calculation for tracked particles, burial/exhumation statistics through
time, dense particle-displacement field extraction (with optional
temperature and deformation-front pass-through), and extraction of
particles along a chosen line/section.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from scipy.interpolate import UnivariateSpline
import peakutils
from collections.abc import Iterable

from .utils import fill, gridize, degridize, do_kdtree, angle, strictly_decreasing
from .io import PIV_framenumbers
from .surfaces import find_below

__all__ = [
    'particle_displacer',
    'particle_displacer_temperatures',
    'particle_displacer_temperatures_dfmfront',
    'depth_calc',
    'find_nearest_spatemp',
    'line_particles',
    'line_particles_subsel',
    'line_particles_subsel_all',
    'sort_tp_df',
    'sorted_line_particles',
    'first_appearance_length',
    'depth_diff',
    'depth_rate',
    'cumu_depth',
    'cumu_exy',
    'starting_depth',
    'starting_max_depth_redux',
    'depth_stats_compilation',
    'cumulative_depth_stats',
    'shift_x_def',
    'shift_x_def_simple',
    'rdp_burial_calc_forlambda',
    'conv_between',
]


def particle_displacer(files,particles,framenumbers,surface_hdf5,
                              begin_file,end_file,
                              step = 2, radius=24,
                              verbose = False,
                              replace_nan = False,
                              fill_array = False,
                              mask_piv = False):
    '''
    Using Matlab PILvab output with x,y,u,v data [files], displace synthetic
        markers [particles], while simutaneously filtering PIV data that
        falls outside of the model wedge, using calcuated surfaces
        [surface_hdf5], between a range of files [begin_file:end_file]
    '''
    #initial variables and lists
    start_frames = particles.frame.unique()
    df = [] #DataFrame
    rolling_particle_nos = 0 # inital count of particles to add to
    total_time = float()
    for n in start_frames:
        # unique particles for each frame in constructed grid of particles
        parts = particles[particles.frame == n]
        # first compared B-image for dataframe
        B_frame_start = framenumbers[framenumbers.B_frame >= n].index.min()
        continuing = []
        for i, f in enumerate(files[B_frame_start:end_file]):
            start_time=time.time()
            # read frame numbers from PIV output
            fn, A_img_no, B_img_no = PIV_framenumbers(f)

            # read PIV data, ignoring extra columns if they exist
            data = pd.read_csv(f,skiprows=3,usecols = [0,1,2,3],\
                               names=['x','y','u','v'])

            # fill nan with zeros, do not use if using fill_array option
            if replace_nan:
                data.loc[np.isnan(data.u)].u = 0.
                data.loc[np.isnan(data.v)].v = 0.

            # calcuate shear strain values
            u = gridize(data.x,data.y,data.u)
            v = gridize(data.x,data.y,data.v)

            # fill array nans with nearest valid neighbors
            if fill_array:
                u = fill(u)
                v = fill(v)

            # smooth velocity arrays, calcuate shear strain
            u = nd.median_filter(u, 4)/(pixelmm_scale*2)
            v = nd.median_filter(v, 4)/(pixelmm_scale*2)
            dUdY = np.gradient(u.T)[0]
            dVdX = np.gradient(v.T)[1]
            ep_xy = 0.5 * (dUdY + dVdX)
            data['exy'] = degridize(ep_xy)[:,2]

            # calcuate vorticity (Meynart, Fung, Eringen, Timoshekno, etc...)
            VO = dVdX - dUdY
            data['vort'] = degridize(VO)[:,2]

            # flip PIvab format upside-down and shift on x-axis
            data.v = -data.v
            data.y = data.y.max()-data.y
            data.x = data.x - x_min
            xy = np.array([data.x,data.y]).T

            # this is not needed if already masked in PIvab
            if mask_piv:
                # read surface
                surface = pd.read_hdf(surface_hdf5,'wedgetop_%05.0f'%B_img_no)
                surface.x = surface.x-xmin
                surface.y = surface.y-(images.frame_shape[0] - ymax)

                # return boolean index of points beneath surface
                subsurface_piv = find_below(surface,xy)

                # remove background velocities
                data = data.loc[subsurface_piv]
            # background displacement coordinate values for kd-tree
            xy_target= np.vstack((data.x,data.y)).T

            # INITIALIZE particle displacement arrays
            if i == 0:
                # initial particle location grid for kd-tree
                xy_orig = np.array(parts[['x','y']])

                # perform kd-tree, only using nearest PIV grid point
                dist,ind = do_kdtree(xy_target,xy_orig,1)

                # find associated displacements, strains, and vorticity
                #   within radius
                u,v,exy,vort = np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig))
                close_ind =[]
                for k in range(len(xy_orig)):
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts)>=1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig)):
                    if close_ind[k] != None: #ignore empty
                        u[k] = data.u.iat[close_ind[k]]
                        v[k] = data.v.iat[close_ind[k]]
                        exy[k] = data.exy.iat[close_ind[k]]
                        vort[k] = data.vort.iat[close_ind[k]]
                nxs = parts.x.values + u
                nys = parts.y.values + v
                nexy = exy
                nvort = vort

                # initial and first movement particle locations
                start_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                start_frame = np.ones(len(start_pnums))*A_img_no
                start_exy = np.zeros(len(start_pnums))
                start_vort = np.zeros(len(start_pnums))
                start = pd.DataFrame(np.vstack((start_frame,start_pnums,\
                                          parts[['x','y']].T,start_exy,
                                          start_vort)).T,\
                                          columns = ['frame','particle','x',\
                                                     'y','exy','vort'])
                first_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                first_frame = np.ones(len(first_pnums))*B_img_no
                first = pd.DataFrame(np.vstack((first_frame,first_pnums,
                                        nxs,nys,nexy,nvort)).T,\
                                        columns = ['frame','particle','x','y',
                                                   'exy','vort'])

            # CALCuATE incremental displacement arrays for each particle
            else:
                # updated particle location grid for kd-tree
                xy_orig_update = np.array([nxs,nys]).T

                # updated kd-tree for new particle locations, using one point
                dist,ind = do_kdtree(xy_target,xy_orig_update,1)

                # calcuate displacements, strains, and vorticity within radius
                u,v,exy,vort = np.zeros(len(xy_orig_update)),\
                                            np.zeros(len(xy_orig_update)),\
                                            np.zeros(len(xy_orig_update)),\
                                            np.zeros(len(xy_orig_update))
                close_ind =[]
                for k in range(len(xy_orig_update)):
                    #determine location of nearest grid points
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts)>=1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig_update)):
                    if close_ind[k] != None: #ignore empty
                        u[k] = data.u.iat[close_ind[k]]
                        v[k] = data.v.iat[close_ind[k]]
                        exy[k] = data.exy.iat[close_ind[k]]
                        vort[k] = data.vort.iat[close_ind[k]]
                nxs = nxs + u
                nys = nys + v
                nexy = exy
                nvort = vort

                # compile further particle and frame info into DataFrame
                continuing_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                continuing_frame = np.ones(len(continuing_pnums))*B_img_no
                continuing_df = pd.DataFrame(np.vstack((continuing_frame,\
                                    continuing_pnums,nxs,nys,nexy,nvort)).T,\
                                    columns = ['frame','particle','x','y',
                                               'exy','vort'])
                continuing.append(continuing_df)
            if verbose:
                frame_elapsed = time.time()-start_time
                total_time += frame_elapsed
                num_displaced = len([i for i in close_ind if i is not None])
                print('Calcuated displacements for %g '%num_displaced + \
                      'particles for File %05.0f, '%(fn) + \
                      'starting at Frame %05i, '%(n) + \
                      'Frames %05.0f and %05.0f '%(A_img_no,B_img_no) + \
                      'in %06.3f seconds, '%(frame_elapsed) + \
                      '%010.3f total seconds elapsed.'%(total_time))
        # compile data (may be large, >20GB)
        df.append(pd.concat((start,first,pd.concat(continuing))).dropna(how='any'))
        # count of particles as they are iteratively added
        rolling_particle_nos += len(parts.particle.unique())
    return pd.concat(df).reset_index(drop=True)


def particle_displacer_temperatures(piv_files,ero_temps,particles,
                                      framenumbers,
                                      surface_hdf5,
                                      begin_file,end_file,
                                      step = 2, radius=24,
                                      verbose = False,
                                      replace_nan = False,
                                      fill_array = False,
                                      mask_piv = False):
    '''
    Using Matlab PIvab output with x,y,u,v data [files], displace synthetic
        markers [particles], while simutaneously filtering PIV data that
        falls outside of the model wedge, using calcuated surfaces
        [surface_hdf5], between a range of files [begin_file:end_file]
    '''
    #initial variables and lists
    start_frames = particles.frame.unique()
    df = [] #DataFrame
    rolling_particle_nos = 0 # inital count of particles to add to
    total_time = float()
    for n in start_frames:
        # unique particles for each frame in constructed grid of particles
        parts = particles[particles.frame == n]
        # first compared B-image for dataframe
        B_frame_start = framenumbers[framenumbers.B_frame >= n].index.min()
        continuing = []
        for i, f in enumerate(piv_files[B_frame_start:end_file]):
            start_time=time.time()
            # read frame numbers from PIV output
            fn, A_img_no, B_img_no = PIV_framenumbers(f)

            # read PIV data, ignoring extra columns if they exist
            data = pd.read_csv(f,skiprows=3,usecols = [0,1,2,3],\
                               names=['x','y','u','v'])

            # fill nan with zeros, do not use if using fill_array option
            if replace_nan:
                data.loc[np.isnan(data.u)].u = 0.
                data.loc[np.isnan(data.v)].v = 0.

            # calcuate shear strain values
            U = gridize(data.x,data.y,data.u)
            V = gridize(data.x,data.y,data.v)

            # fill array nans with nearest valid neighbors
            if fill_array:
                U = fill(U)
                V = fill(V)

            # smooth velocity arrays, calcuate shear strain
            U = nd.median_filter(U, 4)/(pixelmm_scale*2)
            V = nd.median_filter(V, 4)/(pixelmm_scale*2)
            dUdY = np.gradient(U.T)[0]
            dVdX = np.gradient(V.T)[1]
            ep_xy = 0.5 * (dUdY + dVdX)
            data['exy'] = degridize(ep_xy)[:,2]

            # calcuate vorticity (Meynart, Fung, Eringen, Timoshekno, etc...)
            VO = dVdX - dUdY
            data['vort'] = degridize(VO)[:,2]

            # flip PIvab format upside-down and shift on x-axis
            data.v = -data.v
            data.y = data.y.max() - data.y
            data.x = data.x - data.x.min()
            xy = np.array([data.x,data.y]).T

            # read  and merge temperature data, shifting coordinates
            temp_e = pd.read_csv(ero_temps[i], delimiter=' ',
                               names=['x','y','temp_c'])

            # place temps in data by merging to align with existing values
            data = data.merge(temp_e,left_on=['x','y'],right_on=['x','y'],
                       how='left')
            # this is not needed if already masked in PIVlab
            if mask_piv:
                # read surface
                surface = pd.read_hdf(surface_hdf5,'wedgetop_%05.0f'%B_img_no)
                surface.x = surface.x-xmin
                surface.y = surface.y-(images.frame_shape[0] - ymax)

                # return boolean index of points beneath surface
                subsurface_piv = find_below(surface,xy)

                # remove background velocities
                data = data.loc[subsurface_piv]
            # background displacement coordinate values for kd-tree
            xy_target= np.vstack((data.x,data.y)).T

            # INITIALIZE particle displacement arrays
            if i == 0:
                # initial particle location grid for kd-tree
                xy_orig = np.array(parts[['x','y']])

                # perform kd-tree, only using nearest PIV grid point
                dist,ind = do_kdtree(xy_target,xy_orig,1)

                # find associated displacements, strains, and vorticity
                #   within radius
                u,v,exy,vort,e_temp = np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig))
                close_ind =[]
                for k in range(len(xy_orig)):
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts)>=1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig)):
                    if close_ind[k] != None: #ignore empty
                        u[k] = data.u.iat[close_ind[k]]
                        v[k] = data.v.iat[close_ind[k]]
                        exy[k] = data.exy.iat[close_ind[k]]
                        vort[k] = data.vort.iat[close_ind[k]]
                        e_temp[k] = data.temp_c.iat[close_ind[k]]
                nxs = parts.x.values + u
                nys = parts.y.values + v
                nexy = exy
                nvort = vort
                ne_temp = e_temp

                first_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                first_frame = np.ones(len(first_pnums))*B_img_no
                first = pd.DataFrame(np.vstack((first_frame,first_pnums,
                                        nxs,nys,nexy,nvort,ne_temp)).T,\
                                        columns = ['frame','particle','x','y',
                                                   'exy','vort','e_temp'])

            # CALCULATE incremental displacement arrays for each particle
            else:
                # updated particle location grid for kd-tree
                xy_orig_update = np.array([nxs,nys]).T

                # updated kd-tree for new particle locations, using one point
                dist,ind = do_kdtree(xy_target,xy_orig_update,1)

                # calcuate displacements, strains, and vorticity within radius
                u,v,exy,vort,e_temp = np.zeros(len(xy_orig_update)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig))
                close_ind =[]
                for k in range(len(xy_orig_update)):
                    #determine location of nearest grid points
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts)>=1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig_update)):
                    if close_ind[k] != None: #ignore empty
                        u[k] = data.u.iat[close_ind[k]]
                        v[k] = data.v.iat[close_ind[k]]
                        exy[k] = data.exy.iat[close_ind[k]]
                        vort[k] = data.vort.iat[close_ind[k]]
                        e_temp[k] = data.temp_c.iat[close_ind[k]]
                nxs = nxs + u
                nys = nys + v
                nexy = exy
                nvort = vort
                ne_temp = e_temp
                # compile further displaced particle and frame information
                continuing_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                continuing_frame = np.ones(len(continuing_pnums))*B_img_no
                continuing_df = pd.DataFrame(np.vstack((continuing_frame,\
                                    continuing_pnums,nxs,nys,nexy,nvort,
                                    ne_temp)).T,\
                                    columns = ['frame','particle','x','y',
                                               'exy','vort','e_temp'])
                continuing.append(continuing_df)
            if verbose:
                frame_elapsed = time.time()-start_time
                total_time += frame_elapsed
                num_displaced = len([i for i in close_ind if i is not None])
                print('Calcuated displacements for %g '%(num_displaced) + \
                      'particles for File %05.0f, '%(fn) + \
                      'starting at Frame %05i, '%(n) + \
                      'Frames %05.0f and %05.0f '%(A_img_no,B_img_no) + \
                      'in %06.3f seconds, '%(frame_elapsed) + \
                      '%010.3f total seconds elapsed.'%(total_time))

        # compile data (may be large, >20GB)
        # keep NaNs to filter later, if needed
        df.append(pd.concat((first,pd.concat(continuing))))
        # count of particles as they are iteratively added
        rolling_particle_nos += len(parts.particle.unique())
    return pd.concat(df).reset_index(drop=True)


def particle_displacer_temperatures_dfmfront(piv_files,ero_temps,front,
                                      particles,
                                      framenumbers,
                                      surface_hdf5,
                                      begin_file,end_file,
                                      step = 2, radius=24,
                                      verbose = False,
                                      replace_nan = False,
                                      fill_array = False,
                                      mask_piv = False):
    '''
    Using Matlab PIvab output with x,y,u,v data [files], displace synthetic
        markers [particles], while simutaneously filtering PIV data that
        falls outside of the model wedge, using calcuated surfaces
        [surface_hdf5], between a range of files [begin_file:end_file]
    '''
    #initial variables and lists
    start_frames = particles.frame.unique()
    df = [] #DataFrame
    rolling_particle_nos = 0 # inital count of particles to add to
    total_time = float()
    for n in start_frames:
        # unique particles for each frame in constructed grid of particles
        parts = particles[particles.frame == n]
        # first compared B-image for dataframe
        B_frame_start = framenumbers[framenumbers.B_frame >= n].index.min()
        continuing = []
        for i, f in enumerate(piv_files[B_frame_start:end_file]):
            start_time=time.time()
            # read frame numbers from PIV output
            fn, A_img_no, B_img_no = PIV_framenumbers(f)

            # read PIV data, ignoring extra columns if they exist
            data = pd.read_csv(f,skiprows=3,usecols = [0,1,2,3],\
                               names=['x','y','u','v'])

            # fill nan with zeros, do not use if using fill_array option
            if replace_nan:
                data.loc[np.isnan(data.u)].u = 0.
                data.loc[np.isnan(data.v)].v = 0.

            # calcuate shear strain values
            U = gridize(data.x,data.y,data.u)
            V = gridize(data.x,data.y,data.v)

            # fill array nans with nearest valid neighbors
            if fill_array:
                U = fill(U)
                V = fill(V)

            # smooth velocity arrays, calcuate shear strain
            U = ndimage.median_filter(U, 4)/(pixelmm_scale*2)
            V = ndimage.median_filter(V, 4)/(pixelmm_scale*2)
            dUdY = np.gradient(U.T)[0]
            dVdX = np.gradient(V.T)[1]
            ep_xy = 0.5 * (dUdY + dVdX)
            data['exy'] = degridize(ep_xy)[:,2]

            # calcuate vorticity (Meynart, Fung, Eringen, Timoshekno, etc...)
            VO = dVdX - dUdY
            data['vort'] = degridize(VO)[:,2]

            # flip PIVlab format upside-down and shift on x-axis
            data.v = -data.v
            data.y = data.y.max() - data.y
            data.x = data.x - data.x.min()
            xy = np.array([data.x,data.y]).T

            # read  and merge temperature data, shifting coordinates
            temp_e = pd.read_csv(ero_temps[i], delimiter=' ',
                               names=['x','y','temp_c'])

            # place temps in data by merging to align with existing values
            data = data.merge(temp_e,left_on=['x','y'],right_on=['x','y'],
                       how='left')
            # this is not needed if already masked in PIVlab
            if mask_piv:
                # read surface
                surface = pd.read_hdf(surface_hdf5,'wedgetop_%05.0f'%B_img_no)
                surface.x = surface.x-xmin
                surface.y = surface.y-(images.frame_shape[0] - ymax)

                # return boolean index of points beneath surface
                subsurface_piv = find_below(surface,xy)

                # remove background velocities
                data = data.loc[subsurface_piv]
            # background displacement coordinate values for kd-tree
            xy_target = np.vstack((data.x,data.y)).T

            # INITIALIZE particle displacement arrays
            if i == 0:
                # initial particle location grid for kd-tree
                xy_orig = np.array(parts[['x','y']])

                # perform kd-tree, only using nearest PIV grid point
                dist,ind = do_kdtree(xy_target,xy_orig,1)

                # find associated displacements, strains, and vorticity
                #   within radius
                u,v,exy,vort,e_temp = np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig))
                close_ind =[]
                for k in range(len(xy_orig)):
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts)>=1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig)):
                    if close_ind[k] != None: #ignore empty
                        u[k] = data.u.iat[close_ind[k]]
                        v[k] = data.v.iat[close_ind[k]]
                        exy[k] = data.exy.iat[close_ind[k]]
                        vort[k] = data.vort.iat[close_ind[k]]
                        e_temp[k] = data.temp_c.iat[close_ind[k]]
                nxs = parts.x.values + u
                nys = parts.y.values + v
                nexy = exy
                nvort = vort
                ne_temp = e_temp

                # shifted x-value
                nxs_dfm = nxs - front[front.frame == B_img_no].x_df.values

                first_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                first_frame = np.ones(len(first_pnums))*B_img_no
                first = pd.DataFrame(np.vstack((first_frame,first_pnums,
                                        nxs,nys,nexy,nvort,ne_temp,nxs_dfm)).T,\
                                        columns = ['frame','particle','x','y',
                                                   'exy','vort','e_temp',
                                                   'x_df'])

            # CALCULATE incremental displacement arrays for each particle
            else:
                # updated particle location grid for kd-tree
                xy_orig_update = np.array([nxs,nys]).T

                # updated kd-tree for new particle locations, using one point
                dist,ind = do_kdtree(xy_target,xy_orig_update,1)

                # calcuate displacements, strains, and vorticity within radius
                u,v,exy,vort,e_temp = np.zeros(len(xy_orig_update)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig))
                close_ind =[]
                for k in range(len(xy_orig_update)):
                    #determine location of nearest grid points
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts)>=1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig_update)):
                    if close_ind[k] != None: #ignore empty
                        u[k] = data.u.iat[close_ind[k]]
                        v[k] = data.v.iat[close_ind[k]]
                        exy[k] = data.exy.iat[close_ind[k]]
                        vort[k] = data.vort.iat[close_ind[k]]
                        e_temp[k] = data.temp_c.iat[close_ind[k]]
                nxs = nxs + u
                nys = nys + v
                nexy = exy
                nvort = vort
                ne_temp = e_temp

                # shifted x-value
                f_data = front[front.frame == B_img_no]
                if len(f_data) > 0:
                    f_pos = f_data.x_df.values
                else:
                    f_pos = front[front.frame == A_img_no].x_df.values
                nxs_dfm = nxs - f_pos

                # compile further displaced particle and frame information
                continuing_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                continuing_frame = np.ones(len(continuing_pnums))*B_img_no
                continuing_df = pd.DataFrame(np.vstack((continuing_frame,\
                                    continuing_pnums,nxs,nys,nexy,nvort,
                                    ne_temp,nxs_dfm)).T,\
                                    columns = ['frame','particle','x','y',
                                               'exy','vort','e_temp','x_df'])
                continuing.append(continuing_df)
            if verbose:
                frame_elapsed = time.time()-start_time
                total_time += frame_elapsed
                num_displaced = len([i for i in close_ind if i is not None])
                print('Calcuated displacements for %g '%(num_displaced) + \
                      'particles for File %05.0f, '%(fn) + \
                      'starting at Frame %05i, '%(n) + \
                      'Frames %05.0f and %05.0f '%(A_img_no,B_img_no) + \
                      'in %06.3f seconds, '%(frame_elapsed) + \
                      '%010.3f total seconds elapsed.'%(total_time))

        # compile data (may be large, >20GB)
        # keep NaNs to filter later, if needed
        df.append(pd.concat((first,pd.concat(continuing))))
        # count of particles as they are iteratively added
        rolling_particle_nos += len(parts.particle.unique())
    return pd.concat(df).reset_index(drop=True)


def depth_calc(img_h,trajs,frame_spacing,surfname,preflipped=False,silent=True,
                          shift_surface=False,from_larger=False,skip_edge=True,
                          x_min=0,y_max=0,edge_buf=20):
    '''
    USE THIS:
    This version of the function caluates depth for each particle at each
        frame en masse, rather than calcuate an interpolation for each particle
        individually at each frame. This is roughly 1000X faster than other
        functions
    Modified for PIV pseudoparticle tracking, with simpler column format
    Modified to accept surfaces in uncropped image coordinates
    Modified to add strain components
    Modified to add temperature pass-through option
    ----------
    Parameters
    ----------
    img_h: float
        height of imput image
    trajs : trackpy trajectories
        Linked particle paths calcuated from trackpy
    frame_spacing : integer
        'distance' between frames that change in depth shoud be calcuated
    surfname : HDF5 store
        contains the model topography of every image present in experiment

    Returns
    -------
    df : Pandas DataFrame
        a DataFrame containing all corresponding data from trajs, with addition
        of depth other columns.
    '''
    # find image height for shifting
    # unique elements of frames to loop over
    unique_frames = trajs.frame.unique()[::frame_spacing]
    # store interpolation functions for entire experiment in list
    #   this is to avoid interpolation being done repeatedly unnecessarily
    s_funcs=[]
    for fr in unique_frames:
        # corrsponding section of trajs for each combo

        # read in calcuated surface for depth calcuation
        try:
            su = pd.read_hdf(surfname,'wedgetop_%05.0f'%fr).dropna(how='any')
        except:
            su = pd.read_hdf(surfname,'wedgetop_%05.0f'%0).dropna(how='any')
        # if back-of-box edge of surface has edge effects, only use surface
        #   past [edge_buf] pixels
        if skip_edge:
            su = su[edge_buf:]
        # remove any NaNs
        su.dropna(inplace=True)

        # if surfaces were calcuated on larger image and left in those coord.,
        #   shift them so that the surface starts at x=0,y=surface elevation
        #   from base
        if from_larger:
            su.x = su.x - x_min
            su.y = su.y - (img_h-y_max)

        # create univariate interpolation function for surface
        #   (some trajectories fall between pixels)
        if preflipped:
            sf = UnivariateSpline(su.x, su.y)
        else:
            sf = UnivariateSpline(su.x, img_h - su.y)
        s_funcs.append((fr,sf))
        if not silent:
            print('Generated topography function for Frame %g'%fr)
    # pu just the frame # for future use
    f_idx = [s_funcs[i][0] for i in range(len(s_funcs))]
    # loop over all possible combinations
    df = []
    for fr in unique_frames:
        sel = trajs[(trajs.frame == fr)].sort_values(by=['particle'])
        if not sel.empty:
            # calcuate depth for every point (from stored surface functions)
            depth = sel.y - s_funcs[f_idx.index(fr)][1](sel.x)
            if not silent:
                print('Calcuated Depth for %g Particles at Frame %g'%(len(depth),fr))
            # popuate DataFrame to mimic the format of trackpy DataFrames
            sel_dat = sel.join(depth.rename('depth')) # y -> depth rename
            df.append(sel_dat)
    return pd.concat(df).sort_values(by=['particle','frame']).reset_index(drop=True)


def find_nearest_spatemp(tp_df,x,y,t,num):
    '''
    Convience function to extract the nearest points in [tp_df]
        to a specified location in space and convergence [x,y,t],
        to be used to extract points that pass through said point
        at different points in the experiment. Input to [x,y,t] can
        be a single point, lists, or column arrays.
    '''
    target = tp_df[['x','y','cum_conv']]
    orig = np.array((x,y,t)).T
    return do_kdtree(target,orig,num,pykdtree=False)


def line_particles(tp_df,tp_df_fp,line_pos,line_width,subsel=False,sel_frame=0):
    '''
    Extract particles at a given depth, within a narrow [line_width] range
        of the line
    '''
    # sort first appearance by x-dimension
    tp_df_fp = tp_df_fp.sort_values(by=['x'])
    if subsel:
        # only look at particles that start at [sel_frame]
        tp_df_fp = tp_df_fp[tp_df_fp.first_frame==sel_frame]
    # get numbers of these specified particles, sorted by x-position at [sel_frame]
    tp_df_fp_x = tp_df_fp.particle
    # find subset of particles within [line_width] of [line_pos]
    line_parts = tp_df_fp[(tp_df_fp.y>=(line_pos-line_width)) & \
                          (tp_df_fp.y<=(line_pos+line_width))].particle
    return tp_df[tp_df.particle.isin(line_parts)],tp_df_fp_x


def line_particles_subsel(tp_df,tp_df_fp,xmin,xmax,line_pos,line_width,
                          sel_frame=0,frame_buf=5):
    '''
    Extract particles at a given depth, within a narrow [line_width] range
        of the line, as well as a confined length.
    Requires a DataFrame containing particle tracks, a DataFrame containing the
        locations of the first observation of each particle, and spatial limits.
    '''
    tp_df_fp = tp_df_fp.sort_values(by=['x'])
    # only look at particles that start at [sel_frame]
    tp_df_fp = tp_df_fp[(tp_df_fp.first_frame>=sel_frame-frame_buf) &\
                        (tp_df_fp.first_frame<=sel_frame+frame_buf)]
    # get numbers of these specified particles, sorted by x-position at [sel_frame]
    tp_df_fp_x = tp_df_fp.particle
    # find subset of particles within [line_width] of [line_pos]
    line_parts = tp_df_fp[(tp_df_fp.y>=(line_pos-line_width)) & \
                          (tp_df_fp.y<=(line_pos+line_width)) & \
                          (tp_df_fp.x>=xmin) & \
                          (tp_df_fp.x<=xmax)].particle
    return tp_df[tp_df.particle.isin(line_parts)],tp_df_fp_x


def line_particles_subsel_all(tp_df,tp_df_fp,xmin,xmax,line_pos,line_width):
    '''
    Extract particles at a given depth, within a narrow [line_width] range
        of the line, as well as a confined length.
    Requires a DataFrame containing particle tracks, a DataFrame containing the
        locations of the first observation of each particle, and spatial limits.
    '''
    tp_df_fp = tp_df_fp.sort_values(by=['x'])
    # get numbers of these specified particles, sorted by x-position at [sel_frame]
    tp_df_fp_x = tp_df_fp.particle
    # find subset of particles within [line_width] of [line_pos]
    line_parts = tp_df_fp[(tp_df_fp.y>=(line_pos-line_width)) & \
                          (tp_df_fp.y<=(line_pos+line_width)) & \
                          (tp_df_fp.x>=xmin) & \
                          (tp_df_fp.x<=xmax)].particle
    return tp_df[tp_df.particle.isin(line_parts)],tp_df_fp_x


def sort_tp_df(tp_df_frame,tp_df_fp_x):
    '''
    Sort one frame from a larger dataframe based on the listing of particles in
        [tp_df_fp_x], ideally the spatial sorting of particles from hinterland
        to foreland.
    '''
    sorter = list(tp_df_fp_x.values)
    sorterIndex = dict(zip(sorter,range(len(sorter))))
    tp_df_frame['rank'] = tp_df_frame['particle'].map(sorterIndex)
    return tp_df_frame


def sorted_line_particles(tp_df,tp_df_fp,frame,start_frame=0.,line_pos=60.,line_width=5.):
    '''
    Use [line_particles] and [sort_LR] functions to get a dataframe with
        positions of particles near line over time at [frame]
    '''
    lineparts,sorted_parts = line_particles(tp_df,tp_df_fp,line_pos,line_width,
                                            sel_frame=start_frame)
    sorted_tp_df = sort_tp_df(lineparts[lineparts.frame == frame],sorted_parts)
    return sorted_tp_df.sort_values(by=['rank'])


def first_appearance_length(trajectories):
    '''
    Generate a DataFrame containing the first appearance and particle length
    [in frames], along with it's physical location upon first identification
    '''
    df_A_first = {}
    for particle, particles in trajectories.groupby("particle"):
        df_A_first[particle] = \
            particles[['particle','x','y']][particles.frame == particles.frame.min()]
    df_A_first = pd.concat(df_A_first)

    df_A_first_appear = {}
    for particle, particles in trajectories.groupby("particle"):
        df_A_first_appear[particle] = particles.frame.min()
    df_A_first_appear = pd.DataFrame(list(df_A_first_appear.items()),
                                     columns=('particle','first_frame'))

    df_A_plen = {}
    for particle, particles in trajectories.groupby("particle"):
        df_A_plen[particle] = len(particles.frame)
    df_A_plen = pd.DataFrame(list(df_A_plen.items()),
                                     columns=('particle','part_len'))

    df_A_all = pd.DataFrame.merge(pd.DataFrame.merge(df_A_first,
                                                     df_A_first_appear,
                                                     on='particle'),
                                                     df_A_plen,on='particle')

    return df_A_all


    return df_A_all


def depth_diff(group):
    '''
    Cacluate the rate of change of depth for groups of particles in a DataFrame
    Define periods elsewhere outside of function.
    '''
    return group.depth.diff(periods=periods)


def depth_rate(group):
    '''
    Cacluate the rate of change of depth for groups of particles in a DataFrame
    Define periods elsewhere outside of function.
    '''
    return group.depth.diff(periods=periods)


def cumu_depth(group):
    '''
    Calcuate the cumuative sum of the change of change of depth, with negative
        values representing burial and positive representing exhumation.
    '''
    return group.dD.cumsum()


def cumu_exy(group):
    '''
    Calcuate the cumuative sum of the change of change of depth, with negative
        values representing burial and positive representing exhumation.
    '''
    return group.exy.cumsum()


def starting_depth(group):
    '''
    Cacluate the starting depth of each grouped particle
    '''
    return group.depth.head(1).values * np.ones(len(group))


def starting_max_depth_redux(df):
    '''
    Cacluate the starting depth of each grouped particle into a flattened
        list to then merge with DataFrame
    Returns starting depth, max depth, frame where max depth was reached
    '''
    grouped_df = df.groupby('particle')
    sd,md,mdf = [],[],[]
    for i in df.particle.unique():
        group = grouped_df.get_group(i)
        sd.append(group.depth.head(1).values * np.ones(len(group)))
        max_depth=group.depth.min()
        try:
            max_depth_frame=group[group.depth==max_depth].frame.values[0]
        except:
            max_depth_frame=np.nan
        md.append(max_depth * np.ones(len(group)))
        mdf.append(max_depth_frame * np.ones(len(group)))
    sd = np.concatenate(sd).ravel().tolist()
    md = np.concatenate(md).ravel().tolist()
    mdf = np.concatenate(mdf).ravel().tolist()
    return sd, md, mdf


def depth_stats_compilation(trajs):
    '''
    Add columns to trajectory database based on above applied functions.
    '''
    trajs['dD'] = \
    trajs.groupby('particle').apply(depth_rate).reset_index(drop=True)
    trajs['dD_cum'] = \
        trajs.groupby('particle').apply(cumu_depth).reset_index(drop=True)
    trajs['start_depth'] = \
        trajs['depth'].groupby(depth_trajs['particle']).transform('first')
    trajs['end_x'] = \
        trajs['x'].groupby(depth_trajs['particle']).transform('last')
    trajs['end_y'] = \
        trajs['y'].groupby(depth_trajs['particle']).transform('last')
    trajs.sort_values(by=['particle','frame'],inplace=True)
    return trajs


def cumulative_depth_stats(tp_df,calc_burex=True):
    '''
    Calculate the deepest point each particle reaches at every point of
        convergence of observation, by calcuating the cumulative mminimum
        of depth for each individual particle all at once.
    Optionally calculate exhumation, defined as the difference between depth
        and maxlimum depth;
    Optionally calculate burial, defined as the difference between depth and
        the calcuated exhumation, shifted so that their greatest depths match,
        and are in reference to the starting depth of the particle.
    Try to use these calcuations on particles that start before entering
        wedge.
    '''
    # caluate cumulative minimum depth for each particle, which is the deepest
    #   depth due to the fact that depth is negative downwards
    cum_depth = tp_df.groupby('particle').apply(lambda x: x['depth'].cummin())
    tp_df = tp_df.merge(pd.DataFrame({'cumulative_min_depth': cum_depth}),
                        left_on=['particle',tp_df.index.values],
                        right_index=True).drop(['key_1'],axis=1)
    # calcuate exhumation and burial
    if calc_burex:
        tp_df['exhumation'] = tp_df.depth - tp_df.cumulative_min_depth
        burial = tp_df.groupby('particle').apply(lambda x: \
                           (x.depth - (x.exhumation + x.depth.min())) - \
                           (x.depth - (x.exhumation + x.depth.min())).head(1).values)
        tp_df = tp_df.merge(pd.DataFrame({'burial': burial}),
                    left_on=['particle',tp_df.index.values],
                    right_index=True).drop(['key_1'],axis=1)
    return tp_df


def shift_x_def(df, def_front_x):
    '''
    Calcuate the x position of every particle in [df.x], corrected for the
    position of the deformation front at any given frame [def_front_x.x].
    '''
    # work on copy of dataframe, make new column for front-corrected particle
    #   x-values
    df_c = df.copy()
    df_c['def_front_corr_x'] = np.nan

    # loop over every frame, subtract deformation front from existing x-values
    #   place resuts in new column as to no overwrite existing x-data
    for i in df_c.frame.unique():
        if len(def_front_x[def_front_x.frame == i]) > 0:
            df_c.loc[df_c.frame == i, 'def_front_corr_x'] = \
                df_c.loc[df_c.frame == i, 'x'] - \
                float(def_front_x[def_front_x.frame == i].x)
    return df_c


def shift_x_def_simple(df, def_front_x):
    '''
    Calcuate the x position of every particle in [df.x], corrected for the
    position of the deformation front at any given frame [def_front_x.x].
    '''
    # loop over every frame, subtract deformation front from existing x-values
    #   place resuts in new column as to no overwrite existing x-data
    for i in df.frame.unique():
        df.loc[df.frame == i] = df.loc[df.frame == i, 'x'] - float(def_front_x[def_front_x.frame == i].x)
    return df


def rdp_burial_calc_forlambda(traj,ep=3,threshold=5,min_peak_dist=1):
    '''
    Use the Ramer–Douglas–Peucker algorithm to calcuate turning points in
        simplified particle trajectories, in order to calcuate the amount
        of burial after the first instance of burial.

    Also uses the modue 'peakutils' to find the first peak in anguar change
        in simplified trajectories, corresponding to the first burial of a
        particle.

    Shoud be used with a small ep [< 10], or threshold for identifying
        simplified points, otherwise trajectories may become too simple and
        lose important parts of the overall path.

    Likely only works for particles that start at constant depth and get buried,
        not for particles disovered later that only get buried.

    TO CALCULATE start of burial and start of exhumation, use:
        new_df = tp_df.groupby('particle').apply(lambda x: rdp_burial_calc_forlambda(x)[-1])
    '''
    # only run algorithm on particles that have changed in depth > threshold,
    #   else return zeros
    traj['conv_burial'] = 0.
    traj['conv_exhum'] = 0.
    if abs(traj.depth.min()-traj.depth.max()) > threshold:
        traj_arr = traj[['cum_conv','depth']].values
        simplified_trajectory = rdp(traj_arr, epsilon=ep)
        scc, sd = simplified_trajectory.T

        # Compute the direction vectors on the simplified_trajectory.
        directions = np.diff(simplified_trajectory, axis=0)
        theta = angle(directions)
        # match angle vector to length of simplified depth trajectory
        dummy_ang = np.hstack((0,theta,0))

        #only particles that move enough to have an anguar change
        if max(dummy_ang) != 0:
        # peaks in the measure of anguar change over the simplified trajectory
        #   adjust [mean_peak_dist] so quick changes in depth aren't ignored
            angle_peaks = peakutils.indexes(dummy_ang,
                                            thres=0.02/max(dummy_ang),
                                            min_dist=min_peak_dist)

            # find first peak [first burial], and calcuate depth
            if len(angle_peaks) > 0: #only particles with significant angle
                depth_at_turn = sd[angle_peaks[0]]
            else:
                depth_at_turn = np.array(0.)
            max_depth = min(sd)

            # calcuate cum_conv and depth values where depth increase
            if len(angle_peaks) > 0:
                cc_at_turn = scc[angle_peaks[0]]
            else:
                cc_at_turn = np.array(0.)
            cc_at_max_depth = scc[np.where(sd == min(sd))]
            # test if all points before and including maxlimum depth decrease,
            #   these points have no major turn in depth and undergo burial
            #   as soon as they are identified, therefore the start of burial
            #   is the start of the trajectory itself.
            if strictly_decreasing(sd[scc<=cc_at_max_depth]):
                cc_at_turn = scc[0]
                depth_at_turn = sd[0]
            burial_magnitude = depth_at_turn - max_depth
            burial_cum_conv = cc_at_max_depth - cc_at_turn
        else:
            depth_at_turn = np.array(0.)
            max_depth = np.array(0.)
            cc_at_turn = np.array(0.)
            cc_at_max_depth = np.array(0.)
            burial_magnitude = depth_at_turn - max_depth
            burial_cum_conv = cc_at_max_depth - cc_at_turn
        # only return burial of the cum_conv distance is positive (i.e., the
        #   particle is still increasing in depth)
        if isinstance(burial_cum_conv, Iterable):
            if all(burial_cum_conv) > 0:
                traj.loc[traj.cum_conv==cc_at_turn,'conv_burial']=1.
                traj.loc[traj.cum_conv==cc_at_max_depth[0],'conv_exhum']=1.
                return burial_magnitude,burial_cum_conv, depth_at_turn, \
                            max_depth, cc_at_turn, cc_at_max_depth, traj
            else:
                traj.loc[traj.cum_conv==cc_at_turn,'conv_burial']=1.
                traj.loc[traj.cum_conv==cc_at_max_depth[0],'conv_exhum']=1.
                return np.array(0.), np.array(0.), depth_at_turn, \
                            max_depth, cc_at_turn, cc_at_max_depth, traj
        else:
            traj.loc[traj.cum_conv==cc_at_turn,'conv_burial']=1.
            traj.loc[traj.cum_conv==cc_at_max_depth,'conv_exhum']=1.
            return np.array(0.), np.array(0.), depth_at_turn, \
                            max_depth, cc_at_turn, cc_at_max_depth, traj
    else:
        return np.array(0.), np.array(0.), traj


def conv_between(traj):
    '''
    Used to calcuate the amount of convergence from start of burial
        and start of exhumation in a dataframe with columns produced by
        [rdp_burial_calc_forlambda], and applied using lambdas:

        new_df = df.groupby('particle').apply(lambda x: conv_between(x))
    '''
    if len(traj[traj.conv_exhum == 1].frame.values - traj.head(1).frame.values)>0\
        and len(traj[traj.conv_burial == 1].frame.values - traj.head(1).frame.values)>0: #maybe remove
        cc_e = int(traj[traj.conv_exhum == 1].frame.values - traj.head(1).frame.values)
        cc_b = int(traj[traj.conv_burial == 1].frame.values - traj.head(1).frame.values)
        cc_shifted_e = pd.Series(traj.cum_conv).shift(periods=cc_e) - traj.cum_conv.min()
        cc_shifted_b = pd.Series(traj.cum_conv).shift(periods=cc_b) - traj.cum_conv.min()
        traj['conv_during_burial'] = cc_shifted_b
        traj['conv_during_exhumation'] = cc_shifted_e
    else:
        traj['conv_during_burial'] = np.nan
        traj['conv_during_exhumation'] = np.nan
    return traj
