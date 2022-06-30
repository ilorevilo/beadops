# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from scipy.interpolate import griddata as gd
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
from scipy import stats
from vedo import *

def plot_signal(df, bins=200):
    """ plots histogram of signal of detected beads/ particles """
    df["signal"].hist(bins = bins)

def clean_df(df):
    """ drops all rows with nan in x/y/z column and resets df index """
    df.dropna(subset=['x', 'y','z'], inplace = True) #remove nan vals
    df.reset_index(drop = True, inplace = True)
    
    return df

def filter_df(df, min_signal = 50):
    """ filters dataframe with detected beads and keeps only beads with signal > min signal"""
    
    Nstart = len(df.index) # amount of particles initially in df
    subdf = df[df["signal"] >= min_signal] #[sub.nlargest(nlargest, "signal") # extracting nlargest(default=5000) points with highest signal, (better results than mass)  
    Nend = len(subdf.index)
    print(f"filtering particles from initially {Nstart} to {Nend}")
    
    return subdf
    
def scale_df(df, scale_x, scale_y, scale_z):
    """ sets scale for each dimension of bead dataframe """
    df['x'] = df['x'] * scale_x # raises SettingWithCopyWarning: 
        # A value is trying to be set on a copy of a slice from a DataFrame.
    df['y'] = df['y'] * scale_y
    df['z'] = df['z'] * scale_z
    
    return df    

def plot_pts_vedo(dfs, zcols = None, colors=None, alpha = 1, embed = True, r=10, cam =None, axes =1, size=(1500,1500)):
    """
        plots extracted points in 3d using vedo (vtk-based)
        if embed = True: embeds into notebook, if embed = False opens in new window
        provide multiple dfs in a list to plot various points
        provide another list of colors to color dfs individually
        cols: list of names of zval to plot, generally plots z-value as function of x/y, other columns can be als plotted
    """
    
    
    pts = []
    if type(dfs) != list:
        dfs = [dfs]
    for nr, df in enumerate(dfs):
        if zcols != None:
            zcol = zcols[nr]
        else:
            zcol = 'z'
        
        xyz = df[['x','y',zcol]].values
        
        if colors != None:
            col = colors[nr]
        else:
            col = 'red'
        pts.append(Points(xyz, c = col, alpha = alpha, r=r))

    if embed == True:
        embedWindow('k3d') # try panel?
        return show(pts, camera=cam, axes=axes) 
    else:
        embedWindow(False)
        vp = Plotter(size=size, axes=axes)
        vp += pts
        vp.show(antialias=3, camera=cam) # not sure if antialias kwarg matters

def process_df_comp(raw_df, xy_scale, z_scale, min_signal, angle=None, z0=None):
    """
        convencience function which bundles all processing steps
        required input: filtering parameters extracted from sample
    """
    print("processing df")
    df = raw_df.copy()
    df = clean_df(df)
    df = filter_df(df, min_signal)
    df = scale_df(df, xy_scale, xy_scale, z_scale)

    return df

def parse_pressure_run(key):
    """ 
        parser function to rename dictkeys by extracting "pressure" and "run"
        labeling: "sampleinfo..um-{pressure}mbar{run}-dz...."
    """
    
    press = key.split("mbar")[0].split("um-")[1] # parse pressure from filename
    run = int(key.split("mbar")[1].split("-dz")[0]) # parse run from filename
    newkey = press + "mbar-" + str(run)
    return newkey
    
def parse_pressure_run2(key):
    """
        alternative parser func for datasets labeled:
        "sampleinfo-..um_{pressure}mbar-{run}-dz5_00_um---."
    """
    
    press = key.split("mbar")[0].split("um_")[1] # parse pressure from filename
    run = int(key.split("mbar-")[1].split("-dz")[0]) # parse run from filename
    newkey = press + "mbar-" + str(run)
    print(newkey)
    return newkey

def calc_disp(df):
    """ calculates displacement in relation to initial frame of tracked positions """
    df = df[["frame","x","y","z", "signal"]] # pick only relevant columns
    df = df.reset_index().sort_values(["particle","frame"]) # sorting needed for correct shift

    for disp, pos in zip(['ux','uy','uz'],['x','y','z']):
        #df[disp] = df.groupby('particle')[pos].apply(lambda x: x-x.shift(1)) # subtract position of next pressure
        df[disp] = df.groupby('particle')[pos].apply(lambda x: x-x.iloc[0]) # subtract p0 = 0 mbar reference frame
        
    #print("any nan in ux for frame 1?", np.isnan(df[df['frame']==1]['ux'].values).any()) # check if any nan vals present --> no, shift should be correct now    
    return df

def filter_traj_sig(df, zval = 0.0):
    """ 
        filters out possibly wrongly linked trajectories by comparing 
        signal variation along trajectory (= over multiple "frames")
        particle filtered out if its
        standard deviation > mean(std_all) + zval * std(std_all)
        with std_all = standard deviation of all particles
    """ 
    Npre = df.set_index("particle").index.nunique() # particles before filtering
    
    stds = df.groupby('particle')['signal'].std()
    std_std = stds.std()
    std_mean = stds.mean()
    # std_med = stds.median()

    # select all values where std of particle signal over trajectory lies below (one std over) mean of all particles signal stds
    # helps but not perfect
    # std_med might also help
    mask = (stds <= std_mean + zval*std_std)
    df = df.set_index("particle")[mask]
    Nafter = df.index.nunique()
    print(Nafter, "/", Npre, "trajectories filtered based on signal")
    df.reset_index(inplace=True)
    
    return df


def filter_traj_mean2(df, refframe = 1, radius = 100, coord = ['x','y','z'], zval = 1, metric = 'mean'):
    """
        further optimization of filter_traj_mean1, vectorizes query_radius:
        takes 16 vs. 21 s
        still to optimize: mean/ std calculation
    """
    
    Nstart = len(df['particle'].unique())

    refdf = df[df['frame']==refframe].copy() 
    # so far copied, refdf.loc[:,"idxs"] = tree.query_radius raised "SettingWithCopyWarning: "
    # probably due to setting array of indices as column value

    #refdf.reset_index(drop = True, inplace = True)
    refdf.reset_index(drop = True, inplace = True)
    
    xyz = refdf[coord].values
    tree = KDTree(xyz, metric='euclidean') # build kdtree of distances considering ax-coordinate (xyz default)
    refdf.loc[:,"idxs"] = tree.query_radius(xyz, r=radius) # returns list of indices related to coord-array

    for comp in ['ux','uy','uz']:
        if metric == 'mean':
            means = refdf["idxs"].apply(lambda x: refdf.loc[x,comp].mean())
            stds = refdf["idxs"].apply(lambda x: refdf.loc[x,comp].std())
            refdf.loc[:,comp+'_outlier'] = np.abs(refdf[comp]-means) > zval * stds

        else:
            medians = refdf["idxs"].apply(lambda x: refdf.loc[x,comp].median())
            mads = refdf["idxs"].apply(lambda x: stats.median_absolute_deviation(refdf.loc[x,comp]))
            refdf.loc[:,comp+'_outlier'] = np.abs(refdf[comp]-medians) > zval * mads
        

    #refdf['outlier'] = refdf['ux_outlier'] #| refdf['uy_outlier'] | refdf['uz_outlier']
    outlier = refdf[refdf['ux_outlier'] | refdf['uy_outlier'] | refdf['uz_outlier']]['particle'].values
    
    #outlier = refdf[refdf['ux_outlier'] == True]['particle'].values
    
    #print(type(outlier))
    #print(outlier.shape)

    #df_filt = df[(refdf['outlier']==False)]
    df_filt = df.loc[~df["particle"].isin(outlier)] 
    
    Nend = len(df_filt["particle"].unique())
    print(f"filtering particles from initially {Nstart} to {Nend}")
    
    return df_filt


def filter_traj_mean1(df, refframe = 1, radius = 100, coord = ['x','y','z'], zval = 1):
    """
        kd-mean implementation, tested on same dataset: takes 21 s vs. 40 s --> faster
    """
 
    def check_local_outlier(row, ldf, radius):

        coords = row[coord].values.reshape(1, -1) # get ax-coordinate for each point
        idx = tree.query_radius(coords, r=radius)[0] # query points in reference configuration in defined distance to xy-coordinates        
        
        # provide colnames as arg and loop over list ['ux','uy','uz']?
        ux_outlier = np.abs(row['ux'] - ldf.iloc[idx]['ux'].mean() ) > (zval * ldf.iloc[idx]['ux'].std()) # get z-value, package also available
        uy_outlier = np.abs(row['uy'] - ldf.iloc[idx]['uy'].mean() ) > (zval * ldf.iloc[idx]['uy'].std()) # get z-value, package also available
        uz_outlier = np.abs(row['uz'] - ldf.iloc[idx]['uz'].mean() ) > (zval * ldf.iloc[idx]['uz'].std()) # get z-value, package also available
        classify_outlier = ux_outlier | uy_outlier | uz_outlier
        return classify_outlier
    
    ldf = df[df['frame']==refframe]
    Nstart = len(df.index.unique())
    tree = KDTree(ldf[coord].values, metric='euclidean') # build kdtree of distances considering ax-coordinate (xyz default)

    ldf.loc[:,'outlier'] = ldf.apply(lambda row: check_local_outlier(row, ldf, radius=radius), axis=1)

    df_filt = df[(ldf['outlier']==False)]
    
    Nend = len(df_filt.index.unique())
    print(f"filtering particles from initially {Nstart} to {Nend}")
    
    return df_filt

def filter_traj_mean0(df, refframe = 1, distance = 100):
    """ filters out possibly wrongly linked trajectories by comparing displacement with mean displacement in neighbourhood 
        specify refframe = frame which is used for filtering
    """
    
    print("filtering trajectories by looking for local outlier different to mean local displacement")
    Nstart = len(df.index.unique())
    
    def check_local_outlier(row, ldf, colname, d=100):
        dist = np.sqrt((row['x']-ldf['x'])**2 + (row['y']-ldf['y'])**2 + (row['z']-ldf['z'])**2) #calculates distance to each particle
        mask = dist < d
        sel = ldf[mask]
        #print(sel.head())
        mean = sel[colname].mean() #calculate local mean of e.g. ux
        std = sel[colname].std()
        classify_outlier = np.abs(row[colname] - mean) >= std
    
        return classify_outlier
    
    ldf = df[df['frame']==refframe]
    d = distance

    ldf.loc[:,'ux_outlier'] = ldf.apply(lambda row: check_local_outlier(row, ldf, colname = 'ux', d=d), axis=1)
    ldf.loc[:,'uy_outlier'] = ldf.apply(lambda row: check_local_outlier(row, ldf, colname = 'uy', d=d), axis=1)
    ldf.loc[:,'uz_outlier'] = ldf.apply(lambda row: check_local_outlier(row, ldf, colname = 'uz', d=d), axis=1)
    
    df = df[(ldf['ux_outlier']==False)&(ldf['uy_outlier']==False)&(ldf['uz_outlier']==False)]
    
    Nend = len(df.index.unique())
    print(f"filtering particles from initially {Nstart} to {Nend}")
    return df

def interpolate_linear(df, grid):
    """ interpolates for each frame on specified grid, 
        input df columns: frame, x, y, z, ux, uy, uz
        each scalar ux, uy, uz is interpolated separately
        returns dict frame:interpolated_arrays [ux/uy/uz=0/1/2, x, y, z]
    """
    interp_disp = {}
    
    nframes = df['frame'].unique()
    for frame in nframes:
        print("interpolating frame", frame)
        pts = df[df["frame"]==frame][["x","y","z"]].values
        ux = df[df["frame"]==frame]["ux"].values
        uy = df[df["frame"]==frame]["uy"].values
        uz = df[df["frame"]==frame]["uz"].values
        
        method = 'linear'
        ux_interp = gd(pts, ux, (grid[0], grid[1], grid[2]), method = method)
        uy_interp = gd(pts, uy, (grid[0], grid[1], grid[2]), method = method)
        uz_interp = gd(pts, uz, (grid[0], grid[1], grid[2]), method = method)
        interp_disp[frame] = np.array([ux_interp, uy_interp, uz_interp]) # best way to stack array? or along other axis?
        # np.array(list of arrays) is same as np.stack(list of arrays)
    
    return interp_disp

def interpolate_Rbf(df, grid):
    interp_disp = {}
    
    nframes = df['frame'].unique()
    for frame in nframes:
        pts = df[df["frame"]==frame][["x","y","z"]].values
        ux = df[df["frame"]==frame]["ux"].values
        uy = df[df["frame"]==frame]["uy"].values
        uz = df[df["frame"]==frame]["uz"].values
        # print(pts.shape)
        ux_rbfi = Rbf(pts[:,0], pts[:,1], pts[:,2], ux)
        uy_rbfi = Rbf(pts[:,0], pts[:,1], pts[:,2], uy)
        uz_rbfi = Rbf(pts[:,0], pts[:,1], pts[:,2], uz)
        
        igridspacing = 100
        grid_x, grid_y, grid_z = np.mgrid[0:1300:igridspacing, 0:1300:igridspacing, 0:1000:igridspacing]
        # too dense grid causes memory issue:
        # https://stackoverflow.com/questions/11865378/python-memoryerror-in-scipy-radial-basis-function-scipy-interpolate-rbf
        # -> reduce datapoints
        ux_interp = ux_rbfi(grid[0], grid[1], grid[2])
        uy_interp = uy_rbfi(grid[0], grid[1], grid[2])
        uz_interp = uz_rbfi(grid[0], grid[1], grid[2])
        
        #grid = np.array([grid_x, grid_y, grid_z])
        mask = mask_grid(grid, pts)
        interp = np.array([ux_interp, uy_interp, uz_interp])
        interp[:,~mask] = np.nan
        
        interp_disp[frame] = interp
    
    return interp_disp        

def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])
    
def calc_compression(dfield, igridspacing):
    # correct units so far only if same gridspacing for x, y and z used. Otherwise divergence has to be adjusted!
    ux_interp, uy_interp, uz_interp = dfield[0], dfield[1], dfield[2]
    comp = (divergence([ux_interp, uy_interp, uz_interp])/igridspacing)*100
    return comp

def calc_compression2(dfield, igridspacing):
    
    Jacobi = np.empty((*dfield.shape[1:],3,3))
    Jacobi.fill(np.nan) # create empty array of correct dimensions
    
    # assemble Jacobi from partial derivatives
    for i in [0,1,2]:
        for j in [0,1,2]:
            Jacobi[:,:,:,i,j] = np.gradient(dfield[i], axis = j)/igridspacing
    
    # test, should yield same compression as divergence
    #Jacobi[:,:,:,0,1:2] = 0
    #Jacobi[:,:,:,1,0] = 0
    #Jacobi[:,:,:,1,2] = 0
    #Jacobi[:,:,:,2,0:1] = 0
    
    for i in [0,1,2]:
        Jacobi[:,:,:,i,i] = 1 + Jacobi[:,:,:,i,i]
    
    Jdet = np.linalg.det(Jacobi)
    
    #Jdet = Jacobi[:,:,:,0,0] + Jacobi[:,:,:,1,1] + Jacobi[:,:,:,2,2] # same result as divergence (considering Jacobi without added +1 on diagonals)
    #Jdet = Jacobi[:,:,:,0,0] * Jacobi[:,:,:,1,1] * Jacobi[:,:,:,2,2]
    
    #comp = (Jdet/igridspacing)*100
    comp = (Jdet-1)*100
    return comp

def mask_grid(cgrid, pts):
    """ 
        returns masked input of coordinate-grid (created by mgrid, shape 3 x Nx x Ny x Nz) which corresponds to convex hull of inputpoints
        cgrid created by mgrid
        pts in shape Nx3 (xyz coordinates)
    """
    
    shape = cgrid[0].shape
    triangles = Delaunay(pts)
    tgrid = cgrid.T.reshape(-1, 3) # reshape grid to Nx3 shape
    print(tgrid.shape)
    in_triangle = triangles.find_simplex(tgrid) # check each tgrid point if inside
    mask = (in_triangle != -1) # all inside points != -1
    mask = mask.reshape(shape[::-1]).T
    # plt.imshow(mask.reshape(shape), origin = 'lower', extent = [0,1,0,1]) # compare mask and points
    
    return mask
    

def plot_xslice(df, xpos, frame, width = 25, scale = 0.3, outputfolder=None, xlim=[0,1300], ylim=[0,800]):
    """ 
        quiverplot of displacement components. selects vectors in specified width around xpos
        scale = scale of quiver arrows
        if outputfolder != None -> saves output to specified folder (name chosen automatically)
    """
    
    sdf = df[df["frame"]==frame]
    sdf = sdf[sdf['x'].between(xpos-width,xpos+width)]
    quivarr = sdf[["y","z","uy","uz"]].values
    
    fig, ax = plt.subplots(dpi=150)
    quiv = ax.quiver(quivarr[:,0], quivarr[:,1],quivarr[:,2],quivarr[:,3], scale = scale, units='xy')
    ax.set_title("yz-plane, x = " + str(xpos) + " $\pm $" + str(width) +
                 " µm, frame = " + str(frame) + ", scale = " + str(scale))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    if outputfolder is not None:
        fname = "yz_x" + str(xpos) + "_f" + str(frame) + ".png"
        fig.savefig(outputfolder + "/" + fname)
        plt.cla()
        plt.close(fig)
    
def plot_yslice(df, ypos, frame, width = 25, scale = 0.3, outputfolder=None, xlim=[0,1300], ylim=[0,800]):
    """ quiverplot of extracted displacement components. selects vectors in specified width around ypos"""
    
    sdf = df[df["frame"]==frame]
    sdf = sdf[sdf['y'].between(ypos-width,ypos+width)]
    quivarr = sdf[["x","z","ux","uz"]].values
    
    fig, ax = plt.subplots(dpi=150)
    quiv = ax.quiver(quivarr[:,0], quivarr[:,1],quivarr[:,2],quivarr[:,3], scale = scale, units='xy')
    ax.set_title("xz-plane, y = " + str(ypos) + " $\pm $" + str(width) +
                 " µm, frame = " + str(frame) + ", scale = " + str(scale))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    if outputfolder is not None:
        fname = "xz_y" + str(ypos) + "_f" + str(frame) + ".png"
        fig.savefig(outputfolder + "/" + fname)
        plt.cla()
        plt.close(fig)        
        
def plot_zslice(df, zpos, frame, width = 25, scale = 0.3, outputfolder=None, xlim=[0,1300], ylim=[0,1300]):
    """ quiverplot of extracted displacement components. selects vectors in specified width around zpos"""
    
    sdf = df[df["frame"]==frame]
    sdf = sdf[sdf['z'].between(zpos-width,zpos+width)]
    quivarr = sdf[["x","y","ux","uy"]].values
    
    fig, ax = plt.subplots(dpi=150)
    quiv = ax.quiver(quivarr[:,0], quivarr[:,1],quivarr[:,2],quivarr[:,3], scale = scale, units='xy')
    ax.set_title("xy-plane, z = " + str(zpos) + " $\pm $" + str(width) +
                 " µm, frame = " + str(frame) + ", scale = " + str(scale))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    
    if outputfolder is not None:
        fname = "xy_z" + str(zpos) + "_f" + str(frame) + ".png"
        fig.savefig(outputfolder + "/" + fname)        
        plt.cla()
        plt.close(fig)
        
def plot_zslice_int(intfield, grid, compression=None, zpos = 400, outputfolder=None, vmin=0, vmax=20, scale=1):
    """ 
        plot slice of interpolated displacement field
        divergence overlain with quiverplot of interpolated displacement components
        igridspace: spacing in µm of interpolation grid
        for instance: do interpolation here, todo: not efficient change in future to do only once!
    """
    
    fig, ax = plt.subplots()
    
    igridspacing = grid[0,1,0,0]-grid[0,0,0,0] # assume equidistant spacing in xyz
    #print(igridspacing)
    plane = int(zpos/igridspacing) #20
    #sh = int(igridspacing/2) # shift of quiver display
    sh = 0 # omitted -> specify compression extent
    quiv = ax.quiver(grid[0,:,:,plane]+sh, grid[1,:,:,plane]+sh, intfield[0,:,:,plane], intfield[1,:,:,plane], 
        units='xy', angles = 'xy', pivot = 'mid', scale_units = 'xy', width = 5, scale = scale)#, headwidth = 2, headlength = 2)# imshow conventions of x/y-axis?
    
    if compression is not None:
        im0 = ax.imshow(-compression[:,:,plane].T, vmin = vmin, vmax = vmax, origin = 'lower', cmap='inferno_r')
        cbar = fig.colorbar(im0)    
        cbar.ax.set_title('Compression [%]')


        xmin, xmax = grid[0,0,0,0]-igridspacing/2, grid[0,-1,0,0] + igridspacing/2
        ymin, ymax = grid[0,0,0,0]-igridspacing/2, grid[1,0,-1,0] + igridspacing/2
        #zmin, zmax = grid[0,0,0,0]-igridspacing/2, grid[2,0,0,-1] + igridspacing/2
        im0.set_extent([xmin, xmax, ymin, ymax])
    
    ax.set_aspect(1)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    ax.set_xlabel('x [µm]')
    ax.set_ylabel('y [µm]')
    
    #ax.set_title("Interpolated displacement field")
    
    
    if outputfolder is not None:
        fname = "xy_z" + str(zpos) + "_f" + str(frame) + ".png"
        fig.savefig(outputfolder + "/" + fname)        
        plt.cla()
        plt.close(fig)
    
    else:
        return fig, ax
        
def plot_xslice_int(intfield, grid, compression, xpos = 400, outputfolder=None, vmin=0, vmax=20, scale=1):
    """ 
        plot slice of interpolated displacement field
        divergence overlain with quiverplot of interpolated displacement components
        igridspace: spacing in µm of interpolation grid
        for instance: do interpolation here, todo: not efficient change in future to do only once!
    """
    
    fig, ax = plt.subplots()
    
    igridspacing = grid[0,1,0,0]-grid[0,0,0,0] # assume equidistant spacing in xyz
    print(igridspacing)
    plane = int(xpos/igridspacing) #20
    #sh = int(igridspacing/2) # shift of quiver display
    sh = 0 # omitted -> specify compression extent
    quiv = ax.quiver(grid[1,plane,:,:]+sh, grid[2,plane,:,:]+sh, intfield[1,plane,:,:], intfield[2,plane,:,:], 
        units='xy', angles = 'xy', pivot = 'mid', scale_units = 'xy', width = 5, scale = scale)#, headwidth = 2, headlength = 2)# imshow conventions of x/y-axis?
    
    if compression is not None:
        im0 = ax.imshow(-compression[plane,:,:].T, vmin = vmin, vmax = vmax, origin = 'lower', cmap='inferno_r')
        cbar = fig.colorbar(im0)    
        cbar.ax.set_title('Compression [%]')


        #xmin, xmax = grid[0,0,0,0]-igridspacing/2, grid[0,-1,0,0] + igridspacing/2
        ymin, ymax = grid[0,0,0,0]-igridspacing/2, grid[1,0,-1,0] + igridspacing/2
        zmin, zmax = grid[0,0,0,0]-igridspacing/2, grid[2,0,0,-1] + igridspacing/2
        im0.set_extent([ymin, ymax, zmin, zmax])
    
    ax.set_aspect(1)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    ax.set_xlabel('y [µm]')
    ax.set_ylabel('z [µm]')
    
    #ax.set_title("Interpolated displacement field")
    
    
    if outputfolder is not None:
        fname = "xy_z" + str(zpos) + "_f" + str(frame) + ".png"
        fig.savefig(outputfolder + "/" + fname)        
        plt.cla()
        plt.close(fig)
    
    else:
        return fig, ax
        
def plot_yslice_int(intfield, grid, compression, planepos = 400, outputfolder=None, vmin=0, vmax=20, scale=1, planeax = 'y'):
    """ 
        plot slice of interpolated displacement field
        divergence overlain with quiverplot of interpolated displacement components
    """
    
    fig, ax = plt.subplots()
    
    igridspacing = grid[0,1,0,0]-grid[0,0,0,0] # assume equidistant spacing in xyz
    print(igridspacing)
    plane = int(planepos/igridspacing) #20
    #sh = int(igridspacing/2) # shift of quiver display
    sh = 0 # omitted -> specify compression extent
    
    if planeax == 'y':
        xyuv = grid[0,:,plane,:]+sh, grid[2,:,plane,:]+sh, intfield[0,:,plane,:], intfield[2,:,plane,:]
    
    # expand for other planes
        
    quiv = ax.quiver(*xyuv, 
        units='xy', angles = 'xy', pivot = 'mid', scale_units = 'xy', width = 5, scale = scale)#, headwidth = 2, headlength = 2)# imshow conventions of x/y-axis?
    
    if compression is not None:
        
        if planeax == 'y':
            im0 = ax.imshow(-compression[:,plane,:].T, vmin = vmin, vmax = vmax, origin = 'lower', cmap='inferno_r')

            xmin, xmax = grid[0,0,0,0]-igridspacing/2, grid[0,-1,0,0] + igridspacing/2
            #ymin, ymax = grid[0,0,0,0]-igridspacing/2, grid[1,0,-1,0] + igridspacing/2
            zmin, zmax = grid[0,0,0,0]-igridspacing/2, grid[2,0,0,-1] + igridspacing/2
            
            extent = [xmin, xmax, zmin, zmax]
            
        cbar = fig.colorbar(im0)    
        cbar.ax.set_title('Compression [%]')



        im0.set_extent(extent)
    
    ax.set_aspect(1)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    if planeax == 'y':
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('z [µm]')
    
    #ax.set_title("Interpolated displacement field")
    
    
    if outputfolder is not None:
        fname = "xy_z" + str(zpos) + "_f" + str(frame) + ".png"
        fig.savefig(outputfolder + "/" + fname)        
        plt.cla()
        plt.close(fig)
    
    else:
        return fig, ax