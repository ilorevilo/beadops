# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata as gd
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
from vedo import *

def plot_signal(df, bins=200):
    df["signal"].hist(bins = bins)

def clean_df(df):
    df.dropna(subset=['x', 'y','z'], inplace = True) #remove nan vals
    df.reset_index(drop = True, inplace = True)
    
    return df

def filter_df(df, min_signal = 50):
    """ filters dataframe with detected beads by min signal"""
    
    pstart = len(df.index)
    subdf = df[df["signal"] >= min_signal] #[sub.nlargest(nlargest, "signal") # extracting nlargest(default=5000) points with highest signal, (better results than mass)  
    pend = len(subdf.index)
    print(f"filtering particles from initially {pstart} to {pend}")
    
    return subdf
    
def scale_df(df, scale_x, scale_y, scale_z):
    """ sets scale for each dimension of position dataframe """
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
    #df = rotate_xy(df, angle)
    
    #df = remove_outlier(df)
    #df = remove_outlier(df.copy(), radius = 10, zval = 0.5)
    
    #df = subtract_z0(df, z0)
    return df

def parse_pressure_run(key):
    # parser function to rename dictkeys
    # labeling: sampleinfo..um-{pressure}mbar{run}-dz....
    
    press = key.split("mbar")[0].split("um-")[1] # parse pressure from filename
    run = int(key.split("mbar")[1].split("-dz")[0]) # parse run from filename
    newkey = press + "mbar-" + str(run)
    return newkey
    
def parse_pressure_run2(key):
    # parser function to rename dictkeys, other labelling this time...
    # labeling: sampleinfo-..um_{pressure}mbar-{run}-dz5_00_um---.
    
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

def filter_traj_sig(df):
    """ filters out possibly wrongly linked trajectories by comparing signal variation along trajectory """ 
    Npre = df.set_index("particle").index.nunique() # particles before filtering
    
    stds = df.groupby('particle')['signal'].std()
    std_std = stds.std()
    std_mean = stds.mean()
    # std_med = stds.median()

    # select all values where std of particle signal over trajectory lies below (one std over) mean of all particles signal stds
    # helps but not perfect
    # std_med might also help
    mask = (stds <= std_mean + 0.0*std_std)
    df = df.set_index("particle")[mask]
    Nafter = df.index.nunique()
    print(Nafter, "/", Npre, "trajectories filtered based on signal")
    
    return df
    
def filter_traj_mean(df, refframe = 1, distance = 100):
    """ filters out possibly wrongly linked trajectories by comparing displacement with mean displacement in neighbourhood 
        specify refframe = frame which is used for filtering
    """
    
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

    ldf['ux_outlier'] = ldf.apply(lambda row: check_local_outlier(row, ldf, colname = 'ux', d=d), axis=1)
    ldf['uy_outlier'] = ldf.apply(lambda row: check_local_outlier(row, ldf, colname = 'uy', d=d), axis=1)
    ldf['uz_outlier'] = ldf.apply(lambda row: check_local_outlier(row, ldf, colname = 'uz', d=d), axis=1)
    
    df = df[(ldf['ux_outlier']==False)&(ldf['uy_outlier']==False)&(ldf['uz_outlier']==False)]
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