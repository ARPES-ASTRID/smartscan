import matplotlib.pyplot as plt
import numpy as np

def vis_map_with_path(positions,values,reduced_maps:np.ndarray,ax=None):
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(6,5))
    ax.imshow(reduced_maps[...,0],alpha=0.5)
    # ax[1].imshow(reduced_maps[...,0],alpha=0.5)
    ax.plot(positions[:,1],positions[:,0],'r-',alpha=0.5,linewidth=.5)
    ax.scatter(positions[:,1],positions[:,0],s=10)#,c=values[:,0],cmap='viridis')
    # ax[1].scatter(positions[:,1],positions[:,0],s=10)#c=values[:,0],cmap='viridis')

def plot_map_with_path_and_scatterplot(positions,values,reduced_maps):
    fig,ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].imshow(reduced_maps[...,0],alpha=0.5,origin='lower')
    ax[1].imshow(reduced_maps[...,0],alpha=0.0,origin='lower')
    ax[0].plot(positions[:,1],positions[:,0],'r-',alpha=0.5,linewidth=.5)
    ax[0].scatter(positions[:,1],positions[:,0],s=10,c='b')
    ax[1].scatter(positions[:,1],positions[:,0],s=25,c=-values[:,0],cmap='viridis',marker='s')