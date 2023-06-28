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

def plot_acqui_f(gp):
    fig,ax = plt.subplots(1,2,figsize=(12,5))

    x_pred = np.empty((10000,3))
    counter = 0
    x = np.linspace(0,99,100)
    y = np.linspace(0,99,100)

    for i in x:
        for j in y:
            x_pred[counter] = np.array([15*i/100,-15+15*j/100,0])
            counter += 1

    res1 = gp.posterior_mean(x_pred)
    res2 = gp.posterior_covariance(x_pred)


    PM = np.reshape(res1["f(x)"],(100,100))
    PV = np.reshape(res2["v(x)"],(100,100))


    # fig,ax=plt.subplots(3,1,figsize=(8,6))
    # ax[0].imshow(PV)
    # ax[1].imshow(PM)
    # ax[2].imshow(a[:,2].reshape(100,250))
    # for aa in ax:
    #     aa.scatter(points[:-1,1],points[:-1,0],s=1,color='orange')
    #     aa.scatter(points[-1,1],points[-1,0],s=2,color='red')
        

    ax[0].imshow(PM)
    ax[1].imshow(PV)
    plt.show()
    