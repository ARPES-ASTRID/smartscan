from tempfile import TemporaryFile
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
import yaml

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


class Plotter:

    def __init__(self, settings = dict) -> None:
        self.figure = None
        self.settings: dict = settings
        self.map_shape = self.settings["plots"]["posterior_map_shape"]
        self.task_labels = ['mean','laplace']
        self.cmap = 'viridis'
        self.gp = None

        self._x_pred = None

        # self.update()

    @property
    def extent(self) -> list[float]:
        try:
            return[*self.gp.input_space_bounds[0],*self.gp.input_space_bounds[1]]
        except:
            return [0.,1.,0.,1.]
        
    def update(
            self, 
            gp, 
            positions:np.ndarray, 
            values:np.ndarray, 
            last_spectrum=None, 
            old_aqf=None
        ):
        if gp is None:
            self.simulate_posterior()
        else:
            self.gp = gp
            self.calculate_posterior(gp, self.map_shape)
            self.pos = positions
            self.val = values
        self.draw_plots()

    def draw_plots(self):
        if self.figure is None:
            self.create_figure()

        self.draw_posterior()
        self.draw_acquisition_function()
        self.draw_spectrum()
        self.draw_laplacian()
    
        self.figure.canvas.draw()
        plt.pause(0.01)

    def draw_posterior(self):
        posteriors = [
            'posterior_mean_0', 
            'posterior_variance_0', 
            'posterior_mean_1', 
            'posterior_variance_1'
        ]
        for name in posteriors:
            if self.plots[name] is None:
                self.plots[name] = self.axes[name].imshow(
                    getattr(self, name),
                    extent=self.extent, 
                    origin='lower',
                    aspect="equal",
                    cmap=self.cmap,
                )
            else:
                self.plots[name].set_data(getattr(self, name))
                self.plots[name].set_clim(np.quantile(self.posterior_mean_0,(0.02,0.98)))
        
    def draw_acquisition_function(self):
        if self.plots['acquisition_function'] is None:
            self.plots['acquisition_function'] = self.axes['acquisition_function'].imshow(
                self.aqf,
                extent=self.extent, 
                origin='lower',
                aspect="equal",
                cmap=self.cmap,
                # alpha=0.5,
            )
            self.plots['aqf_scatter'] = self.axes['acquisition_function'].scatter(
                self.pos[:,0],self.pos[:,1],
                s=100, 
                c=self.val[:,0],
                cmap='inferno', 
                marker='x'
            )
            self.plots['aqf_last'] = self.axes['acquisition_function'].scatter(
                self.pos[-1,0],self.pos[-1,1],
                s=50, 
                c='r', 
                marker='o'
            )
        else:
            self.plots['acquisition_function'].set_data(self.aqf)
            self.plots['acquisition_function'].set_clim(np.quantile(self.aqf,(0.02,0.98)))
            self.plots['aqf_scatter'].set_offsets(self.pos)
            self.plots['aqf_scatter'].set_array(self.val[:,0])
            # self.plots['aqf_last'].set_data(self.pos[-1,0],self.pos[-1,1])

    def draw_spectrum(self):
        pass

    def draw_laplacian(self):
        pass

    def create_figure(self):
        self.figure = plt.figure(
            figsize=(12,8),
            layout='constrained',
        )
        self.axes = {
            'posterior_mean_0':     self.figure.add_subplot(341),
            'posterior_variance_0': self.figure.add_subplot(342),
            'posterior_mean_1':     self.figure.add_subplot(345),
            'posterior_variance_1': self.figure.add_subplot(346),
            'acquisition_function': self.figure.add_subplot(122),
            'spectrum':             self.figure.add_subplot(349),
            'laplacian':            self.figure.add_subplot(3,4,10),
        }
        self.plots = {
            'posterior_mean_0': None,
            'posterior_variance_0': None,
            'posterior_mean_1': None,
            'posterior_variance_1': None,
            'acquisition_function': None,
            'spectrum': None,
            'laplacian': None,
        }

        self.axes['posterior_mean_0'].set_title(f'PMean {self.task_labels[0]}')
        self.axes['posterior_mean_0'].axis('off')
        self.axes['posterior_variance_0'].set_title(f'PVar {self.task_labels[0]}')
        self.axes['posterior_variance_0'].axis('off')
        self.axes['posterior_mean_1'].set_title(f'PMean {self.task_labels[1]}')
        self.axes['posterior_mean_1'].axis('off')
        self.axes['posterior_variance_1'].set_title(f'PVar {self.task_labels[1]}')
        self.axes['posterior_variance_1'].axis('off')
        self.axes['acquisition_function'].set_title('Acquisition Function')
        # self.axes['acquisition_function'].axis('off')
        self.axes['spectrum'].set_title('Spectrum')
        self.axes['spectrum'].axis('off')
        self.axes['laplacian'].set_title('Laplacian')
        self.axes['laplacian'].axis('off')

        self.figure.show()

    def _get_x_pred(self) -> tuple[np.ndarray]:
        x_pred_0 = np.empty((np.prod(self.map_shape),3))
        x_pred_1 = np.empty((np.prod(self.map_shape),3))
        counter = 0
        x = np.linspace(0, self.map_shape[0]-1, self.map_shape[0])
        y = np.linspace(0, self.map_shape[1]-1, self.map_shape[1])
        
        lim_x = self.gp.input_space_bounds[0]
        lim_y = self.gp.input_space_bounds[1]
        
        delta_x = (lim_x[1] - lim_x[0]) / self.map_shape[0]
        delta_y = (lim_y[1] - lim_y[0]) / self.map_shape[1]

        for i in x:
            for j in y:
                # x_pred[counter] = np.array([15*i/100,-15+15*j/100,0])
                x_pred_0[counter] = np.array([
                    delta_x * i + lim_x[0],
                    delta_y * j + lim_y[0],
                    0
                ])
                x_pred_1[counter] = np.array([
                    delta_x * i + lim_x[0],
                    delta_y * j + lim_y[0],
                    1
                ])
                counter += 1
        return x_pred_0, x_pred_1
    
    @property
    def x_pred(self) -> tuple[np.ndarray]:
        if self._x_pred is None:
            self._x_pred = self._get_x_pred()
        return self._x_pred

    def calculate_posterior(self, gp, shape=None) -> None:
        """ Calculate the posterior mean and variance of a GP
        
        Args:
            gp (GaussianProcess): GP object
        """

        
        a = self.settings["acquisition_function"]["params"]["a"]
        norm = self.settings["acquisition_function"]["params"]["norm"]
        w = self.settings["acquisition_function"]["params"]["weights"]
        if w is None:
            w = (1,1)

        self.posterior_mean_0 = np.reshape(gp.posterior_mean(self.x_pred[0])["f(x)"],shape) * w[0]
        self.posterior_variance_0 = np.reshape(gp.posterior_covariance(self.x_pred[0])["v(x)"],shape) * w[0]
        self.posterior_mean_1 = np.reshape(gp.posterior_mean(self.x_pred[1])["f(x)"],shape) * w[1]
        self.posterior_variance_1 = np.reshape(gp.posterior_covariance(self.x_pred[1])["v(x)"],shape) * w[1]
        
        aqf = norm * (a * np.sqrt(w[0]*self.posterior_variance_0+w[1]*self.posterior_variance_1) +(w[0]*self.posterior_mean_0 + w[1]*self.posterior_mean_1))        
        self.aqf = np.rot90(aqf,k=-1)[:,::-1]

    def simulate_posterior(self) -> None:
        """ Simulate a posterior mean and variance
        """
        self.posterior_mean_0 = np.random.random(self.map_shape)
        self.posterior_variance_0 = np.random.random(self.map_shape)
        self.posterior_mean_1 = np.random.random(self.map_shape)
        self.posterior_variance_1 = np.random.random(self.map_shape)
        self.aqf = np.random.random(self.map_shape)
        self.pos = np.random.random((10,2))
        self.val = np.random.random((10,2))

def plot_acqui_f(
        gp, 
        fig, 
        pos, 
        val, 
        shape=None, 
        old_aqf = None, 
        last_spectrum = None,
        settings = None,
    ):
    """ Plot the acquisition function of a GP
    
    Args:
        gp (GaussianProcess): GP object
        
    Returns:
        None
    """

    positions, values = gp.x_data, gp.y_data

    if shape is None:
        shape = settings["plots"]["posterior_map_shape"]
    x_pred_0 = np.empty((np.prod(shape),3))
    x_pred_1 = np.empty((np.prod(shape),3))
    counter = 0
    x = np.linspace(0, shape[0]-1, shape[0])
    y = np.linspace(0, shape[1]-1, shape[1])
    
    lim_x = gp.input_space_bounds[0]
    lim_y = gp.input_space_bounds[1]
    
    delta_x = (lim_x[1] - lim_x[0]) / shape[0]
    delta_y = (lim_y[1] - lim_y[0]) / shape[1]

    for i in x:
        for j in y:
            # x_pred[counter] = np.array([15*i/100,-15+15*j/100,0])
            x_pred_0[counter] = np.array([
                delta_x * i + lim_x[0],
                delta_y * j + lim_y[0],
                0
            ])
            x_pred_1[counter] = np.array([
                delta_x * i + lim_x[0],
                delta_y * j + lim_y[0],
                1
            ])
            counter += 1

    PM0 = np.reshape(gp.posterior_mean(x_pred_0)["f(x)"],shape)
    PV0 = np.reshape(gp.posterior_covariance(x_pred_0)["v(x)"],shape)
    sPV0 = np.sqrt(PV0)
    PM1 = np.reshape(gp.posterior_mean(x_pred_1)["f(x)"],shape)
    PV1 = np.reshape(gp.posterior_covariance(x_pred_1)["v(x)"],shape)
    sPV1 = np.sqrt(PV1)
    
    a = settings["acquisition_function"]["params"]["a"]
    norm = settings["acquisition_function"]["params"]["norm"]
    w = settings["acquisition_function"]["params"]["weights"]
    if w is None:
        w = (1,1)
    aqf = norm * (a * np.sqrt(w[0]*PV0+w[1]*PV1) +(w[0]*PM0 + w[1]*PM1))
    aqf = np.rot90(aqf,k=-1)[:,::-1]

    # fig,ax=plt.subplots(3,1,figsize=(8,6))
    # ax[0].imshow(PV)
    # ax[1].imshow(PM)
    # ax[2].imshow(a[:,2].reshape(100,250))
    # for aa in ax:
    #     aa.scatter(points[:-1,1],points[:-1,0],s=1,color='orange')
    #     aa.scatter(points[-1,1],points[-1,0],s=2,color='red')
    if fig is None:
        fig = plt.figure('ACQ func',figsize=(12,8), layout='constrained')
    else:
        fig.clear()

    ax = [
        fig.add_subplot(331),
        fig.add_subplot(332),
        fig.add_subplot(333),
        fig.add_subplot(334),
        fig.add_subplot(335),
        fig.add_subplot(336),
        fig.add_subplot(337),
        fig.add_subplot(338),
        fig.add_subplot(339),
    ]

    # fig,ax = plt.subplots(2,2,)
    ax = np.asarray(ax).reshape(3,3)
    for i, PM, PV in zip(range(2),[PM0,PM1], [sPV0,sPV1]):
        PM = np.rot90(PM,k=-1)[:,::-1]
        PV = np.rot90(PV,k=-1)[:,::-1]
        pmmax = PM.max()
        pvmax = PV.max()
        PM /= pmmax
        PV /= pvmax
        
        ax[i,0].imshow(PM, clim=[0,1], extent=[*lim_x,*lim_y], origin='lower', aspect="equal")
        ax[i,0].set_title(f'PM: {pmmax:.3f}')
        ax[i,1].imshow(PV, clim=[0,1], extent=[*lim_x,*lim_y], origin='lower', aspect="equal")
        ax[i,1].set_title(f'PV: {a * np.sqrt(pvmax):.3f}')

        ax[i,0].scatter(positions[:,0],positions[:,1], s=20,c='r')
        ax[i,1].scatter(positions[:,0],positions[:,1], s=20,c='r')
        ax[i,0].scatter(positions[-1,0],positions[-1,1], s=30,c='white')
        ax[i,1].scatter(positions[-1,0],positions[-1,1], s=30,c='white')

    ax[0,2].imshow(np.zeros_like(PM), clim=[0,1], extent=[*lim_x,*lim_y], origin='lower', aspect="equal")
    ax[0,2].scatter(pos[:,0],pos[:,1],s = 25, c=val[:,0],cmap='viridis', marker='o')
    ax[1,2].imshow(np.zeros_like(PM), clim=[0,1], extent=[*lim_x,*lim_y], origin='lower', aspect="equal")
    ax[1,2].scatter(pos[:,0],pos[:,1],s = 25, c=val[:,1],cmap='viridis', marker='o')
    ax[0,2].scatter(pos[-1,0],pos[-1,1],s = 25, c='r', marker='o')
    ax[1,2].scatter(pos[-1,0],pos[-1,1],s = 25, c='r', marker='o')

    ax[2,0].set_title(f'Aq func {aqf.max():.2f}')
    ax[2,0].imshow(
        aqf,
        extent=[*lim_x,*lim_y], 
        origin='lower',
        clim=np.quantile(aqf,(0.01,0.99)), 
        aspect="equal",
    )
    if old_aqf is not None:
        diff = old_aqf - aqf
        ax[2,1].set_title('aqf changes')
        ax[2,1].imshow(
            diff,
            extent=[*lim_x,*lim_y], 
            origin='lower',
            cmap='bwr',
            aspect="equal"
        ) 
    if last_spectrum is not None:
        ax[2,2].imshow(last_spectrum, clim=np.quantile(last_spectrum,(0.02,0.98)), origin='lower', cmap='terrain', aspect="equal")
    # ax[i,0].figure.canvas.draw()
    # ax[i,1].figure.canvas.draw()

    plt.pause(0.01)
    return fig, aqf
    
if __name__ == "__main__":
    import time
    with open(r'C:\Users\stein\OneDrive\Documents\_Work\_code\ARPES-ASTRID\smartscan\scan_settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    
    plotter = Plotter(settings)
    plotter.update(None,None,None)
    while True:
        try:
            time.sleep(.5)
            plotter.update(None,None,None)

        except KeyboardInterrupt:
            break