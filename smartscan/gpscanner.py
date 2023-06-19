import numpy as np

from .scanner import SmartScan
from gpcam.gp_optimizer import fvGPOptimizer


class SmartScanGP(SmartScan):

    _GP_PARAMS = [
        'input_space_dimension',
        'output_space_dimension',
        'output_number',
        'input_space_bounds',
    ]
    _TRAIN_GP_PARAMS = [
        'hyperparameter_bounds',
        'pop_size',
        'tolerance',
        'max_iter',
    ]

    def __init__(
            self,
            host: str,
            port: int,
            gp:fvGPOptimizer = None,
            **kwargs,
        ) -> None:
        super().__init__(host=host,port=port,**kwargs)
        self.gp = gp

        self.retraining_interval = kwargs.pop('retraining_interval', 10)
        # gp parameters
        self.input_space_dimension = kwargs.pop('input_space_dimension',2)
        self.output_space_dimension = kwargs.pop('output_space_dimension',1)
        self.output_number = kwargs.pop('output_number',2)
        self.input_space_bounds = None
        # gp init_fvgp params
        self.init_hyperparameters = kwargs.pop('init_hyperparameters',None)
        self.device = kwargs.pop('device',None)
        self.gp_kernel_function = kwargs.pop('gp_kernel_function',None)
        self.gp_mean_function = kwargs.pop('gp_mean_function',None)
        self.use_inv = kwargs.pop('use_inv',True)
        self.ram_economy = kwargs.pop('ram_economy',True)
        # 
        self.hyperparameter_bounds = kwargs.pop('hyperparameter_bounds',None)
        self.pop_size = kwargs.pop('pop_size',20)
        self.tolerance = kwargs.pop('tolerance',1e-6)
        self.max_iter = kwargs.pop('max_iter',2)
        # train gp kwargs
        self.hyperparameter_bounds = kwargs.pop('hyperparameter_bounds',None)
        self.max_iter = kwargs.pop('max_iter', 10000)
        self.dask_client = kwargs.pop('dask_client', None)
        self.deflation_radius = kwargs.pop('deflation_radius', None)
        self.constraints = kwargs.pop('constraints', ())
        self.local_method = kwargs.pop('local_method', "L-BFGS-B")
        self.global_method = kwargs.pop('global_method', "genetic")
        # ask kwargs
        self.ask_method = kwargs.pop('ask_method', 'global')#Literal['global','local']
        self._value_positions = kwargs.pop('value_positions',None)

    @property
    def value_positions(self) -> np.ndarray:
        if self._value_positions is not None  and len(self._value_positions) == len(self.points):
            return self._value_positions
        elif self._value_positions is None :
            return np.array([[[0],[1]]]*len(self.points))
        else:
            raise ValueError("Wrong value positions")

    def update_gp_params(self,**kwargs) -> None:
        unknown_args = []
        for k,v in kwargs.items():
            if hasattr(self,k):
                setattr(self,k,v)
            else:
                unknown_args.append(k)
        if len(unknown_args) > 0:
            raise ValueError(f"Unknown arguments {k}")

    def init_gp(
        self,
        **kwargs,
    ) -> fvGPOptimizer:
        """ Initialize the GP
        Args:
            kwargs: keyword arguments for the gp

        Returns:
            gp: the initialized gp
        """
        assert len(self.points) > 0, "No data to train the GP"

        # get kwargs for the gp
        if self.input_space_bounds is None: # get info from the scan settings
            self.input_space_bounds = self.limits

        self.update_gp_params(**kwargs)
        
        self.gp = fvGPOptimizer(
            input_space_dimension = self.input_space_dimension,
            output_space_dimension = self.output_space_dimension,
            output_number = self.output_number,
            input_space_bounds = self.input_space_bounds,
        )
        self.tell_gp()
        self.gp.init_fvgp(
            init_hyperparameters=self.init_hyperparameters,
            compute_device=self.device,
            gp_kernel_function=self.gp_kernel_function,
            gp_mean_function=self.gp_mean_function,
            use_inv=self.use_inv,
            ram_economy=self.ram_economy,
            )
        self.train_gp()
        # return self.gp/

    def tell_gp(self) -> None:
        assert self.gp is not None, "No GP initialized"
        self.gp.tell(self.points,self.reduced_values)
    
    def train_gp(self, **kwargs) -> None:
        f""" Train the GP
        
        {fvGPOptimizer.train_gp.__doc__}
        """
        assert self.gp is not None, "No GP initialized"
        self.update_gp_params(**kwargs)
        self.gp.train_gp(
            hyperparameter_bounds = self.hyperparameter_bounds,
            pop_size = self.pop_size,
            tolerance = self.tolerance,
            max_iter = self.max_iter,
        )

    def train_gp_async(
        self,
        **kwargs
    ) -> None:
        f""" Train the GP

        {fvGPOptimizer.train_gp_async.__doc__}
        """
        assert self.gp is not None, "No GP initialized"
        self.update_gp_params(**kwargs)
        self.gp.train_gp_async(
            hyperparameter_bounds=self.hyperparameter_bounds,
            max_iter = self.max_iter,
            dask_client = self.dask_client,
            deflation_radius = self.deflation_radius,
            constraints = self.constraints,
            local_method = self.local_method,
            global_method = self.global_method,
        )

    def ask_gp(
            self, 
            n_points:int = 1, 
            **kwargs
        ) -> np.ndarray:
        return self.gp.ask(
            position = self.positions[-1], 
            n = n_points, 
            acquisition_function = self.acquisition_function, 
            bounds = self.input_space_bounds,
            method= self.ask_method, 
            pop_size = self.pop_size, 
            max_iter = self.max_iter, 
            tol = self.tolerance,
            x0 = None, 
            dask_client = self.dask_client,
        )

    def evaluation(self, iteration:int) -> None:
        """ Evaluate the GP at the current iteration """
        if self.gp is None:
            self.init_gp()
            
        self.tell_gp()
        if iteration % self.retrain_interval == 0 or iteration in self.retrain_at:
            self.train_gp_async()
        next = self.ask_gp()
        next_on_grid = self.file.nearest(next)
        return next_on_grid
    