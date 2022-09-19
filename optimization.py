import numpy as np
from copy import deepcopy as copy
from numpy.random import default_rng
from multiprocessing import Pool
from itertools import repeat

class Optimizer:
    '''
    Key Arguments:
        Model  : Model - Base class for the model to be used for parameter optimization
        params : dict  - Dictionary of Parameter objects, containing details on each parameter

    Hyperparameters:
        n_iterations        : int - Number of iterations per model run, passed to model.run()
        n_epochs            : int - Number of epochs to run optimization for
        n_starting_models   : int - Initial number of models to randomly generate in epoch 0
        hof_size            : int - How many models to save following each epoch
        lgd_vars            : int - How many variables to test during local gradient descent
        n_matings           : int - Number of parent matings to generate during evolution
        n_children          : int - Number of children to generate from each parent mating
        rng_seed            : int - Seed used to initialize random number generator
        n_workers           : int - Number of processor cores to use for multiprocessing
        n_base_models       : int - Number of unique starting conditions to test for every parameter set.

    Model kwargs:
        model_init_kwargs   : dict - kwargs passed to Model.__init__()
        model_seed_kwargs   : dict - kwargs passed to Model.seed()
        model_run_kwargs    : dict - kwargs passed to Model.run()
        model_loss_kwargs   : dict - kwargs passed to Model.loss()

    To be compatible with this optimizaer, a Model needs the following callable functions:
        Model.__init__(params, **model_init_kwargs)
        Model.seed(rng_seed, **model_seed_kwargs)
        Model.run(n_iterations, **model_run_kwargs)
        Model.loss(**model_loss_kwargs)
    '''
    def __init__(self, Model, params,  n_iterations, n_epochs, n_starting_models, hof_size, lgd_vars, n_matings, n_children, rng_seed, n_workers, n_base_models,
                 model_init_kwargs={}, model_seed_kwargs={}, model_run_kwargs={}, model_loss_kwargs={}):

        #Model setup parameters
        self.Model = Model                                                      #Model - Base class for model generation
        self.params = params                                                    #Dict - Dictionary of Parameter objects
        self.test_params = [key for key in params if not params[key].static]    #List - List of keys for parameters to be optimized
        
        #Initialization parameters
        self.n_workers = n_workers                                              #Int - Processor cores to use
        self.n_iterations = n_iterations                                        #Int - Iterations per model run
        self.n_starting_models = n_starting_models                              #Int - Total parameter sets to generate for epoch 0
        self.n_base_models = n_base_models                                      #Int - Number of base models iterate for each parameter set
        
        #Evolution parameters
        self.lgd_vars = lgd_vars                                                #Int - Variables to test during gradient descent
        self.n_matings = n_matings                                              #Int - How many parent model pairs to mate
        self.n_children = n_children                                            #Int - How many children to generate from each mating
        self.potential_matings = self.get_potential_matings(hof_size)           #List - List of tuples containing all possible HOF model pairs
        
        #Random number generation
        self.rng_seed = rng_seed                                                #Int - Number used to initialize random number generator
        self.rng = default_rng(rng_seed)                                        #Obj - np.random.default_rng object used for all stochastic functions
        
        #Epoch parameters
        self.epoch = 0                                                          #Int - Current epoch
        self.n_epochs = n_epochs                                                #Int - Total number of epochs to run
        
        #Models for current run
        self.base_models = []                                                   #List - List of base Model objects, all Models run for every parameter set
        self.models = []                                                        #List - List of dicts, each of which is a parameter set being tested
        self.model_scores = []                                                  #List - List of floats, final scores for each model in .models
        
        #Hall of fame models
        self.hof_size = hof_size                                                #Int - How many models to save to the HOF after each epoch
        self.hof = []                                                           #List - List of dicts, each of which is a HOF parameter set
        self.hof_scores = []                                                    #List - List of floats, scores for each mdoel in .hof
        self.hof_history = []                                                   #List - List of lists, each of which is the scores from a subsequent step

        #Model kwargs
        self.model_init_kwargs = model_init_kwargs                              #Dict - kwargs passed to Model.__init__()
        self.model_seed_kwargs = model_seed_kwargs                              #Dict - kwargs passed to Model.seed()
        self.model_run_kwargs  = model_run_kwargs                               #Dict - kwargs passed to Model.run()
        self.model_loss_kwargs = model_loss_kwargs                              #Dict - kwargs passed to Model.loss()
             
    def run(self):
        '''
        Main run function, takes no arguments.

        First epoch: 
        seed models -> iterate models -> score models, save top to HOF -> Run gradient descent for each model in HOF

        Subsequent epochs:
        Randomly mate HOF models -> iterate children -> score children, save top to HOF -> Run gradient descent for each model in HOF
        '''
        for i in range(self.n_epochs):
            print("Starting epoch ",self.epoch)
            if i==0:
                #Generate initial parameter sets randomly
                print("Seeding initial models")
                self.initialize_models()
            else:
                #Perform crossing over to generate new models in self.models
                print("Randomly mating models")
                self.evolve() 
            
            #Distribute iteration to multiprocessing worker pool, return result of score function
            print("Iterating model pool")
            self.run_models() 
            
            #Compare score function scores to current hof (if exists), save top to hall of fame
            print("Saving top models to HOF")
            self.score_models() 
            self.hof_history.append(self.hof_scores)
            
            #For each model in hof, run local gradient descent, update hof
            print("Running local gradient descent")
            self.gradient_descents() 
            self.hof_history.append(self.hof_scores)
            
            print(f"Epoch {self.epoch} complete.\n")
            self.epoch += 1
                
    def initialize_models(self):
        seeds = self.rng.integers(1, 1000000, self.n_base_models)
        for seed in seeds:
            model = self.Model(params=None, **self.model_init_kwargs)
            model.seed(rng_seed=seed, **self.model_seed_kwargs)
            self.base_models.append(model)
            
        for i in range(self.n_starting_models):
            model_params = {key:param.get_initial_value(self.rng) for (key,param) in self.params.items()}
            self.models.append(model_params)
            
    def run_models(self):
        with Pool(self.n_workers) as pool:
            self.model_scores = pool.map(self.run_model, self.models)
    
    def run_model(self, model_params):
        scores = np.zeros(self.n_base_models)
        for i,model in enumerate(self.base_models):
            model.params = model_params
            model.run(self.n_iterations, **self.model_run_kwargs)
            global_loss,local_loss = model.loss(**self.model_loss_kwargs)
            scores[i] = local_loss
        return scores.mean()
    
    def score_models(self):
        scores = np.array(self.model_scores + self.hof_scores)
        models = np.array(self.models + self.hof)

        sort_index = scores.argsort()
        self.hof_scores = scores[sort_index][-self.hof_size:].tolist()
        self.hof = models[sort_index][-self.hof_size:].tolist()
        
    def gradient_descent(self, hof_index, rng_seed):
        top_model = self.hof[hof_index]
        top_score = self.hof_scores[hof_index]
        
        rng = default_rng(rng_seed)
        test_params = rng.choice(self.test_params, size=self.lgd_vars, replace=False)
                                 
        for test_param in test_params:
            delta = self.params[test_param].get_delta(current_epoch=self.epoch, total_epochs=self.n_epochs)
            bounds = self.params[test_param].bounds
            
            #Increment the test parameter by current delta, iterate and score
            if top_model[test_param] != bounds[1]:
                upper_model = copy(top_model)
                upper_model[test_param] = min(upper_model[test_param]+delta, bounds[1])
                upper_losses = np.zeros(self.n_base_models)
                for i,base_model in enumerate(self.base_models):
                    model = copy(base_model)
                    model.params = upper_model
                    model.iterate(self.n_iterations, prog_bar=False)
                    global_loss,local_loss = model.loss(show_output=False)
                    upper_losses[i] = local_loss
                upper_loss = upper_losses.mean()
            else: upper_loss = top_score
            
            #Decrement the test parameter by current delta, iterate and score
            if top_model[test_param] != bounds[0]:
                lower_model = copy(top_model)
                lower_model[test_param] = max(lower_model[test_param]-delta, bounds[0])
                lower_losses = np.zeros(self.n_base_models)
                for i,base_model in enumerate(self.base_models):
                    model = copy(base_model)
                    model.params = lower_model
                    model.iterate(self.n_iterations, prog_bar=False)
                    global_loss,local_loss = model.loss(show_output=False)
                    lower_losses[i] = local_loss
                lower_loss = lower_losses.mean()
            else: lower_loss = top_score
            
            #Determine if either modification improved the score
            if (lower_loss > upper_loss) and (lower_loss > top_score): 
                top_score = lower_loss
                top_model = lower_model
            elif (upper_loss > lower_loss) and (upper_loss > top_score):
                top_score = upper_loss
                top_model = upper_model
            
        return (top_score, top_model)
            
    def gradient_descents(self):
        rng_seeds = self.rng.integers(1, 1000000, self.hof_size)
        with Pool(self.n_workers) as pool:
            results = pool.starmap(self.gradient_descent, enumerate(rng_seeds))
        self.hof_scores = [result[0] for result in results]
        self.hof = [result[1] for result in results]
        
    def evolve(self):
        self.models = []
        self.model_scores = []
        
        #Select random sample of n_matings
        matings = self.rng.choice(self.potential_matings, size=self.n_matings, replace=False)
        
        #For each mating
        for mating in matings:
            for i in range(self.n_children):
                child_params = {}
                for key in self.params:
                    #If param isnt static, randomly select parameter from one of parents
                    if self.params[key].static: child_params[key] = self.params[key].value
                    else: child_params[key] = self.hof[self.rng.choice(mating)][key]
                self.models.append(child_params)
                 
    def get_potential_matings(self,hof_size):
        parents = list(range(hof_size))
        matings = []
        for i in range(len(parents)-1):
            matings += list(zip(repeat(parents.pop(0),len(parents)),parents))
        return matings
            
class Parameter:
    '''
    Class for model parameters, used to constuct individual instances of parameters for model instances.

    Attributes:
        .value : float
            Set the starting value of this parameter.

        .delta : float or list
            If float, parameter will change by this amount during every gradient descent. 
            If list, delta will decrease linearly from delta[0] to delta[1] during run.
            If not given, parameter will stay static during run.

        .bounds : list
            Sets upper and lower limits on parameter.

    '''
    def __init__(self, value=None, delta=None, bounds=None):
        self.value = value
        self.bounds = bounds
        self.delta = delta
        self.deltas = None
        if delta is None: self.static = True
        else: self.static = False

    def get_initial_value(self,rng=None):
        '''
        Get the starting value for this parameter.
        If parameter has a .value attribute, this is used.
        If not, a number is randomly chosen from between .bounds
        '''
        if self.value: return self.value
        else: return ((rng.random(1)[0] * (self.bounds[1] - self.bounds[0])) + self.bounds[0])
    

    def get_delta(self, current_epoch, total_epochs=None):
        '''
        Get the current delta value based on the current epoch.
        Should be given both current_epoch and toal_epochs when called.
        Only called if delta is not None.
        '''
        if type(self.delta) == float: return self.delta
        elif type(self.delta) == list:
            if self.deltas is None: self.deltas = np.linspace(self.delta[0], self.delta[1], total_epochs)
            return self.deltas[current_epoch]
