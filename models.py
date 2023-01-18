import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from tqdm.notebook import tqdm
from numpy.random import default_rng
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrow

#Original 2D model used in 2022. Simple reaction complex, no diffusion.
class BestagonModel_v1:
    def __init__(self, params=None, size=10, periodic=True, scale=1):
        #Set periodic boundary conditions
        self.periodic = periodic
        
        #Save dictionary of model parameters
        if params: self.params = params
        else: self.params = {
                'dt' : 0.01,
                's1' : 2.48,
                'd1' : 0.48,
                'n1' : 2,
                'k1' : 0.1,
                's2' : 2.93,
                'd2' : 0.70,
                'n2' : 1,
                'k2' : 1.9}
                    
        #Calculate total number of cells
        self.n_cells = (np.array([i for i in range(1,size+1)]) * 6).sum()+1
        
        #Define matrices to hold species concentrations
        self.X = np.zeros((2, self.n_cells, 6)) + 0.001 #Membrane compartments (Species, Cell, Compartment)
        self.Y = np.zeros((2, self.n_cells)) + 0.001 #Cytoplasmic compartment (Species, Cell)
        
        #Define x,y hex positions for every cell
        cell_id = 0
        self.cells = {(0,0):cell_id} #Dictionary of hex_position:cell_index
        self.xy_pos = {cell_id:(0,0)} #Dictionary of cell_index:xy_position
        
        shifts = [(0,-1),(-1,-1),(-1,0),(0,1),(1,1),(1,0)]
        # For each ring of the model
        for i in range(1,size+1):
            #Shift outward to next ring
            h_pos = (i,i) 
            for x,y in shifts:
                for j in range(i):
                    cell_id += 1 #Increment cell_id
                    
                    #Calculate hex and xy position of new cell
                    self.cells[h_pos] = cell_id
                    x_pos = scale * h_pos[0] * (np.sqrt(3)/2)
                    y_pos = scale * (h_pos[1] - (h_pos[0]/2))
                    self.xy_pos[cell_id] = (x_pos,y_pos)
                    
                    #Update to new hex position
                    h_pos = (h_pos[0]+x, h_pos[1]+y)

    ##########################
    ### EXTERNAL FUNCTIONS ###
    ##########################
    # Perform stochastic seeding of initial species concentrations
    def seed(self, rng_seed, D, P):
        rng = default_rng(rng_seed)
        self.Y[0] = (rng.random(self.n_cells) * (D[1] - D[0])) + D[0]
        self.Y[1] = (rng.random(self.n_cells) * (P[1] - P[0])) + P[0]  
        
    # Seed model species using default species values
    def seed_default(self, rng_seed):
        self.seed(rng_seed, D=[5.0,6.0], P=[5.0,6.0])

    # Iterate model forward by n time steps by calling _update() repetetively
    def iterate(self, n, prog_bar=True):
        if prog_bar is True:
            for i in tqdm(range(n)):
                self._update(**self.params)
                
        elif prog_bar:
            prog_bar.reset()
            for i in range(n):
                self._update(**self.params)
                prog_bar.update()
                
        else:
            for i in range(n):
                self._update(**self.params)   
    
    #Display a still image of the current model state on a new fig and axis by calling _draw_current_state()
    def draw(self, figsize=10, layer=0, normalize=None, cmap='coolwarm', hide_axes=False, **kwargs):
        
        #Set up normalization factor
        if normalize: norm = plt.Normalize(normalize[0], normalize[1])
        else: norm = plt.Normalize(self.X[layer].min(), self.X[layer].max())
        
        #Build and display figure
        fig,axis = plt.subplots(figsize=(figsize*1.2,figsize))
        self._draw_current_state(axis, norm, layer, cmap, **kwargs)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis)
        if hide_axes:
            axis.set_xticklabels([])
            axis.set_yticklabels([])
        plt.show()  
                
    #Display a video of the model, alternatively calling _advance_frame() and _draw_current_state()
    def animate(self, n_frames, steps_per_frame, figsize=10, layer=0, normalize=None, cmap='coolwarm', hide_axes=False, **kwargs):
        
        #Set up normalization factor
        if normalize: norm = plt.Normalize(normalize[0], normalize[1])
        else: norm = plt.Normalize(self.Y[layer].min(), self.Y[layer].max())
        
        #Generate figure and axis, draw t=0 state to axis
        plt.rcParams["animation.html"] = "jshtml"
        fig,axis = plt.subplots(figsize=(figsize*1.2,figsize))
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis)
        self._draw_current_state(axis, norm, layer, cmap, **kwargs)
        fig.suptitle(f"Frame = 0")
        if hide_axes:
            axis.set_xticklabels([])
            axis.set_yticklabels([])
        plt.close()
        
        #Run animation
        args = [fig, axis, norm, layer, steps_per_frame, cmap, kwargs]
        return FuncAnimation(fig, self._advance_frame, frames=n_frames, fargs=args, interval=100, repeat=False)  

    #Caculate loss by calling _calculate_vectors() and passing results to _calculate_loss()
    def loss(self, layer=0, show_output=False):
        dx,dy,mag = self._calculate_vectors(layer)
        global_loss,local_loss = self._calculate_loss(dx,dy)
        if show_output:
            print("Global loss = ", global_loss)
            print("Local loss = ", local_loss)
        return local_loss
    
    #Wrapper function for .iterate, called by paramter optimization function
    def run(self, n):
        self.iterate(n, prog_bar=False)

    ##########################
    ### INTERNAL FUNCTIONS ###
    ##########################
    #Returns matrix Z, where Z[:,0,0] represents the species on the membrane opposite to X[:,0,0]
    def _get_opposing_membranes(self):
        Z = np.zeros(self.X.shape)
        for h_pos,i in self.cells.items():
            hx = h_pos[0]
            hy = h_pos[1]
            if (hx,hy+1) in self.cells: Z[:,i,0] = self.X[:, self.cells[(hx,hy+1)], 3]
            elif self.periodic: Z[:,i,0] = self.X[:, self.cells[(-hx,-hy)], 3]
            
            if (hx+1,hy+1) in self.cells: Z[:,i,1] = self.X[:, self.cells[(hx+1,hy+1)], 4]
            elif self.periodic: Z[:,i,1] = self.X[:, self.cells[(-hx,-hy)], 4]
            
            if (hx+1,hy) in self.cells: Z[:,i,2] = self.X[:, self.cells[(hx+1,hy)], 5]
            elif self.periodic: Z[:,i,2] = self.X[:, self.cells[(-hx,-hy)], 5]
            
            if (hx,hy-1) in self.cells: Z[:,i,3] = self.X[:, self.cells[(hx,hy-1)], 0]
            elif self.periodic: Z[:,i,3] = self.X[:, self.cells[(-hx,-hy)], 0]
            
            if (hx-1,hy-1) in self.cells: Z[:,i,4] = self.X[:, self.cells[(hx-1,hy-1)], 1]
            elif self.periodic: Z[:,i,4] = self.X[:, self.cells[(-hx,-hy)], 1]
            
            if (hx-1,hy) in self.cells: Z[:,i,5] = self.X[:, self.cells[(hx-1,hy)], 2]
            elif self.periodic: Z[:,i,5] = self.X[:, self.cells[(-hx,-hy)], 2]
            
        return Z

    #Draw the current model state onto the passed axis, called by draw() and animate()
    def _draw_current_state(self, axis, norm, layer, cmap, scale=1, gap=0.2, linewidth=5, xlim=(-4,4), ylim=(-4,4), add_arrows=True):
        
        #Caculate b (1/2 side length) and c (hypotenuse, distance to vertex)
        a = (scale-gap)/2
        b = a/np.sqrt(3)
        c = (2*a)/np.sqrt(3)
        
        #Draw each cell iteratively
        for h_pos,i in self.cells.items():
            #Obtain cell position in euclidian space
            x_pos,y_pos = self.xy_pos[i]
            
            #Calculate euclidian x and y positions for vertices
            x = (x_pos-c, x_pos-b, x_pos+b, x_pos+c)
            y = (y_pos+a, y_pos, y_pos-a)
            
            #Generate line segments for plotting edges
            points = np.array([[(x[1],y[0])], 
                               [(x[2],y[0])], 
                               [(x[3],y[1])], 
                               [(x[2],y[2])], 
                               [(x[1],y[2])], 
                               [(x[0],y[1])], 
                               [(x[1],y[0])]])
            segments = np.concatenate([points[:-1], points[1:]], axis=1) 
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(self.X[layer,i,:]) #Set the numeric values for each segment
            lc.set_linewidth(linewidth)
            axis.add_collection(lc)
        
        #Add colorbar and show the plot
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        
        if add_arrows:
            self._draw_arrows(axis,layer)
    
    #Returns vector-sum matrices dx,dy, and mag for each cell, called by _calculate_loss() and _draw_arrows()
    def _calculate_vectors(self,layer,scale_factor=10):
        X = self.X[layer,:,:]
        dx = ((math.sqrt(3)/2) * (X[:,1] + X[:,2] - X[:,4] - X[:,5])) / scale_factor #x component of 1, 2, 4, 5
        dy = ((X[:,0] - X[:,3]) + ((X[:,1] + X[:,5] - X[:,2] - X[:,4])/2)) / scale_factor #y component of all 6 sides
        mag = np.sqrt(dx**2 + dy**2) * 0.5 #Magnitude, used to determine head size
        return (dx,dy,mag)
    
    #Draw arrows on the passed axis, calls _calculate_vectors()
    def _draw_arrows(self,axis,layer=0):
        arrows = []
        arrow_kwargs = {'length_includes_head':True, 'color':'k'} 
        dx, dy, mag = self._calculate_vectors(layer)
        
        for h_pos,i in self.cells.items():
            #Obtain cell position in euclidian space
            x_pos,y_pos = self.xy_pos[i]
            
            #Create a new arrow object
            arrows.append(FancyArrow(x_pos, y_pos, dx[i], dy[i], head_width=mag[i], **arrow_kwargs))
        
        #Draw all arrows
        axis.add_collection(PatchCollection(arrows, match_original=True))
  
    # Update model for next time step
    def _update(self, dt, s1, d1, k1, n1, s2, d2, k2, n2):
        X = self.X
        Y = self.Y
        Z = self._get_opposing_membranes()
        dX = np.zeros(self.X.shape)
        dY = np.zeros(self.Y.shape)

        #Change in Dishevelled concentrations
        dX[0,:,:] = ((s1*Y[0,:,None]) / (1 + (X[1,:,:]/k1)**n1)) - (d1 * X[0,:,:])
        dY[0,:] = -dX[0,:,:].sum(axis=1)

        #Change in Prickle concentrations
        dX[1,:,:] = (((s2*Y[1,:,None]) * (Z[0,:,:]**n2)) / ((k2**n2) + (Z[0,:,:]**n2))) - (d2 * X[1,:,:])
        dY[1,:] = -dX[1,:,:].sum(axis=1)

        #Update species matrix
        self.X += (dX*dt)
        self.Y += (dY*dt)
            
    #Advance an animation by clearing the old axis, iterating n steps, and re-drawing. Calls _draw_current_state()        
    def _advance_frame(self, i, fig, axis, norm, layer, steps_per_frame, cmap, kwargs):
        axis.clear()
        self.iterate(steps_per_frame, prog_bar=False)
        self._draw_current_state(axis, norm, layer, cmap, **kwargs)
        fig.suptitle(f"t = {i}")
        
    #Given dx and dy matrices for every cell, calculate loss functions
    def _calculate_loss(self,dx,dy):
        global_loss = np.sqrt((dx.sum()**2) + (dy.sum()**2)) / self.n_cells
        
        local_alignments = np.zeros(self.n_cells)
        for h_pos,i in self.cells.items():
            hx = h_pos[0]
            hy = h_pos[1]
            
            local_cells = []
            
            if (hx,hy+1) in self.cells: local_cells.append(self.cells[(hx,hy+1)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx+1,hy+1) in self.cells: local_cells.append(self.cells[(hx+1,hy+1)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx+1,hy) in self.cells: local_cells.append(self.cells[(hx+1,hy)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx,hy-1) in self.cells: local_cells.append(self.cells[(hx,hy-1)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx-1,hy-1) in self.cells: local_cells.append(self.cells[(hx-1,hy-1)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx-1,hy) in self.cells: local_cells.append(self.cells[(hx-1,hy)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            local_dx = dx[local_cells].sum() / len(local_cells)
            local_dy = dy[local_cells].sum() / len(local_cells)
            local_mag = np.sqrt((local_dx**2) + (local_dy**2))
            local_alignments[i] = local_mag
        local_loss = local_alignments.sum() / self.n_cells
        
        return (global_loss,local_loss)
 
#Work in progress model adding lateral membrane diffusion. Simple reaction complex.
class BestagonModel_v2:
    def __init__(self, params=None, size=10, periodic=True, scale=1):
        #Set periodic boundary conditions
        self.periodic = periodic
        
        #Save dictionary of model parameters
        if params: self.params = params
        else: self.params = {
                'dt' : 0.01,
                's1' : 2.48,
                'd1' : 0.48,
                'n1' : 2,
                'k1' : 0.1,
                's2' : 2.93,
                'd2' : 0.70,
                'n2' : 1,
                'k2' : 1.9}
                    
        #Calculate total number of cells
        self.n_cells = (np.array([i for i in range(1,size+1)]) * 6).sum()+1
        
        #Define matrices to hold species concentrations
        self.X = np.zeros((2, self.n_cells, 6)) + 0.001 #Membrane compartments (Species, Cell, Compartment)
        self.Y = np.zeros((2, self.n_cells)) + 0.001 #Cytoplasmic compartment (Species, Cell)
        
        #Define x,y hex positions for every cell
        cell_id = 0
        self.cells = {(0,0):cell_id} #Dictionary of hex_position:cell_index
        self.xy_pos = {cell_id:(0,0)} #Dictionary of cell_index:xy_position
        
        shifts = [(0,-1),(-1,-1),(-1,0),(0,1),(1,1),(1,0)]
        # For each ring of the model
        for i in range(1,size+1):
            #Shift outward to next ring
            h_pos = (i,i) 
            for x,y in shifts:
                for j in range(i):
                    cell_id += 1 #Increment cell_id
                    
                    #Calculate hex and xy position of new cell
                    self.cells[h_pos] = cell_id
                    x_pos = scale * h_pos[0] * (np.sqrt(3)/2)
                    y_pos = scale * (h_pos[1] - (h_pos[0]/2))
                    self.xy_pos[cell_id] = (x_pos,y_pos)
                    
                    #Update to new hex position
                    h_pos = (h_pos[0]+x, h_pos[1]+y)

    ##########################
    ### EXTERNAL FUNCTIONS ###
    ##########################
    # Perform stochastic seeding of initial species concentrations
    def seed(self, rng_seed, D, P):
        rng = default_rng(rng_seed)
        self.Y[0] = (rng.random(self.n_cells) * (D[1] - D[0])) + D[0]
        self.Y[1] = (rng.random(self.n_cells) * (P[1] - P[0])) + P[0]  
        
    # Seed model species using default species values
    def seed_default(self, rng_seed):
        self.seed(rng_seed, D=[5.0,6.0], P=[5.0,6.0])

    # Iterate model forward by n time steps by calling _update() repetetively
    def iterate(self, n, prog_bar=True):
        if prog_bar is True:
            for i in tqdm(range(n)):
                self._update(**self.params)
                
        elif prog_bar:
            prog_bar.reset()
            for i in range(n):
                self._update(**self.params)
                prog_bar.update()
                
        else:
            for i in range(n):
                self._update(**self.params)   
    
    #Display a still image of the current model state on a new fig and axis by calling _draw_current_state()
    def draw(self, figsize=10, layer=0, normalize=None, cmap='coolwarm', hide_axes=False, **kwargs):
        
        #Set up normalization factor
        if normalize: norm = plt.Normalize(normalize[0], normalize[1])
        else: norm = plt.Normalize(self.X[layer].min(), self.X[layer].max())
        
        #Build and display figure
        fig,axis = plt.subplots(figsize=(figsize*1.2,figsize))
        self._draw_current_state(axis, norm, layer, cmap, **kwargs)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis)
        if hide_axes:
            axis.set_xticklabels([])
            axis.set_yticklabels([])
        plt.show()  
                
    #Display a video of the model, alternatively calling _advance_frame() and _draw_current_state()
    def animate(self, n_frames, steps_per_frame, figsize=10, layer=0, normalize=None, cmap='coolwarm', hide_axes=False, **kwargs):
        
        #Set up normalization factor
        if normalize: norm = plt.Normalize(normalize[0], normalize[1])
        else: norm = plt.Normalize(self.Y[layer].min(), self.Y[layer].max())
        
        #Generate figure and axis, draw t=0 state to axis
        plt.rcParams["animation.html"] = "jshtml"
        fig,axis = plt.subplots(figsize=(figsize*1.2,figsize))
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis)
        self._draw_current_state(axis, norm, layer, cmap, **kwargs)
        fig.suptitle(f"Frame = 0")
        if hide_axes:
            axis.set_xticklabels([])
            axis.set_yticklabels([])
        plt.close()
        
        #Run animation
        args = [fig, axis, norm, layer, steps_per_frame, cmap, kwargs]
        return FuncAnimation(fig, self._advance_frame, frames=n_frames, fargs=args, interval=100, repeat=False)  

    #Caculate loss by calling _calculate_vectors() and passing results to _calculate_loss()
    def loss(self, layer=0, show_output=False):
        dx,dy,mag = self._calculate_vectors(layer)
        return self._calculate_loss(dx,dy)
    
    #Wrapper function for .iterate, called by paramter optimization function
    def run(self, n):
        self.iterate(n, prog_bar=False)

    ##########################
    ### INTERNAL FUNCTIONS ###
    ##########################
    #Returns matrix Z, where Z[:,0,0] represents the species on the membrane opposite to X[:,0,0]
    def _get_opposing_membranes(self):
        Z = np.zeros(self.X.shape)
        for h_pos,i in self.cells.items():
            hx = h_pos[0]
            hy = h_pos[1]
            if (hx,hy+1) in self.cells: Z[:,i,0] = self.X[:, self.cells[(hx,hy+1)], 3]
            elif self.periodic: Z[:,i,0] = self.X[:, self.cells[(-hx,-hy)], 3]
            
            if (hx+1,hy+1) in self.cells: Z[:,i,1] = self.X[:, self.cells[(hx+1,hy+1)], 4]
            elif self.periodic: Z[:,i,1] = self.X[:, self.cells[(-hx,-hy)], 4]
            
            if (hx+1,hy) in self.cells: Z[:,i,2] = self.X[:, self.cells[(hx+1,hy)], 5]
            elif self.periodic: Z[:,i,2] = self.X[:, self.cells[(-hx,-hy)], 5]
            
            if (hx,hy-1) in self.cells: Z[:,i,3] = self.X[:, self.cells[(hx,hy-1)], 0]
            elif self.periodic: Z[:,i,3] = self.X[:, self.cells[(-hx,-hy)], 0]
            
            if (hx-1,hy-1) in self.cells: Z[:,i,4] = self.X[:, self.cells[(hx-1,hy-1)], 1]
            elif self.periodic: Z[:,i,4] = self.X[:, self.cells[(-hx,-hy)], 1]
            
            if (hx-1,hy) in self.cells: Z[:,i,5] = self.X[:, self.cells[(hx-1,hy)], 2]
            elif self.periodic: Z[:,i,5] = self.X[:, self.cells[(-hx,-hy)], 2]
            
        return Z

    #Draw the current model state onto the passed axis, called by draw() and animate()
    def _draw_current_state(self, axis, norm, layer, cmap, scale=1, gap=0.2, linewidth=5, xlim=(-4,4), ylim=(-4,4), add_arrows=True):
        
        #Caculate b (1/2 side length) and c (hypotenuse, distance to vertex)
        a = (scale-gap)/2
        b = a/np.sqrt(3)
        c = (2*a)/np.sqrt(3)
        
        #Draw each cell iteratively
        for h_pos,i in self.cells.items():
            #Obtain cell position in euclidian space
            x_pos,y_pos = self.xy_pos[i]
            
            #Calculate euclidian x and y positions for vertices
            x = (x_pos-c, x_pos-b, x_pos+b, x_pos+c)
            y = (y_pos+a, y_pos, y_pos-a)
            
            #Generate line segments for plotting edges
            points = np.array([[(x[1],y[0])], 
                               [(x[2],y[0])], 
                               [(x[3],y[1])], 
                               [(x[2],y[2])], 
                               [(x[1],y[2])], 
                               [(x[0],y[1])], 
                               [(x[1],y[0])]])
            segments = np.concatenate([points[:-1], points[1:]], axis=1) 
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(self.X[layer,i,:]) #Set the numeric values for each segment
            lc.set_linewidth(linewidth)
            axis.add_collection(lc)
        
        #Add colorbar and show the plot
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        
        if add_arrows:
            self._draw_arrows(axis,layer)
    
    #Returns vector-sum matrices dx,dy, and mag for each cell, called by _calculate_loss() and _draw_arrows()
    def _calculate_vectors(self,layer,scale_factor=10):
        X = self.X[layer,:,:]
        dx = ((math.sqrt(3)/2) * (X[:,1] + X[:,2] - X[:,4] - X[:,5])) / scale_factor #x component of 1, 2, 4, 5
        dy = ((X[:,0] - X[:,3]) + ((X[:,1] + X[:,5] - X[:,2] - X[:,4])/2)) / scale_factor #y component of all 6 sides
        mag = np.sqrt(dx**2 + dy**2) * 0.5 #Magnitude, used to determine head size
        return (dx,dy,mag)
    
    #Draw arrows on the passed axis, calls _calculate_vectors()
    def _draw_arrows(self,axis,layer=0):
        arrows = []
        arrow_kwargs = {'length_includes_head':True, 'color':'k'} 
        dx, dy, mag = self._calculate_vectors(layer)
        
        for h_pos,i in self.cells.items():
            #Obtain cell position in euclidian space
            x_pos,y_pos = self.xy_pos[i]
            
            #Create a new arrow object
            arrows.append(FancyArrow(x_pos, y_pos, dx[i], dy[i], head_width=mag[i], **arrow_kwargs))
        
        #Draw all arrows
        axis.add_collection(PatchCollection(arrows, match_original=True))
  
    # Update model for next time step
    def _update(self, dt, k1, k2, k3, n1, k4, k5, k6, k7, n2):
        X = self.X
        Y = self.Y
        Z = self._get_opposing_membranes()
        dX = np.zeros(self.X.shape)
        dY = np.zeros(self.Y.shape)

        #Change in Dishevelled concentrations
        dX[0,:,:] = ((k1*Y[0,:,None]) / (1 + (X[1,:,:]/k2)**n1)) - (k3 * X[0,:,:])
        dY[0,:] = -dX[0,:,:].sum(axis=1)

        #Change in Prickle concentrations
        dX[1,:,:] = k4*Y[1,:,None] + (((k5*Y[1,:,None]) * (Z[0,:,:]**n2)) / ((k6**n2) + (Z[0,:,:]**n2))) - (k7 * X[1,:,:])
        dY[1,:] = -dX[1,:,:].sum(axis=1)

        #Update species matrix
        self.X += (dX*dt)
        self.Y += (dY*dt)
            
    #Advance an animation by clearing the old axis, iterating n steps, and re-drawing. Calls _draw_current_state()        
    def _advance_frame(self, i, fig, axis, norm, layer, steps_per_frame, cmap, kwargs):
        axis.clear()
        self.iterate(steps_per_frame, prog_bar=False)
        self._draw_current_state(axis, norm, layer, cmap, **kwargs)
        fig.suptitle(f"t = {i}")
        
    #Given dx and dy matrices for every cell, calculate loss functions
    def _calculate_loss(self,dx,dy):
        local_alignments = np.zeros(self.n_cells)
        for h_pos,i in self.cells.items():
            hx = h_pos[0]
            hy = h_pos[1]
            
            local_cells = []
            
            if (hx,hy+1) in self.cells: local_cells.append(self.cells[(hx,hy+1)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx+1,hy+1) in self.cells: local_cells.append(self.cells[(hx+1,hy+1)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx+1,hy) in self.cells: local_cells.append(self.cells[(hx+1,hy)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx,hy-1) in self.cells: local_cells.append(self.cells[(hx,hy-1)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx-1,hy-1) in self.cells: local_cells.append(self.cells[(hx-1,hy-1)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            if (hx-1,hy) in self.cells: local_cells.append(self.cells[(hx-1,hy)])
            elif self.periodic: local_cells.append(self.cells[(-hx,-hy)])
            
            local_dx = dx[local_cells].sum() / len(local_cells)
            local_dy = dy[local_cells].sum() / len(local_cells)
            local_mag = np.sqrt((local_dx**2) + (local_dy**2))
            local_alignments[i] = local_mag
        local_loss = local_alignments.sum() / self.n_cells
        
        return local_loss