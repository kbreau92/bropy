import tqdm.notebook as tqdm

class pbars:
    '''
    Class for creating multiple nested progress bars.

    Each passed kwarg creates a separate bar in jupyter.
    (display_name = int)       : Creates a bar with a max of value int.
    (display_name = iterable)  : Creates a bar with a max of value len(iterable).
    
    To update individual bars, use: pbars.bars[0].update()
    '''
    
    def __init__(self, **kwargs):
        self.bars = []
        for key,arg in kwargs.items():
            if type(arg)==int:
                bar = tqdm.tqdm(total=arg, desc=f"{key}")
            elif type(arg)==list:
                bar = tqdm.tqdm(arg, desc=f"{key}")
            self.bars.append(bar)
            
    def close(self):
        '''
        Closes all bars
        '''
        for bar in self.bars:
            bar.close()
