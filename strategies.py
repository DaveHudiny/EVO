# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Year = 2022/2023
# File = strategies.py
# Short Description = file containing strategies for project to subject Applied Evolutionary Algorithms

import numpy as np
where = np.array([6, 7, 8, 11, 12, 13, 16, 17, 18])

def two_outputs(func, inputor):
    changeif, new_value = func(*inputor)
    if changeif > 0.7:
        return new_value
    else:
        return inputor[12]
    
def deterministic(func, inputor):
    # filter implementation
    
    neighbourhood = inputor[where]
    mean = np.mean(neighbourhood)
    std = np.std(neighbourhood)
    if np.abs(mean - neighbourhood[4]) > std:
        return func(*inputor)
    else:
        return inputor[12]
    
def no_threshold(func, inputor):
    return func(*inputor)

def three_outputs(func, inputor):
    out1, out2, out3 = func(*inputor)
    if out1 >= 0:
        return out2
    else:
        return out3
        
def four_outputs(func, inputor):
    out1, out2, out3, out4 = func(*inputor)
    if out1 > 0.3:
        return out2
    elif out1 > -0.3:
        return out3
    else:
        return out4
