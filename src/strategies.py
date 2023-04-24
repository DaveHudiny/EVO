# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Year = 2022/2023
# File = strategies.py
# Short Description = file containing strategies for project to subject Applied Evolutionary Algorithms

import numpy as np
where = np.array([6, 7, 8, 11, 12, 13, 16, 17, 18])

def two_outputs(func, inputor):
    """Implementation of two outputs strategy. It lets CGP itself to create single detection output.
       and then it uses it to select old pixel, or result of evolued filter.

    Args:
        func (function): Function evolved by CGP. Expects two output values (floats)
        inputor (list): List of 25 values (neighborhood of evaulated pixel)

    Returns:
        float: Returns new value of repaired pixel.
    """
    changeif, new_value = func(*inputor)
    if changeif > 0.7:
        return new_value
    else:
        return inputor[12]
    
def deterministic(func, inputor):
    """Detects noises with usage of mean and standard deviation in neighborhood of single standard deviation.
       If the filter detects noise, then CGP reconstructs the pixel by its neighborhood.

    Args:
        func (function): Function evolved by CGP. Expects single output value.
        inputor (list): List of 25 values (neighborhood of evaulated pixel)

    Returns:
        float: Returns new value of repaired pixel.
    """
    
    neighbourhood = inputor[where]
    mean = np.mean(neighbourhood)
    std = np.std(neighbourhood)
    if np.abs(mean - neighbourhood[4]) > std:
        return func(*inputor)
    else:
        return inputor[12]
    
def no_threshold(func, inputor):
    """Simplest strategy. It lets CGP to do everything

    Args:
        func (function): Function evolved by CGP. Expects single output value.
        inputor (list): List of 25 values (neighborhood of evaulated pixel)

    Returns:
        float: Returns new value of repaired pixel.
    """
    return func(*inputor)

def three_outputs(func, inputor):
    """This strategy works with three outputs CGP. If first output is over 0, 
       it uses second output for reconstruction, if under, it uses third output.

    Args:
        func (function): Function evolved by CGP. Expects three output values.
        inputor (list): List of 25 values (neighborhood of evaulated pixel)

    Returns:
        float: Returns new value of repaired pixel.
    """
    out1, out2, out3 = func(*inputor)
    if out1 >= 0:
        return out2
    else:
        return out3
        
def four_outputs(func, inputor):
    """This strategy works with four outputs CGP. Similar idea to strategy above.

    Args:
        func (function): Function evolved by CGP. Expects single output value.
        inputor (list): List of 25 values (neighborhood of evaulated pixel)

    Returns:
        float: Returns new value of repaired pixel.
    """
    out1, out2, out3, out4 = func(*inputor)
    if out1 > 0.3:
        return out2
    elif out1 > -0.3:
        return out3
    else:
        return out4
