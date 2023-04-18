# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Year = 2022/2023
# File = strategies.py
# Short Description = file containing strategies for project to subject Applied Evolutionary Algorithms

import inspect

def two_outputs(func, inputor):
    changeif, new_value = func(*inputor)
    if changeif > 0.7:
        return new_value
    else:
        return inputor[12]
    
def deterministic(func, inputor):
    # filter implementation
    # if deterministic_says_yes():
        return func(*inputor)
    # else:
        return inputor[12]