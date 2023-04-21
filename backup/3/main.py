# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Year = 2022/2023
# Short Description = main file of project to subject Applied Evolutionary Algorithms

import cgp
import os
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import pickle as pkl
import random as random

class ConstantZero(cgp.ConstantFloat):
    _def_output = "0.0"

class ConstantOne(cgp.ConstantFloat):
    _def_output = "1.0"

class DivProtected(cgp.OperatorNode):
    """A node that devides its first by its second input."""
    _arity = 2
    _def_output = "x_0 / (x_1 + 0.000001)"

class Identity(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0"

class AbsSub(cgp.OperatorNode):
    _arity = 2
    _def_output = "abs(x_0 - x_1)"

class Maxi(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0 if x_0 >= 0 else x_1"
    _def_numpy_output = "np.maximum(x_0, x_1)"

class Mini(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0 if x_0 <= 0 else x_1"
    _def_numpy_output = "np.minimum(x_0, x_1)"

class Avg(cgp.OperatorNode):
    _arity = 2
    _def_output = "(x_0 + x_1) / 2"
    

class CGP_interface():
    def __init__(self, correct, noisy, error_function, seed : int):
        self._correct = (correct / 255.0) - 0.5
        self._noisy = (noisy / 255.0) - 0.5
        self._error_function = error_function
        
        self.population_params = {"n_parents": 2, "seed": seed}
        self.genome_params = {
            "n_inputs": 25,
            "n_outputs": 2,
            "n_columns": 10,
            "n_rows": 10,
            "levels_back": 7,
            "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat, cgp.IfElse, DivProtected, ConstantOne, ConstantZero, Identity, Mini, Maxi, AbsSub, Avg),
            }
        self.ea_params = {"n_offsprings": 5, "mutation_rate": 0.3, "n_processes": 4}
        self.evolve_params = {"max_generations": 400}
    
    



    def objective(self, individual):
        func = functor(individual.to_func())
        filtered_image = nd.generic_filter(self._noisy, func.work, (5,5))
        individual.fitness = self._error_function(self._correct, filtered_image)
        return individual

class functor():
    def __init__(self, function):
        self._function = function
    
    def work(self, inputor):
        changeif, new_value = self._function(*inputor)
        if changeif > 0.7:
            return new_value
        else:
            return inputor[12]
    


def MSE(img1, img2):
    height, width = img1.shape
    # diff = np.sum(cv2.subtract(img2, img1) ** 2)
    diff = np.sum((np.asarray(img1) - np.asarray(img2)) ** 2)
    scaled = ((height*width) - diff) / (height*width)
    return scaled

def load_image(path_correct : str, path_noisy : str):
    if os.path.exists(path_correct) and os.path.exists(path_noisy):
        # Read image
        img1 = cv2.imread(path_correct)
        img2 = cv2.imread(path_noisy)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        return img1, img2
    else:
        return None, None

# print("Ahoj")
def print_image(img, path):
    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(path)

history = {}
history["champion_fitness"] = []

def recording_callback(pop):
    global iteration
    if iteration % 20 == 0:
        history["champion_fitness"].append(pop.champion.fitness)
    iteration += 1


if __name__ == "__main__":
    img1, img2 = load_image("./tshushima_small.jpg", "./tshushima_small_20percent.jpg")
    if img1 is None:
        print("Ilegální obrázek")
        exit(1)
    
    with open("seeds.txt", "r") as f:
        seeds_string = f.read()
        seeds = seeds_string.split(" ")
        best_of_the_best = None
        global iteration
        for i, seed in zip(range(10), seeds[:10]):
            interface = CGP_interface(img1, img2, MSE, int(seed))
            
            iteration = 0
            history["champion_fitness"] = []
            # print(interface.objective(np.array([[0.01,0.024,0.01],[0.024,0.9,0.024],[0.01,0.024,0.01]])))
            pop = cgp.Population(**interface.population_params, genome_params=interface.genome_params)
            ea = cgp.ea.MuPlusLambda(**interface.ea_params)
            try:
                cgp.evolve(pop, interface.objective, ea, **interface.evolve_params, print_progress=True, callback=recording_callback)
            except:
                with open('ended_randomly.pkl', 'wb') as handle:
                    pkl.dump(pop.champion, handle, protocol=pkl.HIGHEST_PROTOCOL)
            fun = functor(pop.champion.to_func())
            if best_of_the_best is None or pop.champion.fitness > best_of_the_best.fitness:
                best_of_the_best = pop.champion
            print_image(img2, "in.jpg")
            img_new = nd.generic_filter((img2 / 255.0) - 0.5, fun.work, (5,5))
            # print(MSE(img1, img_new))
            

            print_image(img_new, f"res{i}.jpg")
            with open("histories.txt", "a") as f:
                f.write(' '.join(str(aux) for aux in history["champion_fitness"]))
                f.write('\n')
            with open('best_sol.pkl', 'wb') as handle:
                pkl.dump(best_of_the_best, handle, protocol=pkl.HIGHEST_PROTOCOL)