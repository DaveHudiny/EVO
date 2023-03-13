# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Short Description = main file of project to subject Applied Evolutionary Algorithms

import cgp
import os
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd

class constantZero(cgp.ConstantFloat):
        _def_output = "0.0"

class constantFull(cgp.ConstantFloat):
    _def_output = "255.0"

class constantOne(cgp.ConstantFloat):
    _def_output = "1.0"

class DivProtected(cgp.OperatorNode):
    """A node that devides its first by its second input."""

    _arity = 2
    _def_output = "x_0 / (x_1 + 0.000001)"

class CGP_interface():
    def __init__(self, correct, noisy, error_function):
        self._correct = correct
        self._noisy = noisy
        self._error_function = error_function
        
        self.population_params = {"n_parents": 10, "seed": 10}
        self.genome_params = {
            "n_inputs": 9,
            "n_outputs": 1,
            "n_columns": 10,
            "n_rows": 10,
            "levels_back": 7,
            "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat, cgp.IfElse, DivProtected, constantFull, constantZero),
            }
        self.ea_params = {"n_offsprings": 10, "mutation_rate": 0.2, "tournament_size": 2, "n_processes": 4}
        self.evolve_params = {"max_generations": 10}
    
    



    def objective(self, individual):
        func = functor(individual.to_func())
        filtered_image = nd.generic_filter(self._noisy, func.work, (3,3))
        individual.fitness = self._error_function(self._correct, filtered_image)
        return individual

class functor():
    def __init__(self, function):
        self._function = function
    
    def work(self, inputor):
        return self._function(*inputor)
    


def apply_image_filter(kernel, image):
    new_image = cv2.filter2D(image,kernel=kernel, ddepth=-1)
    return new_image

def MAE(img1, img2):
    height, width = img1.shape
    diff = np.sum(np.abs(cv2.subtract(img1, img2)))
    scaled = diff/(height*width)
    return -scaled

def MSE(img1, img2):
    height, width = img1.shape
    diff = np.sum(cv2.subtract(img2, img1) ** 2)
    print(cv2.subtract(img2, img1))
    scaled = (height*width*255) - diff
    return scaled

def load_image(path_correct : str, path_noisy : str):
    if os.path.exists(path_correct) and os.path.exists(path_noisy):
        # Read image
        img1 = cv2.imread(path_correct)
        img2 = cv2.imread(path_noisy)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        # print(fitness(img1, np.array([[0,0,0],[0,1,0],[0,0,0]]), img2, MSE))
        # plt.figure(figsize=(8,8))
        # plt.imshow(loaded_img,cmap="gray")
        # plt.axis("off")
        # plt.savefig("image.png")
        return img1, img2
    else:
        return None, None

# print("Ahoj")
def print_image(img, path):
    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(path)

if __name__ == "__main__":
    img1, img2 = load_image("./ghostrunner_greyscale.jpg", "./ghostrunner_noise.jpg")
    if img1 is None:
        print("Ilegální obrázek")
        exit(1)
    interface = CGP_interface(img1, img2, MSE)
    # print(interface.objective(np.array([[0.01,0.024,0.01],[0.024,0.9,0.024],[0.01,0.024,0.01]])))
    print(MSE(img1, img2))
    pop = cgp.Population(**interface.population_params, genome_params=interface.genome_params)
    ea = cgp.ea.MuPlusLambda(**interface.ea_params)

    history = {}
    history["fitness_parents"] = []
    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())
    cgp.evolve(pop, interface.objective, ea, **interface.evolve_params, print_progress=True, callback=recording_callback)
    fun = functor(pop.champion.to_func())
    print_image(img2, "in.jpg")
    img_new = nd.generic_filter(img2, fun.work, (3,3))
    print(MSE(img1, img_new))
    print_image(img_new, "res.jpg")
    