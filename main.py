# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Short Description = main file of project to subject Applied Evolutionary Algorithms

import cgp
import os
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp

class CGP_interface():
    def __init__(self, correct, noisy, error_function):
        self._correct = correct
        self._noisy = noisy
        self._error_function = error_function
        
        self.population_params = {"n_parents": 10, "seed": 8188211}
        self.genome_params = {
            "n_inputs": 9,
            "n_outputs": 1,
            "n_columns": 5,
            "n_rows": 5,
            "levels_back": 3,
            "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.Div, cgp.ConstantFloat),
            }
        self.ea_params = {"n_offsprings": 10, "mutation_rate": 0.03, "tournament_size": 2, "n_processes": 2}
        self.evolve_params = {"max_generations": 1000, "min_fitness": 0.0}
        


    def objective(self, individual):
        return self._error_function(self._correct, apply_image_filter(individual, self._noisy))
        

def apply_image_filter(kernel, image):
    new_image = cv2.filter2D(image,kernel=kernel, ddepth=-1)
    return new_image

def MAE(img1, img2):
    height, width = img1.shape
    diff = np.sum(np.abs(cv2.subtract(img1, img2)))
    scaled = diff/(height*width)
    return scaled

def MSE(img1, img2):
    height, width = img1.shape
    diff = np.sum(cv2.subtract(img1, img2) ** 2)
    scaled = diff/(height*width)
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
    plt.imshow(img1,cmap="gray")
    plt.axis("off")
    plt.savefig(path)

if __name__ == "__main__":
    img1, img2 = load_image("./ghostrunner_greyscale.jpg", "./ghostrunner_noise.jpg")
    if img1 is None:
        print("Ilegální obrázek")
        exit(1)
    interface = CGP_interface(img1, img2, MAE)
    print(interface.objective(np.array([[0.01,0.024,0.01],[0.024,0.9,0.024],[0.01,0.024,0.01]])))
    
    pop = cgp.Population(**interface.population_params, genome_params=interface.genome_params)
    ea = cgp.ea.MuPlusLambda(**interface.ea_params)

    history = {}
    history["fitness_parents"] = []
    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())
    cgp.evolve(pop, interface.objective, ea, **interface.evolve_params, print_progress=True, callback=recording_callback)
    