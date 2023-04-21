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
from strategies import two_outputs, deterministic, no_threshold, three_outputs, four_outputs
import argparse
import inspect

class ConstantZero(cgp.ConstantFloat):
    _def_output = "0.0"
    _def_output_numpy = "0.0"

class ConstantOne(cgp.ConstantFloat):
    _def_output = "1.0"
    _def_output_numpy = "1.0"

class DivProtected(cgp.OperatorNode):
    """A node that devides first input by its second input."""
    _arity = 2
    _def_output = "x_0 / (x_1 + 0.000001)"
    _def_output_numpy = "x_0 / (x_1 + 0.000001)"

class Identity(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0"
    _def_output_numpy = "x_0"

class AbsSub(cgp.OperatorNode):
    _arity = 2
    _def_output = "np.abs(x_0 - x_1)"
    _def_output_numpy = "np.abs(x_0 - x_1)"

class Maxi(cgp.OperatorNode):
    _arity = 2
    _def_output = "np.maximum(x_0, x_1)"
#     _def_output_numpy = "np.max((x_0, x_1), axis=0)"

class Mini(cgp.OperatorNode):
    _arity = 2
    _def_output = "np.minimum(x_0, x_1)"
#     _def_output_numpy = "np.min((x_0, x_1), axis=0)"

class Avg(cgp.OperatorNode):
    _arity = 2
    _def_output = "(x_0 + x_1) / 2"
    _def_output_numpy = "(x_0 + x_1) / 2"

class CGP_interface():
    def __init__(self, correct, noisy, error_function, seed : int, strategy, iterations):
        self._correct = (correct / 255.0) - 0.5
        self._noisy = (noisy / 255.0) - 0.5
        self._changes = (self._correct != self._noisy)
        self._error_function = error_function
        self.functor = Functor(None, strategy, 2)
        self.population_params = {"n_parents": 2, "seed": seed}
        self.height, self.width = self._correct.shape

        global size_prod 
        size_prod = self.height * self.width
        if strategy == four_outputs:
            self.n_outputs = 4
        elif strategy == three_outputs:
            self.n_outputs = 3
        elif strategy == two_outputs:
            self.n_outputs = 2
        else:
            self.n_outputs = 1
        self.genome_params = {
            "n_inputs": 25,
            "n_outputs": self.n_outputs,
            "n_columns": 7,
            "n_rows": 9,
            "levels_back": 7,
            "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat, cgp.IfElse, DivProtected, 
                           ConstantOne, ConstantZero, Maxi, Mini, Identity, AbsSub, Avg),
            }
        self.ea_params = {"n_offsprings": 5, "mutation_rate": 0.2, "n_processes": 4}
        self.evolve_params = {"max_generations": iterations}
    
    def objective_deterministic(self, individual : cgp.IndividualSingleGenome):
        self.functor.set_function(individual.to_func())
        filtered_image = nd.generic_filter(self._noisy, self.functor.work, (5,5))
        individual.fitness = (self._error_function(self._correct, filtered_image))
        return individual
    
    def objective_two_outputs(self, individual : cgp.IndividualSingleGenome):
        global size_prod
        self.functor.set_function(individual.to_func())
        filtered_image = nd.generic_filter(self._noisy, self.functor.work, (5,5))
        filtered_pixels = (filtered_image == self._noisy)
        correct_changes = np.sum(filtered_pixels != self._changes)
        individual.fitness = (self._error_function(self._correct, filtered_image)
                              - 0.1 * (correct_changes / (size_prod)))
        return individual

class Functor():
    def __init__(self, function = None, strategy = two_outputs, output_size = 2):
        self._function = function
        
        self._strategy = strategy
        self.output_size = output_size
    
    def set_function(self, function):
        self._function = function
        
    def work(self, inputor):
        return self._strategy(self._function, inputor)

def MSE(img1, img2):
    global size_prod
    # diff = np.sum(cv2.subtract(img2, img1) ** 2)
    diff = np.sum((img1 - img2) ** 2)
    scaled = ((size_prod) - diff) / (size_prod)
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
    global seed
    if iteration == 0:
        history["champion_fitness"].append(seed)
    history["champion_fitness"].append(pop.champion.fitness)
    iteration += 1

def parser_init():
    parser = argparse.ArgumentParser(description='Program for cgp algorithm')
    parser.add_argument("--clean_path", type=str, help="Path to clear image (no noises).", default="./data/tshushima_small_60.jpg")
    parser.add_argument("--noisy_path", type=str, help="Path to noisy image.", default="./data/tshushima_small_15percent.jpg")
    parser.add_argument("--runs", type=int, help="Number of program runs.", default=15)
    parser.add_argument("--result_path", type=str, help="Path to folder for results.", default="./experimenty")
    parser.add_argument("--strategy", help="Strategy for two outputs.", choices=["two_outputs", "deterministic", "no_threshold", "three_outputs", "four_outputs", "two_mutations"])
    parser.add_argument("--iterations", help="Iterations (generations) for each run.", default=400, type=int)
    return parser

def select_strategy(args):
    if args.strategy is not None:
        if args.strategy == "two_outputs":
            strategy = two_outputs
            strategy_name = "two_outputs2"
        elif args.strategy == "deterministic":
            strategy = deterministic
            strategy_name = "deterministic"
        elif args.strategy == "no_threshold":
            strategy = no_threshold
            strategy_name = "no_threshold"
        elif args.strategy == "three_outputs":
            strategy = three_outputs
            strategy_name = "three_outputs"
        elif args.strategy == "four_outputs":
            strategy = four_outputs
            strategy_name = "four_outputs"
        elif args.strategy == "two_mutations":
            strategy = two_outputs
            strategy_name = "two_mutations"
    else:
        print("No strategy was selected! Two outputs strategy will be performed!")
        strategy = two_outputs
        strategy_name = "two_outputs"
    return strategy, strategy_name

if __name__ == "__main__":
    parser = parser_init()
    args = parser.parse_args()
    strategy, strategy_name = select_strategy(args)

    # print(args.clean_path)
    img1, img2 = load_image(args.clean_path, args.noisy_path)
    if img1 is None:
        print("Ilegální obrázek")
        exit(1)
    
    print_image(img2, f"{args.result_path}/{strategy_name}/in.jpg")
    best_of_the_best = None
    global iteration
    global seed
    for i in range(args.runs):
        seed = int.from_bytes(os.urandom(4), 'big')
        interface = CGP_interface(img1, img2, MSE, seed, strategy, args.iterations)
        iteration = 0
        history["champion_fitness"] = []
        # print(interface.objective(np.array([[0.01,0.024,0.01],[0.024,0.9,0.024],[0.01,0.024,0.01]])))
        pop = cgp.Population(**interface.population_params, genome_params=interface.genome_params)
        ea = cgp.ea.MuPlusLambda(**interface.ea_params)
        # try:
        if strategy_name == "deterministic":
            cgp.evolve(pop, interface.objective_deterministic, ea, **interface.evolve_params, print_progress=True, callback=recording_callback)
        else:
            cgp.evolve(pop, interface.objective_two_outputs, ea, **interface.evolve_params, print_progress=True, callback=recording_callback)
        # except:
            # with open('ended_randomly.pkl', 'wb') as handle:
            #     pkl.dump(pop.champion, handle, protocol=pkl.HIGHEST_PROTOCOL)
            # exit()
        fun = Functor(pop.champion.to_func(), strategy)
        if best_of_the_best is None or pop.champion.fitness > best_of_the_best.fitness:
            best_of_the_best = pop.champion
        img_new = nd.generic_filter((img2 / 255.0) - 0.5, fun.work, (5,5))
            # print(MSE(img1, img_new))
            
        filtered_pixels = (img_new == interface._noisy)
        correct_changes = np.sum(filtered_pixels != interface._changes)
        height, width = filtered_pixels.shape
        if strategy_name == "deterministic":
            history[1:] -= 0.1 * correct_changes / (height * width)
        print_image(img_new, f"{args.result_path}/{strategy_name}/res{i}.jpg")
        with open(f"{args.result_path}/{strategy_name}/histories.txt", "a") as f:
            f.write(' '.join(str(aux) for aux in history["champion_fitness"]))
            f.write('\n')
        with open(f"{args.result_path}/{strategy_name}/best_sol.pkl", 'wb') as handle:
            pkl.dump(best_of_the_best, handle, protocol=pkl.HIGHEST_PROTOCOL)
