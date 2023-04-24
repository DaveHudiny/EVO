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


########################################
# Definitions of multiple CGP blocks.

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


class Mini(cgp.OperatorNode):
    _arity = 2
    _def_output = "np.minimum(x_0, x_1)"


class Avg(cgp.OperatorNode):
    _arity = 2
    _def_output = "(x_0 + x_1) / 2"
    _def_output_numpy = "(x_0 + x_1) / 2"

############################################
# CGP interface class.


class CGP_interface():
    """Class for definition of objective functions, genome parameters etc.
    """

    def __init__(self, correct, noisy, error_function, seed: int, strategy, iterations, mutation_rate):
        """CGP_interface init function.

        Args:
            correct (np.ndarray): Correct image.
            noisy (np.ndarray): Noised image.
            error_function (function): Error function (for example MSE)
            seed (int): Seed for experiment
            strategy (function): One of the functions from file strategies.py, which determines how the outputs are printed.
            iterations (int): Number of generations of evolutionary algorithm. 
            mutation_rate (float): Mutation rate of CGP algorithm.
        """
        self._correct = (correct / 255.0) - 0.5
        self._noisy = (noisy / 255.0) - 0.5
        self._changes = (self._correct != self._noisy)
        self._error_function = error_function
        self.functor = Functor(None, strategy, 2)
        self.population_params = {"n_parents": 2, "seed": seed}
        self.height, self.width = self._correct.shape
        self.mutation_rate = mutation_rate

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
        self.ea_params = {"n_offsprings": 5,
                          "mutation_rate": mutation_rate, "n_processes": 4}
        self.evolve_params = {"max_generations": iterations}

    def objective_deterministic(self, individual: cgp.IndividualSingleGenome):
        """Objective for deterministic filter. The incorrect filtering penalty is count after the function.

        Args:
            individual (cgp.IndividualSingleGenome): Single individual of CGP algorithm.

        Returns:
            cgp.IndividualSingleGenome: Returns individual with new fitness value.
        """
        self.functor.set_function(individual.to_func())
        filtered_image = nd.generic_filter(
            self._noisy, self.functor.work, (5, 5))
        individual.fitness = (self._error_function(
            self._correct, filtered_image))
        return individual

    def objective_two_outputs(self, individual: cgp.IndividualSingleGenome):
        """Objective for multiple outputs filters. The incorrect filtering penalty is count inside the function.

        Args:
            individual (cgp.IndividualSingleGenome): Single individual of CGP algorithm.

        Returns:
            cgp.IndividualSingleGenome: Returns individual with new fitness value.
        """
        global size_prod
        self.functor.set_function(individual.to_func())
        filtered_image = nd.generic_filter(
            self._noisy, self.functor.work, (5, 5))
        # Does filtering.
        filtered_pixels = (filtered_image == self._noisy)
        correct_changes = np.sum(filtered_pixels != self._changes)
        individual.fitness = (self._error_function(self._correct, filtered_image)
                              - 0.1 * (correct_changes / (size_prod)))  # Penalty
        return individual


class Functor():
    """Auxilary class for handling functions for filters.
    """

    def __init__(self, func=None, strategy=two_outputs, output_size=2):
        """Init function for Functor class.

        Args:
            func (function, optional): Error function. Defaults to None.
            strategy (function, optional): One of the functions from strategies.py. Defaults to two_outputs.
            output_size (int, optional): Size of output of CGP algorithm. Unused. Defaults to 2.
        """
        self._func = func
        self._strategy = strategy
        self.output_size = output_size

    def set_function(self, func):
        """Changes function of Functor instance. Used in CGP algorithm for each individual.

        Args:
            func (function): New function (individual.to_func()).
        """
        self._func = func

    def work(self, inputor):
        """Does work of Functor. Used in filter.

        Args:
            inputor (list): List of inputs (of len 25).

        Returns:
            float: New pixel value.
        """
        return self._strategy(self._func, inputor)


def MSE(img1, img2):
    """Mean square error function.

    Args:
        img1 (np.ndarray): Array of pixels of the first image.
        img2 (np.ndarray): Array of pixels of the second image.

    Returns:
        float: Value of mean square error. Scaled to be inside (-inf, 1>. 
    """
    global size_prod
    # diff = np.sum(cv2.subtract(img2, img1) ** 2)
    diff = np.sum((img1 - img2) ** 2)
    scaled = ((size_prod) - diff) / (size_prod)
    return scaled


def load_image(path_correct: str, path_noisy: str):
    """Loads images from paths.

    Args:
        path_correct (str): Path to correct image.
        path_noisy (str): Path to noised image.

    Returns:
        np.ndarray, np.ndarray: Two arrays of loaded images (or None). 
    """
    if os.path.exists(path_correct) and os.path.exists(path_noisy):
        # Read image
        img1 = cv2.imread(path_correct)
        img2 = cv2.imread(path_noisy)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return img1, img2
    else:
        return None, None


def print_image(img, path):
    """Prints image to path.

    Args:
        img (np.ndarray): Image to be printed.
        path (str): Path to resulting image.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(path)


history = {}
history["champion_fitness"] = []


def recording_callback(pop):
    """Callback function for printing to history file.

    Args:
        pop: Population of CGP algorithm.
    """
    global iteration
    global seed
    if iteration == 0:  # Within first iteration, add seed to it.
        history["champion_fitness"].append(seed)
    history["champion_fitness"].append(pop.champion.fitness)
    iteration += 1


def parser_init():
    """Creates parser.

    Returns:
        argparse.ArgumentParser: New created parser.
    """
    parser = argparse.ArgumentParser(description="Script for cgp algorithm experiments.")
    parser.add_argument("--clean_path", type=str, help="Path to clear image (no noises).",
                        default="../data/tshushima_small_60.jpg")
    parser.add_argument("--noisy_path", type=str, help="Path to noisy image.",
                        default="../data/tshushima_small_15percent.jpg")
    parser.add_argument("--runs", type=int,
                        help="Number of program runs.", default=15)
    parser.add_argument("--result_path", type=str,
                        help="Path to folder for results.", default="../experimenty")
    parser.add_argument("--strategy", help="Strategy for two outputs.", choices=[
                        "experimental", "two_outputs", "deterministic", "no_threshold", "three_outputs", "four_outputs", "two_mutations"])
    parser.add_argument(
        "--iterations", help="Iterations (generations) for each run.", default=400, type=int)
    parser.add_argument("--mutation_rate",
                        help="Mutation rate of CGP", default=0.2, type=float)
    return parser


def select_strategy(args):
    """Selects strategy by args value.

    Args:
        args: Parsed arguments by argparse.
    Returns:
        function, str: Returns selected strategy and its name.
    """
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
        elif args.strategy == "experimental":
            strategy = two_outputs
            strategy_name = "experimental"
    else:
        print("No strategy was selected! Two outputs strategy will be performed!")
        strategy = two_outputs
        strategy_name = "two_outputs"
    return strategy, strategy_name


if __name__ == "__main__":
    parser = parser_init()
    args = parser.parse_args()
    strategy, strategy_name = select_strategy(args)

    img1, img2 = load_image(args.clean_path, args.noisy_path)
    if img1 is None:
        print("Ilegální obrázek")
        exit(1)

    print_image(img2, f"{args.result_path}/{strategy_name}/in.jpg")
    best_of_the_best = None
    global iteration
    global seed
    for i in range(args.runs):  # Main cycle.
        seed = int.from_bytes(os.urandom(4), 'big')
        interface = CGP_interface(
            img1, img2, MSE, seed, strategy, args.iterations, args.mutation_rate)
        iteration = 0
        history["champion_fitness"] = []

        ##################################
        # CGP itself
        pop = cgp.Population(**interface.population_params,
                             genome_params=interface.genome_params)
        ea = cgp.ea.MuPlusLambda(**interface.ea_params)
        if strategy_name == "deterministic":
            cgp.evolve(pop, interface.objective_deterministic, ea, **
                       interface.evolve_params, print_progress=True, callback=recording_callback)
        else:
            cgp.evolve(pop, interface.objective_two_outputs, ea, **
                       interface.evolve_params, print_progress=True, callback=recording_callback)
        fun = Functor(pop.champion.to_func(), strategy)
        if best_of_the_best is None or pop.champion.fitness > best_of_the_best.fitness:
            best_of_the_best = pop.champion

        ##################################
        # Application of filter and other workarounds.

        img_new = nd.generic_filter((img2 / 255.0) - 0.5, fun.work, (5, 5))
        filtered_pixels = (img_new == interface._noisy)
        correct_changes = np.sum(filtered_pixels != interface._changes)
        height, width = filtered_pixels.shape
        # Correction of deterministic strategy fitness values.
        if strategy_name == "deterministic":
            history[1:] -= 0.1 * correct_changes / (height * width)
        print_image(img_new, f"{args.result_path}/{strategy_name}/res{i}.jpg")
        with open(f"{args.result_path}/{strategy_name}/histories.txt", "a") as f:
            f.write(' '.join(str(aux) for aux in history["champion_fitness"]))
            f.write('\n')
        with open(f"{args.result_path}/{strategy_name}/best_sol.pkl", 'wb') as handle:
            pkl.dump(best_of_the_best, handle, protocol=pkl.HIGHEST_PROTOCOL)
