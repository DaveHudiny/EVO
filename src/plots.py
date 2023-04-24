# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Year = 2022/2023
# Short Description = plots.py file of project to subject Applied Evolutionary Algorithms. There I plot figures.

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def load_history(path: str):
    """Reads experiments folder history file.

    Args:
        path (str): Path where to history file (eg. ../experimenty/two_outputs/histories.txt)

    Returns:
        list, list, list: Lists with complete histories, seeds (useless) and final results.
    """
    seeds = []
    histories = []
    results = []
    with open(path, "r") as file:
        for line in file.readlines():
            splitor = line.split(" ")
            seeds.append(int(splitor[0]))
            histories.append(list(map(float, splitor[1:])))
            results.append(float(splitor[-1]))
    return histories, seeds, results


def plt_convergent(mins, meds, maxs, color: str = "red"):
    """Plots convergent curve with different colors. You have to plot and save it outside
       (because of plotting multiple curves to single figure). 

    Args:
        mins (np.array): Array of min values of multiple runs.
        meds (np.array): Array of median values of multiple runs.
        maxs (np.array): Array of maximum values of multiple runs.
        color (str, optional): Color of output convergent curve. Defaults to "red".
    """
    RUNS, EVALS, MAX_FIT = 15, 400, 1
    mins[mins < 0.96] = 0.96
    x = np.arange(1, EVALS+1)
    # plt.xscale('log')
    plt.plot(x, meds, color=color)
    plt.fill_between(x, mins, maxs, alpha=0.3, color=color)
    plt.axhline(MAX_FIT, color="black",
                linestyle="dashed")
    plt.xlabel("Pocet evaluací")
    plt.ylabel("Fitness")
    plt.title("Konvergenční křivka")


def plt_boxplots(results):
    """Plots boxplots of result fitnesses.

    Args:
        results (dict): Dictionary of keys by strategy with lists with resulting fitnesses of runs.
    """
    MAX_FIT = 1
    dfr = pd.DataFrame.from_dict(results)
    dfr = dfr.rename(columns={"deterministic": "Deter", "two_outputs": "2 Outs H", "three_outputs_03": "3 Outs H",
                              "four_outputs_03": "4 Outs H", "no_threshold": "No Thresh", "two_mutations": "2 Outs L"})
    ax = sns.boxplot(data=dfr)

    plt.xlabel("Zvolená strategie")
    plt.ylabel("Fitness")
    plt.savefig("../boxplots.pdf")
    plt.show()


def statistic_compare_results(results1, results2):
    """Pair t-test for two result array.

    Args:
        results1 (np.ndarray): First results.
        results2 (np.ndarray): Second results.
    """
    t, p = st.ttest_ind(results1, results2)
    print(f"P-hodnota pro dva zvolené výběry = {p}")


if __name__ == "__main__":
    seeds = {}
    histories = {}
    results = {}
    for name in ["deterministic", "two_mutations", "two_outputs",  "three_outputs_03", "four_outputs_03", "no_threshold"]:
        histories[name], seeds[name], results[name] = load_history(
            f"../experimenty/{name}/histories.txt")

    deter = histories["deterministic"]
    mins = np.min(deter, axis=0)
    meds = np.median(deter, axis=0)
    maxs = np.max(deter, axis=0)
    plt_convergent(mins, meds, maxs, "red")
    deter = histories["two_mutations"]
    mins = np.min(deter, axis=0)
    meds = np.median(deter, axis=0)
    maxs = np.max(deter, axis=0)
    plt_convergent(mins, meds, maxs, color="blue")
    plt.savefig("../convergent.pdf")
    plt.show()
    plt_boxplots(results)

    statistic_compare_results(
        results1=results["two_mutations"], results2=results["two_outputs"])
