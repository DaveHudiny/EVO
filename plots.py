# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Year = 2022/2023
# Short Description = main file of project to subject Applied Evolutionary Algorithms

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def load_history(path):
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


def plot_boxplots():
    pass

def plot_konvergent_curve():
    pass

if __name__ == "__main__":
    seeds = {}
    histories = {}
    results = {}
    for name in ["deterministic", "two_outputs2", "two_mutations", "three_outputs", "four_outputs", "no_threshold"]:
        histories[name], seeds[name], results[name] = load_history(f"./experimenty/{name}/histories.txt")
    
    dfh = pd.DataFrame(histories)
    dfr = pd.DataFrame(results)
    average = dfr["two_mutations"].mean()

    print(average)
    print(dfr)
    sns.boxplot(data=dfr)
    plt.show()