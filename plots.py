# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Year = 2022/2023
# Short Description = main file of project to subject Applied Evolutionary Algorithms

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":
    seeds = {}
    histories = {}
    results = {}
    for name in ["deterministic", "two_outputs2", "two_mutations", "three_outputs_03", "four_outputs", "no_threshold"]:
        histories[name], seeds[name], results[name] = load_history(f"./experimenty/{name}/histories.txt")
    
    RUNS, EVALS, MAX_FIT = 15, 400, 1

    deter = histories["two_mutations"]
    mins = np.min(deter, axis=0)
    meds = np.median(deter, axis=0)
    maxs = np.max(deter, axis=0)

    x = np.arange(1,EVALS+1)
    # plt.xscale('log')
    plt.plot(x,meds)
    plt.fill_between(x,mins,maxs,alpha=0.4)
    plt.axhline(MAX_FIT, color="black",\
    linestyle="dashed")
    # plt.ylim([0.8, 1])
    plt.xlabel('Pocet evaluaci')
    plt.ylabel('Fitness')
    plt.title('Konvergencni krivka')
    plt.show()
        
    dfr = pd.DataFrame.from_dict(results)
    dfr = dfr.rename(columns={"deterministic": "Deterministic", "two_outputs2" : "2 Outputs 0.3"})
    ax = sns.boxplot(data=dfr)
    plt.show()
    # print(dfh)
    # sns.lineplot(data=dfh, hue=None)
    # plt.show()