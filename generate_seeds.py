# Author = David Hudak
# Login = xhudak03
# Subject = EVO
# Year = 2022/2023
# Short Description = file for generating random seeds

import random as random

def generate_seeds(output : str, how_much : int):
    seeds = []
    with open(output, "w") as f:
        for i in range(how_much):
            f.write(f"{random.randint(0, 1000000)} ")

if __name__ == "__main__":
    generate_seeds(output="./seeds.txt", how_much = 200)
    
    
        