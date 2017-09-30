import EvolutionAlgorithm  as EA
import numpy as np



def main():
    Optimizer = EA.Firefly(3000, fitness, Iteration = 5000, PopSize = 40)
    #print(Optimizer.PopSize)
    # print(Optimizer.Pop)
    #print(fitness(np.array([0.5,0.5,0.5])))
    Optimizer.run()
    #print(Optimizer.Pop)


def fitness(solution):
    Upperbond = 5.12
    Lowerbond = -5.12
    Interval = Upperbond - Lowerbond
    solution = Lowerbond + (solution * Interval)
    fitness_op = EA.sample_fitness()
    return fitness_op.Rastrigin( solution )
pass


if __name__ == '__main__':
    main()
pass 