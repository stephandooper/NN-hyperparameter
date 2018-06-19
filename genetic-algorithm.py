import random
import numpy
import array

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import representations, fitness


INITIAL_BLOCKS = 3 # Represents how many random layer blocks each NNet should start with
POPULATION = 3
GENERATIONS = 2
PROB_MUTATIONS = 0.0 # Probability of mutating in a new generation
PROB_MATE = 0.5 # Probability of mating / crossover in a new generation
NUMBER_EPOCHS = 2 #Epochs when training the network

#---------------------

def getRandomIndividual(iterations=1):
    #Possible networks to choose from:
    '''
    networks = [
        [representations.make_conv2d_repr(),
        representations.make_pool_repr()],

        [representations.make_dropout_repr(),
        representations.make_conv2d_repr()],

        [representations.make_batchnorm_repr()],

        [representations.make_noise_repr()]
    ]

    probabilities = [0.3, 0.3, 0.25, 0.15]

    
    out = []

    
    for x in range(0,iterations):
        choice = numpy.random.choice(networks, p=probabilities)
        for layer in choice:
            out.append(layer)
    
    return out
    '''
    networks = [
        representations.make_conv2d_pool_repr(),
        representations.make_conv2d_dropout_repr(),
        representations.make_batchnorm_repr(),
        representations.make_noise_repr()
    ]
    probabilities = [0.3, 0.3, 0.25, 0.15]
    
    return numpy.random.choice(networks,p=probabilities)


'''
Evaluation function (should return the fitness)
'''
def evaluateFunc(individual):
    return fitness.evaluate_nn(individual, NUMBER_EPOCHS), #<--- IMPORTANT: add the comma ','; as it needs to return a tuple


# -------------- Init / Main stuff ----------------------

# Create attributes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray,  fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("individual", tools.initRepeat, creator.Individual, getRandomIndividual, n=INITIAL_BLOCKS) #<-- Creates 3 elements
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", evaluateFunc) #register the evaluation function
toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mutate", mutations.mutate_append_remove, prob_remove=1)
toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("select", tools.selBest)
#deap.tools.selBest(individuals, k, fit_attr='fitness')¶



def main():
    random.seed(1337)
    pop = toolbox.population(n=POPULATION)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


    hof = tools.HallOfFame(1, similar=numpy.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=PROB_MATE, mutpb=PROB_MUTATIONS, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    print("\n----------------------------------")
    print(log)
    print("----------------------------------\n")
    print("Best network:")
    print(hof[0])
    print("With a fitness of: ", hof[0].fitness)

if __name__ == "__main__":
    main()