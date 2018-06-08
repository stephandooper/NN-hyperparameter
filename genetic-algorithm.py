import random
import array
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import representations



# Create attributes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)


BLOCK_TYPES = ['dense', 'conv']
NODE_SIZES = [16, 32, 64, 28, 12, 10, 86, 100]


'''
Returns random block elements.
'''
def getRandomBlock(blockType=None, nodeLength=None):
    if blockType is None:
        blockType = random.choice(BLOCK_TYPES)
    if nodeLength is None:
        nodeLength = random.choice(NODE_SIZES)

    return (blockType, nodeLength)


def getRandomIndividual(iterations=3):
    #Possible networks to choose from:
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

# Container for populations:

toolbox = base.Toolbox()
# Attribute generator (happens randomly at each creation)
#toolbox.register("nnelem", getRandomReprFunc)
# Structure initializers

creator.create("NNCreator", numpy.ndarray, fitness=creator.FitnessMax)

toolbox.register("individual", getRandomIndividual) #<-- Creates 3 elements
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.nnelem, 3) #<-- Creates 3 elements
#toolbox.register("individual", tools.initRepeat, list, toolbox.nnelem, 3) #<-- Creates 3 elements
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ---------------- Up till here it should work just fine


'''
Evaluation function (should return the fitness)
'''
def evaluateFunc(individual):
    return sum(individual), #<--- IMPORTANT: add the comma ','; as it needs to return a tuple



toolbox.register("evaluate", evaluateFunc) #register the evaluation function
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("select", tools.selBest)
#deap.tools.selBest(individuals, k, fit_attr='fitness')¶



def main():
    random.seed(1337)
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                                   stats=stats, halloffame=hof, verbose=True)
    #print('log: ', log)

if __name__ == "__main__":
    main()