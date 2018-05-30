import random
import array
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# Create attributes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)



elements = [] # TODO

'''
TODO:

Create list of elements where it randomly fills the possible blocks with integers:
ex:
dense, 32
dense, 16
dense, 64
conv, 16
conv, 64

so basically random: (type, integer)

'''


# Container for populations:

toolbox = base.Toolbox()
# Attribute generator (happens randomly at each creation)
toolbox.register("nn-element", random.choice, elements.keys())
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.nn-element, 3) #<-- Creates 3 elements
toolbox.register("population", tools.initRepeat, list, toolbox.individual)







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
    pop = toolbox.population(n=300)
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