from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from pprint import pprint
import array
import numpy as np
import random
import representations, fitness, mutations

INITIAL_BLOCKS = 5
POPULATION = 10
GENERATIONS = 20
PROB_MUTATIONS = 0.6
PROB_MATE = 0.4
NUMBER_EPOCHS = 30
DATASET = 'fashion'  # or 'cifar10'


def getRandomLayer():
    networks = [
        representations.make_conv2d_repr(),
        representations.make_conv2d_pool_repr(),
        representations.make_conv2d_dropout_repr(),
        representations.make_noise_repr(),
    ]
    probabilities = np.array([.4, .5, .5, .05])
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(networks, p=probabilities)


def evaluateFunc(individual):
    return fitness.evaluate_nn(individual, NUMBER_EPOCHS, verbose=True, dataset=DATASET),


def initRepeatRandom(container, func, n):
    '''
    Extended toolbox.initRepeat() function to work with random initialization
    instead of fixed numbers.  Set this length to be minimal of 2, as we do two
    point crossover!!
    '''
    return container(func() for _ in range(np.random.randint(2, n)))


# Create attributes
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', np.ndarray,  fitness=creator.FitnessMax)

mutations.setIndividual(creator.Individual)
mutations.setInitialization(getRandomLayer)

toolbox = base.Toolbox()

toolbox.register('individual', initRepeatRandom, creator.Individual,
                 getRandomLayer, n=INITIAL_BLOCKS)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluateFunc) #register the evaluation function
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', mutations.mutate_network, mutations=1)
toolbox.register('select', tools.selTournament, tournsize=3)


def main():
    random.seed(1337)

    pop = toolbox.population(n=POPULATION)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(10, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=PROB_MATE,
                                   mutpb=PROB_MUTATIONS, ngen=GENERATIONS,
                                   stats=stats, halloffame=hof, verbose=True)

    print('\n----------------------------------')
    print(log)
    print('\n----------------------------------\n')

    print('Hall of Fame:\n')
    for h in hof:
        pprint(h)
        print('Fitness: {}'.format(h.fitness))


if __name__ == '__main__':
    main()
