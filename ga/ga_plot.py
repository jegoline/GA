from ga import City, World, GA

import csv
import matplotlib.pyplot as plt
import random
import copy
import os
import jsonpickle
from multiprocessing.dummy import Pool as ThreadPool


def plot_route(ax, route, color='r'):
    x = []
    y = []
    ax.clear()
    for city in route.route:
        x.append(city.get_x())
        y.append(city.get_y())

    ax.plot(x, y, 'bo', x, y, color)
    ax.set_title(route.get_distance())
    for i in range(1, len(x)):
        ax.annotate(str(i), (x[i], y[i]))


# generate world (all cities lay between 2 concentric squares)
def generate_world():
    w = World()
    while w.number_of_cities() < 50:
        x = random.random()
        y = random.random()

        if (0.1 >= x or x >= 0.9) or (0.1 >= y or y >= 0.9) or (0.55 >= y >= 0.45):
            x = int(x * 1000)
            y = int(y * 1000)
            city = City(x, y, str(w.number_of_cities()))

            w.add_city(city)
    return w


def save_world(world, filename):
    f = open(filename, 'w')
    f.write(jsonpickle.encode(world))


def load_world(filename):
    f = open(filename, 'r+')
    world = jsonpickle.decode(f.read())
    return world


# perform migration (ring topology)
def migrate (islands, num_migrants):
    migrants = islands[0].get_bests(num_migrants)
    for i in range(1, len(islands)):
        fittest_n = islands[i].get_bests(num_migrants)
        islands[i - 1].add_migrants(fittest_n)
    #    print 'Migrate from ' + islands[i].label + ' to ' + islands[i - 1].label + ' migrant ' \
    #          + str(fittest.get_distance())
    islands[len(islands) - 1].add_migrants(migrants)

    #print 'Migrate from ' + islands[0].label + ' to ' + islands[len(islands) - 1].label + ' migrant ' \
    #      + str(migrant.get_distance())


def find_fittest_island(islands):
    # find fittest among all islands
    fittest = None
    for island in islands:
        island_fittest = island.get_best()
        if fittest is None:
            fittest = island_fittest
        elif fittest.get_fitness() < island_fittest.get_fitness():
            fittest = island_fittest
    return fittest


def save_xps(filename, data):
    f = open(filename, 'a')
    for d in data:
        for t in d:
            f.write(t)
    f.close


class XP(object):
    def __init__(self, migration_interval, num_migrants, world, fitness_thresholds):
        self.migration_interval = migration_interval
        self.num_migrants = num_migrants
        self.world = world
        self.fitness_thresholds = fitness_thresholds

    def format_data(self, xp_num, generation_num, fitness, distance, num_migrants, migr_int, threshold):
        return '{},{},{},{},{},{},{}\n'.format(xp_num, generation_num, fitness, distance, num_migrants, migr_int, threshold)

    def run_xp(self, xp_num):
        results = []
        # init islands
        islands = []
        for i in range(1, 5):
            island = GA(self.world, 0.015, 5, True, 350, 'Island:' + str(i))
            islands.append(island)

        # run evolution
        thresholds = copy.copy(self.fitness_thresholds)
        for gen in range(1, 600):
            if gen % self.migration_interval == 0:
                migrate(islands, self.num_migrants)

            for island in islands:
                island.run(1)

            fittest = find_fittest_island(islands)
            threshold = thresholds[0]

            if fittest.get_distance() < threshold:
                results.append(self.format_data(xp_num, gen, fittest.get_fitness(), fittest.get_distance(),
                                        self.num_migrants, self.migration_interval, threshold))
                thresholds.pop(0)
                if len(thresholds) == 0:
                    return results

        return results


def explore_migration_interval():
    # use predefined world if necessary
    file_exist = os.path.isfile('world.json')
    if file_exist and os.path.getsize('world.json') != 0:
        world = load_world('world.json')
    else:
        world = generate_world()
        save_world(world, 'world.json')

    max_xp_num = 1
    num_migrants = 5
    fitness_thresholds = [11000, 6500]

    for migration_interval in range(1, 250, 2):
        print 'Run ' + str(max_xp_num) + ' XPs for migration interval ' + str(migration_interval)
        xp = XP(migration_interval, num_migrants, world, fitness_thresholds)

        pool = ThreadPool(8)
        results = pool.map(xp.run_xp, range(0, max_xp_num))
        pool.close()
        pool.join()

        save_xps('experiment_data1.csv', results)
        print 'Finish XPs for migration interval ' + str(migration_interval)


def animate_island_model():
    file_exist = os.path.isfile('world.json')
    if file_exist and os.path.getsize('world.json') != 0:
        world = load_world('world.json')
    else:
        world = generate_world()
        save_world(world, 'world.json')

    migration_interval = 10
    num_migrants = 1
    print 'Migration pattern: ' + str(migration_interval) + str(num_migrants)

    plt.ion()
    # init islands
    islands = []
    for i in range(1, 5):
        island = GA(world, 0.015, 5, True, 400, 'Island:' + str(i))
        islands.append(island)

    # prepare visualization
    fig = plt.figure()
    fig.subplots_adjust(left=0.2, wspace=0.6)
    ax1 = fig.add_subplot(221)
    plot_route(ax1, islands[0].get_best())
    ax2 = fig.add_subplot(222)
    plot_route(ax2, islands[1].get_best())
    ax3 = fig.add_subplot(223)
    plot_route(ax3, islands[2].get_best())
    ax4 = fig.add_subplot(224)
    plot_route(ax4, islands[3].get_best())
    plt.pause(0.24)

    # run evolution
    for interval_num in range(0, 800 / migration_interval):
        # evaluate N populations on every island
        for island in islands:
            island.run(migration_interval)
        # migrate(islands, num_migrants)
        # plot fittest for each island

        plot_route(ax1, islands[0].get_best())
        plot_route(ax2, islands[1].get_best())
        plot_route(ax3, islands[2].get_best())
        plot_route(ax4, islands[3].get_best())

        fittest = find_fittest_island(islands)
        print str(fittest.get_distance())

        plt.pause(1)

def load_experimental_data():
    with open('experiment_data.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            print ', '.join(row)


if __name__ == "__main__":
    #animate_island_model()
    explore_migration_interval()
    #load_and_plot_experimental_data()