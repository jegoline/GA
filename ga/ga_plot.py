from ga import City, World, GA

import ga
import matplotlib.pyplot as plt
import random
import sys
import os
import jsonpickle
from multiprocessing.dummy import Pool as ThreadPool


def plot_route(ax, route, color='r'):
    x = []
    y = []
    for city in route.route:
        x.append(city.get_x())
        y.append(city.get_y())

    ax.plot(x, y, 'bo', x, y, color)
    for i in range(1, len(x)):
        ax.annotate(str(i), (x[i], y[i]))


# generate world (all cities lay between 2 concentric squares)
def generate_world():
    w = World()
    while w.number_of_cities() < 35:
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


def save_xp(filename, data):
    f = open(filename, 'a')
    for d in data:
        for t in d:
            f.write(t)
    f.close


class XP(object):
    def __init__(self, migration_interval, num_migrants, world):
        self.migration_interval = migration_interval
        self.num_migrants = num_migrants
        self.world = world

    def format_data(self, xp_num, island_num, generation_num, fitness, distance, num_migrants, migr_int):
        return '{},{},{},{},{},{},{}\n'.format(xp_num, island_num, generation_num, fitness, distance, num_migrants, migr_int)

    def run_xp(self, xp_num):
        results = []
        # init islands
        islands = []
        for i in range(1, 5):
            island = GA(self.world, 0.015, 5, True, 400, 'Island:' + str(i))
            islands.append(island)
            # prepare visualization
            # fig = plt.figure()
            # fig.subplots_adjust(left=0.2, wspace=0.6)
            # ax1 = fig.add_subplot(221)
            # ax2 = fig.add_subplot(222)
            # ax3 = fig.add_subplot(223)
            # ax4 = fig.add_subplot(224)

        # run evolution
        gen_num = 0
        for interval_num in range(0, 400 / self.migration_interval):
            # evaluate N populations on every island
            island_num = 0
            for island in islands:
                fittest = island.run(self.migration_interval)
                island_num += 1
                gen_num += self.migration_interval
                results.append(self.format_data(xp_num, island_num, gen_num, fittest.get_fitness(), fittest.get_distance(),
                            self.num_migrants, self.migration_interval))
            migrate(islands, self.num_migrants)
        return results


def main(argv):
    # use predefined world if necessary
    file_exist = os.path.isfile('world.json')
    if file_exist and os.path.getsize('world.json') != 0:
        world = load_world('world.json')
    else:
        world = generate_world()
        save_world(world, 'world.json')

    max_xp_num = 4
    for migration_interval in range(1, 100, 5):
        print 'migration interval ' + str(migration_interval)
        for num_migrants in range(1, 20, 4):
            print 'Run XPs for num migrants ' + str(num_migrants)
            xp = XP(migration_interval, num_migrants, world)

            pool = ThreadPool(4)
            results = pool.map(xp.run_xp, range(0, max_xp_num))
            # close the pool and wait for the work to finish
            pool.close()
            pool.join()
            print 'XPs are done'
            save_xp('experiment_data.csv', results)

    # plot fittest for each island
    #plot_route(ax1, islands[0].get_best())
    #plot_route(ax2, islands[1].get_best())
    #plot_route(ax3, islands[2].get_best())
    #plot_route(ax4, islands[3].get_best())

    #fittest = find_fittest_island(islands)
    #print str(fittest.get_distance())
    #plt.show()


if __name__ == "__main__":
    main(sys.argv)