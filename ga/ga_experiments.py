from ga_algorithm import GA, measure_entropy
from ga_utils import get_world, migrate, find_fittest_island
import csv
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool

def save_xps(filename, data):
    f = open(filename, 'a')
    for d in data:
        for t in d:
            f.write(t)
    f.close


def format_data(xp_num, generation_num, fitness, distance, num_migrants, migr_int):
    return '{},{},{},{},{},{}\n'.format(xp_num, generation_num, fitness, distance, num_migrants, migr_int)


def format_data_with_entropy(xp_num, generation_num, fitness, distance, num_migrants, migr_int, entropy):
    return '{},{},{},{},{},{},{}\n'.format(xp_num, generation_num, fitness, distance, num_migrants, migr_int, entropy)


class XP(object):
    def __init__(self, migration_interval, num_migrants, world):
        self.migration_interval = migration_interval
        self.num_migrants = num_migrants
        self.world = world

    def run_xp(self, xp_num):
        results = []
        # init islands
        islands = []
        for i in range(0, 5):
            island = GA(self.world, 0.015, 5, True, 80, 'Island:' + str(i))
            islands.append(island)

        # run evolution
        for gen in range(1, 30):
            if gen % self.migration_interval == 0:
                migrate(islands, self.num_migrants)

            general_population = []
            for island in islands:
                island.run(1)
                general_population.extend(island.get_population())

            entropy = measure_entropy(general_population)
            fittest = find_fittest_island(islands)
            results.append(format_data_with_entropy(xp_num, gen, fittest.get_fitness(), fittest.get_distance(),
                                        self.num_migrants, self.migration_interval, entropy))

        return results


def explore_island_vs_conventional():
    world = get_world()
    max_xp_num = 10
    num_migrants = 1
    migration_interval = 15

    print 'Run ' + str(max_xp_num) + ' XPs for migration interval ' + str(migration_interval)
    xp = XP(migration_interval, num_migrants, world)

    pool = ThreadPool(2)
    results = pool.map(xp.run_xp, range(0, max_xp_num))
    pool.close()
    pool.join()

    save_xps('experiment_island_vs_conv_15_1_with_entropy_2.csv', results)
    print 'Finish XPs for migration interval ' + str(migration_interval)

    print 'Run conventional model'
    results = []
    for xp_num in range(0, max_xp_num):
        conv = GA(world, 0.015, 5, True, 400, 'Conv')
        # run evolution
        for gen in range(1, 600):
            conv.run(1)
            fittest = conv.get_best()
            entropy = conv.measure_entropy()
            results.append(format_data_with_entropy(xp_num, gen, fittest.get_fitness(), fittest.get_distance(), 0, 0, entropy))
    save_xps('experimental_data/experiment_island_vs_conv_with_entropy_2.csv', results)
    print 'Finish conventional model'


def explore_migration_interval():
    # use predefined world if necessary
    world = get_world()
    max_xp_num = 10
    num_migrants = 2

    for migration_interval in range(12, 113, 100):
        print 'Run ' + str(max_xp_num) + ' XPs for migration interval ' + str(migration_interval)
        xp = XP(migration_interval, num_migrants, world)

        pool = ThreadPool(8)
        results = pool.map(xp.run_xp, range(0, max_xp_num))
        pool.close()
        pool.join()

        save_xps('experimental_data/test.csv', results)
        print 'Finish XPs for migration interval ' + str(migration_interval)


def explore_migration_size():
    world = get_world()
    max_xp_num = 10
    migration_interval = 1

    for num_migrants in range(1, 42, 10):
        print 'Run ' + str(max_xp_num) + ' XPs for migration size ' + str(num_migrants)
        xp = XP(migration_interval, num_migrants, world)

        pool = ThreadPool(2)
        results = pool.map(xp.run_xp, range(0, max_xp_num))
        pool.close()
        pool.join()

        save_xps('experimental_data/experiment_data_size_1_51_10_int_1_v2.csv', results)
        print 'Finish XPs for migration num_migrants ' + str(num_migrants)


def load_and_plot_different_migration_sizes():
    fig = plt.figure()
    fig.subplots_adjust(left=0.1, wspace=0.4)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    load_and_plot_experimental_data_migration_size('experiment_data_size_1_51_10_int_1_v2.csv', ax1)
    load_and_plot_experimental_data_migration_size('experiment_data_size_1_51_10_int_15.csv', ax2)
    load_and_plot_experimental_data_migration_size('experiment_data_size_1_51_10_int_30.csv', ax3)

    plt.ylabel('fitness')
    plt.xlabel('generation')

    plt.show()


def load(filename, ax, c, label):
    data = []
    with open(filename, 'rb') as csvfile:
        row_data = csv.reader(csvfile, delimiter=',', quotechar='|')
        curr_size = 1

        gen_to_fit = []
        plot_data = []

        for row in row_data:
            xp_num = int(row[0])
            gen_num = int(row[1])
            size = int(row[4])
            interval = int(row[5])
            fitness = float(row[2])
            entropy = float(row[6])

            print str(gen_num) + ' ' + str(len(gen_to_fit)) + ' ' + str(size)
            if len(gen_to_fit) < gen_num:
                gen_to_fit.append([])

            gen_to_fit[gen_num - 1].append(entropy)

    for gen in gen_to_fit:
        plot_data.append(np.mean(gen))

    data.append((curr_size, plot_data))

    line1, = ax.plot(range(0, len(data[0][1])), data[0][1], c, label=label)
    return line1


def load_and_plot_islands_vs_conventional():
    fig = plt.figure()
    fig.subplots_adjust(left=0.15, bottom=0.15,  wspace=0.4)

    line1 = load('experiment_island_vs_conv_with_entropy_2.csv', plt, 'k-', 'conventional model')
    line2 = load('experiment_island_vs_conv_15_1_with_entropy_2.csv', plt, 'm-', 'island model')

    plt.legend(handles=[line1, line2], loc=1)

    plt.ylabel('fitness')
    plt.xlabel('generation')

    plt.show()


def load_and_plot_experimental_data_migration_size(filename, ax):
    ax.set_xlabel('generation')
    ax.set_ylabel('fitness')
    ax.set_ylim([0.00004, 0.00018])
    data = []
    interval = 0
    with open(filename, 'rb') as csvfile:
        row_data = csv.reader(csvfile, delimiter=',', quotechar='|')
        curr_size = 1

        gen_to_fit = []
        plot_data = []

        for row in row_data:
            xp_num = int(row[0])
            gen_num = int(row[1])
            size = int(row[4])
            interval = int(row[5])
            fitness = float(row[2])

            if size != curr_size:
                for gen in gen_to_fit:
                    plot_data.append(np.mean(gen))
                data.append((curr_size, plot_data))
                curr_size = size

                gen_to_fit = []
                plot_data = []

            print str(gen_num) + ' ' + str(len(gen_to_fit)) + ' ' + str(size)
            if len(gen_to_fit) < gen_num:
                gen_to_fit.append([])

            gen_to_fit[gen_num - 1].append(fitness)

        for gen in gen_to_fit:
            plot_data.append(np.mean(gen))
        data.append((curr_size, plot_data))

        line1, = ax.plot(range(0, len(data[0][1])), data[0][1], 'k-', label='size=' + str(data[0][0]))
        line2, = ax.plot(range(0, len(data[1][1])), data[1][1], 'm--', label='size=' + str(data[1][0]))
        line3, = ax.plot(range(0, len(data[2][1])), data[2][1], 'g-.', label='size=' + str(data[2][0]), lw=1.5)
        line4, = ax.plot(range(0, len(data[3][1])), data[3][1], 'b--', label='size=' + str(data[3][0]))
        line5, = ax.plot(range(0, len(data[4][1])), data[4][1], 'r-', label='size=' + str(data[4][0]))

        ax.legend(handles=[line1, line2, line3, line4, line5], loc=4)
        ax.set_title('interval = ' + str(interval))


# plot several lines (one for every interval),
# each lines represents the average fitness as a function of generation number
def load_and_plot_experimental_data():
    data = []
    with open('experiment_data_1_30_2.csv', 'rb') as csvfile:
        row_data = csv.reader(csvfile, delimiter=',', quotechar='|')
        curr_int= 1

        gen_to_fit = []
        plot_data = []

        for row in row_data:
            xp_num = int(row[0])
            gen_num = int(row[1])
            interval = int(row[5])
            fitness = float(row[2])

            if interval != curr_int:
                for gen in gen_to_fit:
                    plot_data.append(np.mean(gen))
                data.append((curr_int, plot_data))
                curr_int = interval

                gen_to_fit = []
                plot_data = []

            print str(gen_num) + ' ' + str (len(gen_to_fit)) + ' ' + str(interval)
            if len(gen_to_fit) < gen_num:
                gen_to_fit.append([])

            gen_to_fit[gen_num-1].append(fitness)

        for gen in gen_to_fit:
            plot_data.append(np.mean(gen))
        data.append((curr_int, plot_data))

    line1, = plt.plot(range(0, len(data[0][1])), data[0][1], 'k-', label='interval=' + str(data[0][0]))
    line2, = plt.plot(range(0, len(data[1][1])), data[1][1], 'm--', label='interval=' + str(data[1][0]))
    line3, = plt.plot(range(0, len(data[2][1])), data[2][1], 'g-.', label='interval=' + str(data[2][0]), lw=1.5)
    line4, = plt.plot(range(0, len(data[3][1])), data[3][1], 'b--', label='interval=' + str(data[3][0]))
    line5, = plt.plot(range(0, len(data[12][1])), data[12][1], 'r-', label='interval=' + str(data[12][0]))

    plt.legend(handles=[line1, line2, line3, line4, line5], loc = 4)
    plt.ylabel('fitness')
    plt.xlabel('generation')

    plt.show()


def load_and_plot_experimental_data_3():
    data = []
    with open('experiment_data_compare_low_and_high.csv', 'rb') as csvfile:
        row_data = csv.reader(csvfile, delimiter=',', quotechar='|')
        curr_int = 12

        gen_to_fit = []
        plot_data = []

        for row in row_data:
            xp_num = int(row[0])
            gen_num = int(row[1])
            interval = int(row[5])
            fitness = float(row[2])

            if interval != curr_int:
                for gen in gen_to_fit:
                    plot_data.append(np.mean(gen))
                plot_data = plot_data[0:700]
                data.append((curr_int, plot_data))
                curr_int = interval

                gen_to_fit = []
                plot_data = []

            print str(gen_num) + ' ' + str (len(gen_to_fit)) + ' ' + str(interval)
            if len(gen_to_fit) < gen_num:
                gen_to_fit.append([])

            gen_to_fit[gen_num-1].append(fitness)

        for gen in gen_to_fit:
            plot_data.append(np.mean(gen))
        plot_data = plot_data[0:700]
        data.append((curr_int, plot_data))

    line1, = plt.plot(range(0, len(data[0][1])), data[0][1], 'k-', label='interval=' + str(data[0][0]))
    line2, = plt.plot(range(0, len(data[1][1])), data[1][1], 'm--', label='interval=' + str(data[1][0]))

    plt.legend(handles=[line1, line2], loc = 4)
    plt.ylabel('fitness')
    plt.xlabel('generation')

    plt.show()


# plots the best achieved fitness as a function of migration interval
def load_and_plot_experimental_data_2():
    intervals = []
    best_fitness = []
    with open('experiment_data_1_30_2.csv', 'rb') as csvfile:
        row_data = csv.reader(csvfile, delimiter=',', quotechar='|')
        curr_int= 1

        gen_to_fit = []
        plot_data = []

        for row in row_data:
            xp_num = int(row[0])
            gen_num = int(row[1])
            interval = int(row[5])
            fitness = float(row[2])

            if interval != curr_int:
                for gen in gen_to_fit:
                    plot_data.append(np.mean(gen))

                intervals.append(curr_int)
                best_fitness.append(max(plot_data))

                curr_int = interval
                gen_to_fit = []
                plot_data = []

            print str(gen_num) + ' ' + str (len(gen_to_fit)) + ' ' + str(interval)
            if len(gen_to_fit) < gen_num:
                gen_to_fit.append([])

            gen_to_fit[gen_num-1].append(fitness)

        for gen in gen_to_fit:
            plot_data.append(np.median(gen))
        intervals.append(curr_int)
        best_fitness.append(max(plot_data))

    plt.plot(intervals, best_fitness, 'm-')
    plt.ylabel('best fitness')
    plt.xlabel('interval')
    plt.show()


if __name__ == "__main__":
    explore_migration_interval()
    #explore_migration_size()
    #explore_island_vs_conventional()

    #load_and_plot_islands_vs_conventional()
    #load_and_plot_different_migration_sizes()
    #load_and_plot_experimental_data()
    #load_and_plot_experimental_data_3()