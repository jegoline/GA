from ga_algorithm import City, World
import random
import jsonpickle
import os


def get_world():
    file_exist = os.path.isfile('world.json')
    if file_exist and os.path.getsize('world.json') != 0:
        world = load_world('world.json')
    else:
        world = generate_world()
        save_world(world, 'world.json')
    return world


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


def plot_route(ax, route, color='r'):
    x = []
    y = []
    ax.clear()
    for city in route.route:
        x.append(city.get_x())
        y.append(city.get_y())

    ax.plot(x, y, 'bo', x, y, color)

    ax.set_title("%.2f" % route.get_distance())
    for i in range(1, len(x)):
        ax.annotate(str(i), (x[i], y[i]))


# perform migration (ring topology)
def migrate (islands, num_migrants):
    migrants = islands[0].get_bests(num_migrants)
    for i in range(1, len(islands)):
        fittest_n = islands[i].get_bests(num_migrants)
        islands[i - 1].add_migrants(fittest_n)
    islands[len(islands) - 1].add_migrants(migrants)


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