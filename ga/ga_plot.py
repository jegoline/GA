from ga import City, World, GA

import matplotlib.pyplot as plt
import random


def plot_route(route, color='r'):
    x = []
    y = []
    for city in route.route:
        x.append(city.get_x())
        y.append(city.get_y())

    plt.plot(x, y, 'bo', x, y, color)


# generate world (all cities lay between 2 concentric squares)
def generate_world():
    w = World()
    while w.number_of_cities() < 50:
        x = random.random()
        y = random.random()

        if (0.2 >= x or x >= 0.8) or (0.2 >= y or y >= 0.8):
            x = int(x * 100)
            y = int(y * 100)
            city = City(x, y, str(w.number_of_cities()))

            w.add_city(city)
    return w


# initialize list of islands
islands = []
world = generate_world()
for i in range(0, 5):
    island = GA(world, 0.015, 3, True, 150, 'Island:' + str(i))
    islands.append(island)

# run evolution
for interval_num in range(1, 5):
    # evaluate N populations on every island
    for island in islands:
        fittest = island.run(40)
        print island.label + ' ' + str(fittest.get_distance())

    # perform migration (ring topology)
    migrant = islands[0].get_best()
    for i in range(1, len(islands)):
        fittest = islands[i].get_best()
        islands[i - 1].add_migrant(fittest)
        print 'Migrate from ' + islands[i].label + ' to ' + islands[i - 1].label + ' migrant ' \
              + str(fittest.get_distance())
    islands[len(islands) - 1].add_migrant(migrant)

    print 'Migrate from ' + islands[0].label + ' to ' + islands[len(islands) - 1].label + ' migrant ' \
          + str(migrant.get_distance())

# find fittest among all islands
fittest = None
for island in islands:
    island_fittest = island.get_best()
    if fittest is None:
        fittest = island_fittest
    elif fittest.get_fitness() < island_fittest.get_fitness():
        fittest = island_fittest


# plot fittest
plot_route(fittest)
print str(fittest.get_distance())
plt.show()
