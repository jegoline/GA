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
    world = World()
    while world.number_of_cities() < 50:
        x = random.random()
        y = random.random()

        if (0.2 >= x or x >= 0.8) or (0.2 >= y or y >= 0.8):
            x = int(x * 100)
            y = int(y * 100)
            city = City(x, y, str(world.number_of_cities()))
            world.add_city(city)
    return world


# initialize list of islands
islands = []
world = generate_world()
for i in range(0, 5):
    island = GA(world, 0.095, 3, True, 200, 'Island:' + str(i))
    islands.append(island)

# run evolution
for interval_num in range(1, 5):
    # evaluate N populations on every island
    for island in islands:
        fittest = island.run(40)
        print island.label + ' ' + str(fittest.get_distance())

    # perform migration (ring topology)
    migrant = islands[0].get_fittest()
    for i in range(1, len(islands)):
        fittest = islands[i].get_fittest()
        islands[i - 1].add_migrant(fittest)

    islands[len(islands) - 1].add_migrant(migrant)

fittest = None
for island in islands:
    island_fittest = island.get_fittest()
    if fittest is None:
        fittest = island_fittest
    elif fittest.get_fitness() < island_fittest.get_fitness():
        fittest = island_fittest

# plot fittest
plot_route(fittest)
print str(fittest.get_distance())
plt.show()
