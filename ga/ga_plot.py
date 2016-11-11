from ga import City, World, Route
import ga
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

world = World()
while world.number_of_cities() < 50:
    x = random.random()
    y = random.random()

    if (0.2 >= x or x >= 0.8) or (0.2 >= y or y >= 0.8):
        x = int(x * 100)
        y = int(y * 100)
        city = City(x, y, str(world.number_of_cities()))
        world.add_city(city)

# 1
fittest = ga.run(world, 250, 250)
X = []
Y = []

for city in fittest.route:
    X.append(city.get_x())
    Y.append(city.get_y())

# 2
fittest = ga.run(world, 250, 250)
X1 = []
Y1 = []

for city in fittest.route:
    X1.append(city.get_x())
    Y1.append(city.get_y())

plt.plot(X, Y, 'bo', X, Y, 'r')
plt.plot(X1, Y1)
plt.show()
