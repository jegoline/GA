import math
import random
import heapq

# the code was adapted from the
# http://www.theprojectspot.com/tutorial-post/applying-a-genetic-algorithm-to-the-travelling-salesman-problem/5


# represents city on the grid
class City(object):
    def __init__(self, x, y, label=None):
        self.x = x
        self.y = y
        self.label = label

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __repr__(self):
        return self.label


# represent world with cities
class World(object):
    def __init__(self):
        self.cities = []

    def add_city(self, city):
        self.cities.append(city)

    def get_city(self, index):
        return self.cities[index]

    def get_cities(self):
        return self.cities

    def number_of_cities(self):
        return len(self.cities)

    def __repr__(self):
        return self.cities


def calc_distance(city1, city2):
    x_diff = city1.x - city2.x
    y_diff = city1.y - city2.y
    return math.sqrt(pow(x_diff, 2) + pow(y_diff, 2))


# represent genes\solutions
class Route:
    def __init__(self, world, route=None):
        self.world = world
        self.route = [None] * world.number_of_cities() if route is None else route
        self.distance = 0

    def __repr__(self):
        return str(self.route)

    def get_city(self, index):
        return self.route[index]

    def set_city(self, index, city):
        self.route[index] = city
        self.distance = 0

    def get_fitness(self):
        return 1.0 / self.get_distance()

    def get_distance(self):
        if self.distance == 0:
            route_distance = 0
            route_size = len(self.route)
            for city_index in range(0, route_size):
                from_city = self.get_city(city_index)
                next_city_index = city_index + 1 if city_index + 1 < route_size else 0
                next_city = self.get_city(next_city_index)
                route_distance += calc_distance(from_city, next_city)
            self.distance = route_distance
        return self.distance

    def contains_city(self, city):
        return city in self.route


class Population:
    def __init__(self, population_size):
        self.routes = [None] * population_size

    def initialize_randomly(self, world):
        for i in range(0, len(self.routes)):
            new_route = Route(world, list(world.get_cities()))
            random.shuffle(new_route.route)
            self.set_route(i, new_route)

    def set_route(self, index, route):
        self.routes[index] = route

    def get_route(self, index):
        return self.routes[index]

    def size(self):
        return len(self.routes)

    def add_migrants(self, migrants):
        worst = heapq.nsmallest(len(migrants), self.routes, key=lambda s: s.get_fitness())
        for w in worst:
            self.routes.remove(w)

        for m in migrants:
            self.routes.append(m)

    def get_best(self):
        return max(self.routes, key=lambda x: x.get_fitness())

    def get_bests(self, n):
        return heapq.nlargest(n, self.routes, key=lambda s: s.get_fitness())


class GA:
    def __init__(self, world, mutation_rate, tournament_size, elitism, population_size, label=None):
        self.world = world
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.label = label
        # Initialize population
        self.population = Population(population_size)
        self.population.initialize_randomly(world)

    def get_best(self):
        return self.population.get_best()

    def get_bests(self, n):
        return self.population.get_bests(n)

    def add_migrants(self, routes):
        self.population.add_migrants(routes)

    # run algorithm
    def run(self, num_of_generations):
        for i in range(0, num_of_generations):
            self.population = self.evolve(self.population)

        return self.population.get_best()

    def evolve(self, population):
        new_population = Population(population.size())
        offset = 0

        # save most fittest pop
        if self.elitism:
            new_population.set_route(0, population.get_best())
            offset = 1

        # produce crossover offsprings and apply mutation
        for i in range(offset, new_population.size()):
            parent1 = self.select_parents(population)
            parent2 = self.select_parents(population)

            offspring = self.crossover(parent1, parent2)
            self.mutate(offspring)
            new_population.set_route(i, offspring)

        return new_population

    def crossover(self, parent1, parent2):
        offspring = Route(self.world)
        route_size = self.world.number_of_cities()
        rand1 = int(random.random() * route_size)
        rand2 = int(random.random() * route_size)

        start_pos = min(rand1, rand2)
        end_pos = max(rand1, rand2)

        # inherit part from parent1
        for i in range(0, route_size):
            if start_pos <= i < end_pos:
                offspring.set_city(i, parent1.get_city(i))

        # fill others city from parent2, keeping the order if possible
        for city in parent2.route:
            if not offspring.contains_city(city):
                for i in range(0, route_size):
                    if offspring.get_city(i) is None:
                        offspring.set_city(i, city)
                        break

        return offspring

    def mutate(self, route):
        route_size = self.world.number_of_cities()
        for pos1 in range(0, route_size):
            city1 = route.get_city(pos1)
            if random.random() < self.mutation_rate:
                pos2 = int(route_size * random.random())
                city2 = route.get_city(pos2)

                route.set_city(pos2, city1)
                route.set_city(pos1, city2)

    def select_parents(self, population):
        tournament = Population(self.tournament_size)
        for i in range(0, self.tournament_size):
            ind = int(random.random() * population.size())
            tournament.set_route(i, population.get_route(ind))
        return tournament.get_best()
