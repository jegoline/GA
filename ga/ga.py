import math
import random
import copy


# represents city on the grid
class City:
    def __init__(self, x, y, label=None):
        self.x = x
        self.y = y
        self.label = label

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def distance_to(self, city):
        x_diff = abs(self.x - city.x)
        y_diff = abs(self.y - city.y)
        distance = math.sqrt(pow(x_diff, 2) + pow(y_diff, 2))

        return distance

    def __repr__(self):
        return self.label


class World:
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
        return 1 / float(self.get_distance())

    def get_distance(self):
        if self.distance == 0:
            route_distance = 0
            route_size = len(self.route)
            for city_index in range(0, route_size):
                from_city = self.get_city(city_index)
                next_city_index = city_index + 1 if city_index + 1 < route_size else 0
                next_city = self.get_city(next_city_index)
                route_distance += from_city.distance_to(next_city)
            self.distance = route_distance
        return self.distance

    def contains_city(self, city):
        return city in self.route


class Population:
    def __init__(self, population_size):
        self.routes = [None] * population_size

    def initialize_randomly(self, world):
        for i in range(0, len(self.routes)):
            new_route = Route(world, copy.copy(world.get_cities()))
            random.shuffle(new_route.route)
            self.save_route(i, new_route)

    def save_route(self, index, route):
        self.routes[index] = route

    def get_route(self, index):
        return self.routes[index]

    def get_fittest(self):
        fittest = self.routes[0]
        for t in self.routes:
            if fittest.get_fitness() <= t.get_fitness():
                fittest = t
        return fittest

    def population_size(self):
        return len(self.routes)


class GA:
    def __init__(self, world, mutation_rate, tournment_size):
        self.world = world
        self.mutation_rate = mutation_rate
        self.tournment_size = tournment_size
        self.elitism = False

    def evolve_population(self, population):
        new_population = Population(population.population_size())
        elitism_offset = 0

        # save most fittest pop
        if self.elitism:
            new_population.save_route(0, population.get_fittest())
            elitism_offset = 1

        # produce crossover offsprings
        for i in range(elitism_offset, new_population.population_size()):
            parent1 = self.select_parents(population)
            parent2 = self.select_parents(population)

            offspring = self.crossover(parent1, parent2)
            new_population.save_route(i, offspring)

        # mutate offsprings
        for i in range(elitism_offset, new_population.population_size()):
            self.mutate(new_population.get_route(i))

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
            if random.random() < self.mutation_rate:
                pos2 = int(route_size * random.random())

                city1 = route.get_city(pos1)
                city2 = route.get_city(pos2)

                route.set_city(pos2, city1)
                route.set_city(pos1, city2)

    def select_parents(self, population):
        tournament = Population(self.tournment_size)
        for i in range(0, self.tournment_size):
            ind = int(random.random() * population.population_size())
            tournament.save_route(i, population.get_route(ind))
        fittest = tournament.get_fittest()

        return fittest


# run algorithm
def run(world, population_size, num_of_generations):
    # Initialize population
    population = Population(population_size);
    population.initialize_randomly(world)

    print "Initial distance: " + str(population.get_fittest().get_distance())

    ga = GA(world, 0.015, 5)
    for i in range(0, num_of_generations):
        print "Iteration: " + str(i) + " " + str(population.get_fittest().get_distance()) + " " + str(population.get_fittest())
        population = ga.evolve_population(population)

    print "Final distance: " + str(population.get_fittest().get_distance())
    return population.get_fittest()