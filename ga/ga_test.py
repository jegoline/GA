import unittest
from ga import City, World, Route, calc_distance


class MyTestCase(unittest.TestCase):

    def test_city_distance(self):
        city1 = City(1, 1, "A")
        city2 = City(5, 4, "B")
        self.assertEquals(5, calc_distance(city1, city2))

    def test_route_distance(self):
        world = World()
        city1 = City(1, 1, "A")
        world.add_city(city1)
        city2 = City(5, 4, "B")
        world.add_city(city2)

        self.assertEquals(2, world.number_of_cities())

        route = Route(world)
        route.set_city(0, city1)
        route.set_city(1, city2)

        self.assertEquals(10, route.get_distance())
        self.assertEquals(1.0/10, route.get_fitness())

if __name__ == '__main__':
    unittest.main()
