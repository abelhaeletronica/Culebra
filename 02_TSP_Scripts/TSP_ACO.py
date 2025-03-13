"""Solve the travelling salesman problem for a number of cities with a brute-force, genetic, or ant colony optimization algorithm.
    Inputs:
        Cities: Optional list of 2d or 3d points or vectors, representing the cities to visit.
        Algorithm: 0 for brute-force, 1 for genetic algorithm, or 2 for ant colony optimization.
        Reset: True to reset the travelling salesman algorithm.
        Run: True to run, or False to pause the travelling salesman algorithm.
    Output:
        InitialOrder: List of 2d or 3d points or vectors, representing the cities in their initial order.
        BestOrder: List of 2d or 3d points or vectors, representing the cities in their current best order.
        CurrentOrder: List of 2d or 3d points or vectors, representing the cities in their currently evaluated order."""
        
        
__author__ = "p1r4t3b0y"


ghenv.Component.Name = "Pythonic Travelling Salesman Problem Solver"
ghenv.Component.NickName = "TravellingSalesman"


import Rhino.Geometry as rg
from scriptcontext import sticky
from sys import maxint
import random
import math
from collections import defaultdict


# Global variables
WIDTH = 300 # lateral boundaries for random cities, when none are input
HEIGHT = 300 # upper and lower boundaries for random cities, when none are input


class TSPBruteForce:
    """Brute-force algorithm for the Travelling Salesman Problem (TSP).
    
    Given a list of cities and the distances between each pair of cities,
    what is the shortest possible route to visit each city (and return 
    to the origin city)?
    
    Attributes:
      cities: None or a list of cities (points, vectors or lists of 3 coordinates)
    
    To use:
      >>> ts = TSPBruteForce(c)
      >>> ts.setup()
      >>> for i in range(r):
      >>>     ts.update()
      >>>     ts.get_current_best()
              [(x0,y0,z0), (x1,y1,z1), .., (xn, yn, zn)]
      >>>     ts.get_best_ever()
              [(x0,y0,z0), (x1,y1,z1), .., (xn, yn, zn)]
    """
    
    def __init__(self, cities=None):
        self.cities = cities
        self.num_cities = None
        self.permutations = None
        self.record_distance = maxint
        self.current_best_order = []
        self.best_ever_order = []
        self.count = 0
    
    def setup(self):
        """Sets the travelling salesman algorithm up."""
        order = []  # initial city order
        if len(self.cities) > 0:  # cities are provided externally
            self.num_cities = len(self.cities)
            # Create initial order
            for i in range(self.num_cities):
                order.append(i)
        else:  # cities are not provided
            self.num_cities = random.randint(3, 8)
            # Create initial cities and initial order
            self.cities = []
            for i in range(self.num_cities):
                city = self.vector_random(WIDTH/2, HEIGHT/2)
                self.cities.append(city)
                order.append(i)
        # Calculate total number of permutations
        self.permutations = self.factorial(self.num_cities)
        self.current_best_order = order
        self.best_ever_order = order
        # Update counter
        self.count += 1
    
    def update(self):
        """Updates the travelling salesman algorithm when called repeatedly."""
        # Calculate current distance
        d = self.calculate_distance(self.current_best_order)
        # If current distance is less than the record distance
        if d < self.record_distance:
            # Save the new record
            self.record_distance = d
            self.best_ever_order = self.current_best_order[:]
        # Get next lexicographical permutation
        self.current_best_order = self.next_lexicographical_permutation(self.current_best_order)
        # Update counter
        self.count += 1
    
    def vector_random(self, lenX=1, lenY=1, lenZ=None, rhino=True):
        """Creates a random vector."""
        x = random.uniform(-lenX, lenX)
        y = random.uniform(-lenY, lenY)
        z = 0.0
        if lenZ != None:
            z = random.uniform(-lenZ, lenZ)
        vec = [x, y, z]
        if rhino: 
            vec = rg.Vector3d(x,y,z)
        return vec
    
    def factorial(self, n):
        """Returns the factorial of n."""
        if n == 1:
            return n
        else:
            return n * self.factorial(n-1)
    
    def calculate_distance(self, order):
        """Calculates the total distance between ordered cities."""
        total = 0
        for i in range(len(order)-1):
            city_a_index = order[i]
            city_b_index = order[i+1]
            city_a = self.cities[city_a_index]
            city_b = self.cities[city_b_index]
            d = self.distance(city_a, city_b)
            total += d
        return total
    
    def distance(self, p1, p2):
        """Returns the distance between two points."""
        x2 = (p1[0] - p2[0])**2
        y2 = (p1[1] - p2[1])**2
        z2 = (p1[2] - p2[2])**2
        d = math.sqrt(x2 + y2 + z2)
        return d
    
    def swap(self, a, i, j):
        """Swaps values at index i and j in list a."""
        a[i], a[j] = a[j], a[i]
    
    def next_lexicographical_permutation(self, a):
        """Computes the next lexicographical permutation of list a in-place.
        Returns a if there is no next permutation.
        """
        i = len(a) - 2
        while not (i < 0 or a[i] < a[i+1]):
            i -= 1
        if i < 0:
            return a
        # else
        j = len(a) - 1
        while not (a[j] > a[i]):
            j -= 1
        self.swap(a, i, j)
        a[i+1:] = a[len(a)-1:i:-1]
        return a
    
    def get_current_best(self):
        """Returns the cities in their current best order."""
        ordered_cities = []
        for i in range(len(self.current_best_order)):
            n = self.current_best_order[i]
            ordered_cities.append(self.cities[n])
        return ordered_cities
    
    def get_best_ever(self):
        """Returns the cities in their best order so far."""
        ordered_cities = []
        for i in range(len(self.best_ever_order)):
            n = self.best_ever_order[i]
            ordered_cities.append(self.cities[n])
        return ordered_cities


class TSPGenetic:
    """Genetic algorithm for the Travelling Salesman Problem (TSP).
    
    Given a list of cities and the distances between each pair of cities,
    what is the shortest possible route to visit each city (and return 
    to the origin city)?
    
    Attributes:
      cities: None or a list of cities (points, vectors or lists of 3 coordinates)
    
    To use:
      >>> ts = TSPGenetic(c)
      >>> ts.setup()
      >>> for i in range(r):
      >>>     ts.update()
      >>>     ts.get_current_best()
              [(x0,y0,z0), (x1,y1,z1), .., (xn, yn, zn)]
      >>>     ts.get_best_ever()
              [(x0,y0,z0), (x1,y1,z1), .., (xn, yn, zn)]
    """
    
    def __init__(self, cities=None):
        self.cities = cities
        self.num_cities = None
        self.population = []
        self.fitness = []
        self.record_distance = maxint
        self.current_best_order = []
        self.best_ever_order = []
        self.count = 0
    
    def setup(self):
        """Sets the travelling salesman algorithm up."""
        order = []  # initial city order
        if len(self.cities) > 0:  # cities are provided externally
            self.num_cities = len(self.cities)
            # Create initial order
            for i in range(self.num_cities):
                order.append(i)
        else:  # cities are not provided
            self.num_cities = random.randint(3, 8)
            # Create initial cities and initial order
            self.cities = []
            for i in range(self.num_cities):
                city = self.vector_random(WIDTH/2, HEIGHT/2)
                self.cities.append(city)
                order.append(i)
        # Create population
        pop_size = 500
        for i in range(pop_size):
            population = order[:]
            random.shuffle(population)
            self.population.append(population)
        self.current_best_order = order
        self.best_ever_order = order
        # Update counter
        self.count += 1
    
    def update(self):
        """Updates the travelling salesman algorithm when called repeatedly."""
        # Calculate fitness
        self.calculate_fitness()
        # Get fittest order
        self.current_best_order = self.get_fittest()
        # Calculate current distance
        d = self.calculate_distance(self.current_best_order)
        # If current distance is less than the record distance
        if d < self.record_distance:
            # Save the new record
            self.record_distance = d
            self.best_ever_order = self.current_best_order[:]
        # Create next generation
        self.next_generation()
        # Update counter
        self.count += 1
    
    def vector_random(self, lenX=1, lenY=1, lenZ=None, rhino=True):
        """Creates a random vector."""
        x = random.uniform(-lenX, lenX)
        y = random.uniform(-lenY, lenY)
        z = 0.0
        if lenZ != None:
            z = random.uniform(-lenZ, lenZ)
        vec = [x, y, z]
        if rhino: 
            vec = rg.Vector3d(x,y,z)
        return vec
    
    def calculate_fitness(self):
        """Calculates the fitness of each member in the population."""
        self.fitness = []
        for i in range(len(self.population)):
            d = self.calculate_distance(self.population[i])
            f = 1 / (d**8 + 1)
            self.fitness.append(f)
    
    def calculate_distance(self, order):
        """Calculates the total distance between ordered cities."""
        total = 0
        for i in range(len(order)-1):
            city_a_index = order[i]
            city_b_index = order[i+1]
            city_a = self.cities[city_a_index]
            city_b = self.cities[city_b_index]
            d = self.distance(city_a, city_b)
            total += d
        return total
    
    def distance(self, p1, p2):
        """Returns the distance between two points."""
        x2 = (p1[0] - p2[0])**2
        y2 = (p1[1] - p2[1])**2
        z2 = (p1[2] - p2[2])**2
        d = math.sqrt(x2 + y2 + z2)
        return d
    
    def get_fittest(self):
        """Returns the fittest member in the population."""
        record = 0
        index = 0
        for i in range(len(self.fitness)):
            if self.fitness[i] > record:
                record = self.fitness[i]
                index = i
        return self.population[index][:]
    
    def normalize_fitness(self):
        """Normalizes the fitness of each member in the population."""
        total = sum(self.fitness)
        for i in range(len(self.fitness)):
            self.fitness[i] = self.fitness[i] / total
    
    def next_generation(self):
        """Creates the next generation through crossover and mutation."""
        self.normalize_fitness()
        new_population = []
        for i in range(len(self.population)):
            order_a = self.pick_one(self.population, self.fitness)[:]
            order_b = self.pick_one(self.population, self.fitness)[:]
            order = self.crossover(order_a, order_b)
            self.mutate(order, 0.01)
            new_population.append(order)
        self.population = new_population
    
    def pick_one(self, lst, prob):
        """Returns one element from a list based on probability values."""
        index = 0
        r = random.random()
        while r > 0:
            r = r - prob[index]
            index += 1
        index -= 1
        return lst[index]
    
    def crossover(self, order_a, order_b):
        """Creates a child from two parent orders."""
        start = random.randint(0, len(order_a)-1)
        end = random.randint(start+1, len(order_a))
        new_order = order_a[start:end]
        for i in range(len(order_b)):
            city = order_b[i]
            if not city in new_order:
                new_order.append(city)
        return new_order
    
    def mutate(self, order, mutation_rate):
        """Mutates an order based on a mutation rate."""
        for i in range(len(order)):
            if random.random() < mutation_rate:
                index_a = random.randint(0, len(order)-1)
                index_b = (index_a + 1) % len(order)
                self.swap(order, index_a, index_b)
    
    def swap(self, a, i, j):
        """Swaps values at index i and j in list a."""
        a[i], a[j] = a[j], a[i]
    
    def get_current_best(self):
        """Returns the cities in their current best order."""
        ordered_cities = []
        for i in range(len(self.current_best_order)):
            n = self.current_best_order[i]
            ordered_cities.append(self.cities[n])
        return ordered_cities
    
    def get_best_ever(self):
        """Returns the cities in their best order so far."""
        ordered_cities = []
        for i in range(len(self.best_ever_order)):
            n = self.best_ever_order[i]
            ordered_cities.append(self.cities[n])
        return ordered_cities


class TSPACO:
    """Ant Colony Optimization algorithm for the Travelling Salesman Problem (TSP).
    
    Given a list of cities and the distances between each pair of cities,
    what is the shortest possible route to visit each city (and return 
    to the origin city)?
    
    Attributes:
      cities: None or a list of cities (points, vectors or lists of 3 coordinates)
    
    To use:
      >>> ts = TSPACO(c)
      >>> ts.setup()
      >>> for i in range(r):
      >>>     ts.update()
      >>>     ts.get_current_best()
              [(x0,y0,z0), (x1,y1,z1), .., (xn, yn, zn)]
      >>>     ts.get_best_ever()
              [(x0,y0,z0), (x1,y1,z1), .., (xn, yn, zn)]
    """
    
    def __init__(self, cities=None):
        self.cities = cities
        self.num_cities = None
        self.cities_dists = dict()
        self.num_ants = 20
        self.ants = []  # lista de formigas
        self.pheromone_trails = defaultdict(float)
        self.record_distance = float('inf')
        self.current_best_order = []
        self.best_ever_order = []
        self.evaporation_rate = 0.1
        self.pheromone_strength = 1.0
        self.count = 0
    
    def setup(self):
        """Sets the travelling salesman algorithm up."""
        order = []  # initial city order
        if len(self.cities) > 0:  # cities are provided externally
            self.num_cities = len(self.cities)
            # Create initial order
            for i in range(self.num_cities):
                order.append(i)
        else:  # cities are not provided
            self.num_cities = random.randint(3, 8)
            # Create initial cities and initial order
            self.cities = []
            for i in range(self.num_cities):
                city = self.vector_random(WIDTH/2, HEIGHT/2)
                self.cities.append(city)
                order.append(i)
                
        self.current_best_order = order
        self.best_ever_order = order
        
        # Calcular distâncias entre cidades
        self.calculate_distances()
        
        # Inicializar formigas
        self.ants = [TSPAnt(self.cities) for _ in range(self.num_ants)]
        
        # Atualizar contador
        self.count += 1
        
    def update(self):
        """Updates the travelling salesman algorithm using ACO when called repeatedly."""
        # Mover todas as formigas
        completed_ants = []
        for ant in self.ants:
            finished = ant.select_next_city(self.pheromone_trails, self.cities_dists)
            if finished:
                if ant.total_distance > 0 and ant.total_distance < self.record_distance:
                    self.record_distance = ant.total_distance
                    self.best_ever_order = ant.path[:]
                completed_ants.append(ant)
                
        # Resetar formigas que completaram o circuito
        for ant in completed_ants:
            ant.reset()
            
        # Evaporar feromônios
        for trail in self.pheromone_trails:
            self.pheromone_trails[trail] *= (1.0 - self.evaporation_rate)
            
        # Depositar novos feromônios
        for ant in completed_ants:
            if ant.total_distance > 0:  # Evitar divisão por zero
                deposit = self.pheromone_strength / ant.total_distance
                path = ant.path[:]  # Fazer uma cópia do caminho
                for i in range(len(path)-1):
                    trail_key = tuple(sorted([path[i], path[i+1]]))
                    self.pheromone_trails[trail_key] += deposit
                
        # Atualizar melhor caminho atual
        if completed_ants:
            valid_ants = [ant for ant in completed_ants if ant.total_distance > 0]
            if valid_ants:
                best_ant = min(valid_ants, key=lambda x: x.total_distance)
                self.current_best_order = best_ant.path[:]
            
        # Atualizar contador
        self.count += 1
        
    def vector_random(self, lenX=1, lenY=1, lenZ=None, rhino=True):
        """Creates a random vector."""
        x = random.uniform(-lenX, lenX)
        y = random.uniform(-lenY, lenY)
        z = 0.0
        if lenZ != None:
            z = random.uniform(-lenZ, lenZ)
        vec = [x, y, z]
        if rhino: 
            vec = rg.Vector3d(x,y,z)
        return vec
        
    def calculate_distances(self):
        """Calculates all the distances between the cities."""
        for i in range(len(self.cities)):
            self.cities_dists[i] = dict()
            for j in range(len(self.cities)):
                if i != j:
                    x2 = (self.cities[i][0] - self.cities[j][0])**2
                    y2 = (self.cities[i][1] - self.cities[j][1])**2
                    z2 = (self.cities[i][2] - self.cities[j][2])**2
                    dist = math.sqrt(x2 + y2 + z2)
                    self.cities_dists[i][j] = dist
                    
    def get_current_best(self):
        """Returns the cities in their current best order."""
        ordered_cities = [self.cities[i] for i in self.current_best_order]
        return ordered_cities
        
    def get_best_ever(self):
        """Returns the cities in their best order so far."""
        ordered_cities = [self.cities[i] for i in self.best_ever_order]
        return ordered_cities


class TSPAnt:
    def __init__(self, cities):
        self.cities = cities
        self.reset()
        
    def reset(self):
        """Reseta o estado da formiga para começar uma nova busca."""
        self.current_city = 0
        self.path = [0]  # Começa na cidade 0
        self.total_distance = 0.0
        self.visited = {0}  # Conjunto de cidades visitadas
        
    def select_next_city(self, pheromone_trails, distances):
        """Seleciona a próxima cidade para visitar baseado em feromônios e distâncias."""
        # Se todas as cidades foram visitadas, voltar para a cidade inicial
        if len(self.visited) == len(self.cities):
            if self.current_city != 0:  # Se não estiver na cidade inicial
                self.path.append(0)
                self.total_distance += distances[self.current_city][0]
            return True
            
        # Calcular probabilidades para cidades não visitadas
        probabilities = []
        unvisited = [i for i in range(len(self.cities)) if i not in self.visited]
        
        for city_idx in unvisited:
            trail_key = tuple(sorted([self.current_city, city_idx]))
            pheromone = pheromone_trails.get(trail_key, 0.1)
            distance = max(distances[self.current_city][city_idx], 0.1)  # Evitar divisão por zero
            
            probability = (pheromone ** 1.0) * ((1.0 / distance) ** 2.0)
            probabilities.append((city_idx, probability))
            
        # Selecionar próxima cidade
        total = sum(p[1] for p in probabilities)
        if total <= 0:
            next_city = random.choice(unvisited)
        else:
            r = random.random() * total
            cumsum = 0
            for city_idx, prob in probabilities:
                cumsum += prob
                if cumsum >= r:
                    next_city = city_idx
                    break
            else:
                next_city = probabilities[-1][0]
        
        # Atualizar estado da formiga
        self.path.append(next_city)
        self.visited.add(next_city)
        self.total_distance += distances[self.current_city][next_city]
        self.current_city = next_city
        
        return False


def update_component():
    """Updates this component, similar to using a Grasshopper timer"""
    import Grasshopper as gh
    def call_back(e):
        ghenv.Component.ExpireSolution(False)
    ghDoc = ghenv.Component.OnPingDocument()
    ghDoc.ScheduleSolution(1, gh.Kernel.GH_Document.GH_ScheduleDelegate(call_back))


# Initialize or reset count, and reset the Travelling Salesman algorithm
if "count" not in globals() or Reset:
    count = 0
    sticky.pop("TSPA", None)
    ghenv.Component.Message = "Reset"


# Run or pause the Travelling Salesman algorithm
if Run:
    if not "TSPA" in sticky.keys():
        # Initialize the travelling salesman algorithm
        if Algorithm == 0 or Algorithm == None:  # Brute-force algorithm
            ts = TSPBruteForce(Cities)
        elif Algorithm == 1:  # Genetic algorithm
            ts = TSPGenetic(Cities)
        else:  # ACO algorithm (Algorithm == 2)
            ts = TSPACO(Cities)
        ts.setup()
        sticky["TSPA"] = ts
        
    else:
        # Run the travelling salesman algorithm
        ts = sticky["TSPA"]
        ts.update()
        sticky["TSPA"] = ts
    
    # Dynamically increment the counter value
    if Algorithm == 0 or Algorithm == None:  # Brute-force algorithm
        if ts.count <= ts.permutations: 
            BestOrder = ts.get_best_ever()
            CurrentOrder = ts.get_current_best()
            update_component()
            if ts.num_cities < 12:
                ghenv.Component.Message = "Running... (%.3f%%)" %((100 * (ts.count / ts.permutations)))
            else:
                ghenv.Component.Message = "Running... (Gen. %d)" %(ts.count)
        else:
            BestOrder = ts.get_best_ever()
            ghenv.Component.Message = "Completed (100%)"
    
    elif Algorithm == 1:  # Genetic algorithm
        BestOrder = ts.get_best_ever()
        CurrentOrder = ts.get_current_best()
        update_component()
        ghenv.Component.Message = "Running... (Gen. %d)" %(ts.count)
        
    else:  # ACO algorithm
        BestOrder = ts.get_best_ever()
        CurrentOrder = ts.get_current_best()
        update_component()
        ghenv.Component.Message = "Running... (Colony %d)" %(ts.count)

else:
    # Pause the travelling salesman algorithm
    if "TSPA" in sticky.keys():
        ts = sticky["TSPA"]
        BestOrder = ts.get_best_ever()
        CurrentOrder = ts.get_current_best()
        ghenv.Component.Message = "Paused"


InitialOrder = Cities  # the same as 'Cities' input

def calculate_total_distance(path):
    total_distance = 0
    for i in range(len(path)):
        city1 = path[i]
        city2 = path[(i + 1) % len(path)]  # Volta para a primeira cidade no final
        dx = city2[0] - city1[0]
        dy = city2[1] - city1[1]
        dz = city2[2] - city1[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        total_distance += distance
    return total_distance

# Test with specific coordinates
test_cities = [
    (88.055764, 209.274364, 0),
    (7.47693, 124.173781, 0),
    (2.393017, 12.015346, 0),
    (107.900005, 80.674085, 0),
    (195.56669, 40.259868, 0),
    (299.534247, 29.258372, 0),
    (219.253889, 127.005492, 0),
    (279.603777, 246.804659, 0)
]

# Calcular distância na ordem original
print(f"Distância total (ordem original): {calculate_total_distance(test_cities):.2f}")

# Algumas ordens diferentes para teste
orders = [
    [0, 1, 2, 3, 4, 5, 6, 7],  # ordem original
    [0, 7, 6, 5, 4, 3, 2, 1],  # ordem reversa
    [0, 2, 4, 6, 1, 3, 5, 7],  # alternando índices
]

for i, order in enumerate(orders):
    path = [test_cities[i] for i in order]
    print(f"Distância total (ordem teste {i+1}): {calculate_total_distance(path):.2f}")