"""Solve the travelling salesman problem for a number of cities with a brute-force or genetic algorithm.
    Inputs:
        Cities: Optional list of 2d or 3d points or vectors, representing the cities to visit.
        Algorithm: 0 for the brute-force algorithm, or 1 for genetic algorithm.
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
        self.cities_dists = dict()
        self.record_distance = 0
        self.current_best_order = []
        self.best_ever_order = []
        self.permutations = 0
        self.count = 0
        
    
    def setup(self):
        """Sets the travelling salesman algorithm up."""
        order = [] # initial city order
        if len(self.cities) > 0: # cities are provided externally
            self.num_cities = len(self.cities)
            # Create initial order
            for i in range(self.num_cities):
                order.append(i)
        else: # cities are not provided
            self.num_cities = random.randint(3, 8)
            # Create initial cities and initial order
            self.cities = []
            for i in range(self.num_cities):
                city = self.vector_random(WIDTH/2, HEIGHT/2)
                self.cities.append(city)
                order.append(i)
        # Set the initial current (best) order
        self.current_best_order = order
        
        # Calculates all distances between the cities in their initial order
        self.calculate_distances()
        # Set the initial (record) distance
        self.record_distance = self.sum_distances(order)
        
        # Calculate the total number of permutations
        self.permutations = math.factorial(len(self.cities))
        # Update the permutation counter for the setup step
        self.count += 1
    
    
    def update(self):
        """Updates the travelling salesman algorithm by lexicographic order, when called repeatedly."""
        # Find the largest x, such that P[x] < P[x+1]
        lx = None
        for x in range(len(self.current_best_order)-1):
            if self.current_best_order[x] < self.current_best_order[x+1]:
                lx = x
            
        if lx != None:
            # Find the largest y, such that P[x] < P[y]
            ly = None
            for y in range(len(self.current_best_order)):
                if self.current_best_order[lx] < self.current_best_order[y]:
                    ly = y
            
            # Swap P[x] and P[y]
            tmp = self.current_best_order[lx]
            self.current_best_order[lx] = self.current_best_order[ly]
            self.current_best_order[ly] = tmp
            
            # Reverse P[x+1 .. n]
            start_lt = self.current_best_order[:lx+1]
            end_lt = self.current_best_order[lx+1::]
            rev_end_lt = list(reversed(end_lt))
            self.current_best_order = list(start_lt + rev_end_lt)
            
            # Compares the total distance of the current cities order to the record distance
            dsum = self.sum_distances(self.current_best_order)
            if dsum < self.record_distance:
                self.record_distance = dsum
                self.best_ever_order = list(self.current_best_order)
        
        # Update the permutation counter for each update step
        self.count += 1

    
    def vector_random(self, lenX=1, lenY=1, lenZ=None, rhino=True):
        """Creates a random vector.
    
        Args:
          lenX: An upper and lower limit (number) in x-direction.
          lenY: An upper and lower limit (number) in y-direction.
          lenZ: None, or an upper and lower limit (number) in y-direction.
          rhino: 
            If True a Rhino.Geometry.Vector3d() is returned. 
            If False a list of 3 coordinates [number, number, number] is returned.
        Returns:
          The new vector.
        """
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
        """Calculates all the distances between the cities in self.cities.
    
        The distances are calculated in the initial order of self.cities and 
        saved in the structured dictionary of dictionaries self.cities_dists.
        The dictionary keys corrsepond to item indices of self.cities and the values
        to new dictionaries, with item indices as keys and distances as values.
        Each upper level item index forms pairs, with its lower level item indices,
        that represent the cities that the distance was measured between. 
    
        For example:
        {0: {1: distance from self.cities[0] to self.cities[1], 
             2: distance from self.cities[0] to self.cities[2]},
         1: {0: distance from self.cities[1] to self.cities[0], 
             2: distance from self.cities[1] to self.cities[2]},
         i: {0: distance from self.cities[i] to self.cities[0], 
             1: distance from self.cities[i] to self.cities[1]}}
        """
        for i in range(len(self.cities)):
            self.cities_dists[i] = dict()
            for j in range(len(self.cities)):
                if i != j:
                    x2 = (self.cities[i][0] - self.cities[j][0])**2
                    y2 = (self.cities[i][1] - self.cities[j][1])**2
                    z2 = (self.cities[i][2] - self.cities[j][2])**2
                    dist = math.sqrt(x2 + y2 + z2)
                    self.cities_dists[i][j] = dist
    
    
    def sum_distances(self, order):
        """Calculates the sum of distances between the cities in a given order.
        
        Args:
          order: A list of indices (integers) for each city in cities.
        Returns:
          The sum of distances (number).
        """
        dsum = 0
        for i in range(len(order)-1):
            curr_order = order[i]
            next_order = order[i+1]
            dist = self.cities_dists[curr_order][next_order]
            dsum += dist
        return dsum
        
     
    def get_current_best(self):
        """Returns the cities in their current best order."""
        ordered_cities = [self.cities[i] for i in self.current_best_order]
        return ordered_cities
        
    
    def get_best_ever(self):
        """Returns the cities in their best order so far."""
        ordered_cities = [self.cities[i] for i in self.best_ever_order]
        return ordered_cities   
        



class TSPGenetic:
    """Genetic algorithm for the Travelling Salesman Problem (TSP).
    
    Given a list of cities and the distances between each pair of cities,
    what is the shortest possible route to visit each city (and return 
    to the origin city)?
    
    Attributes:
      cities: None or a list of cities (points, vectors or lists of 3 coordinates)
    
    To use:
      >>> ts = TSPGenetic(n)
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
        self.num_populations = 20
        self.population = [] # list of order lists
        self.record_distance = maxint
        self.current_best_order = []
        self.best_ever_order = []
        self.mutation_rate = 0.05
        self.fitness = []
        self.permutations = 0
        self.count = 0
    
    
    def setup(self):
        """Sets the travelling salesman algorithm up."""
        order = [] # initial city order
        if len(self.cities) > 0: # cities are provided externally
            self.num_cities = len(self.cities)
            # Create initial order
            for i in range(self.num_cities):
                order.append(i)
        else: # cities are not provided externally
            self.num_cities = random.randint(3, 8)
            # Create initial cities and initial order
            self.cities = []
            for i in range(self.num_cities):
                city = self.vector_random(WIDTH/2, HEIGHT/2)
                self.cities.append(city)
                order.append(i)
        self.current_best_order = order
        
        # Calculates all distances between the cities in their initial order
        self.calculate_distances()
        
        # Create the population of random city orders
        for _ in range(self.num_populations):
            tmp_order = list(order)
            random.shuffle(tmp_order)
            self.population.append(tmp_order)
        
        # Calculate the total number of permutations
        self.permutations = math.factorial(len(self.cities))
        # Update the permutation counter for the setup step
        self.count += 1

    
    def update(self):
        """Updates the travelling salesman algorithm genetically, when called repeatedly."""
        self.calculate_fitness()
        self.normalize_fitness()
        self.next_generation()
        # Update the permutation counter for each update step
        self.count += 1

    
    def vector_random(self, lenX=1, lenY=1, lenZ=None, rhino=True):
        """Creates a random vector.
    
        Args:
          lenX: An upper and lower limit (number) in x-direction.
          lenY: An upper and lower limit (number) in y-direction.
          lenZ: None, or an upper and lower limit (number) in y-direction.
          rhino: 
            If True a Rhino.Geometry.Vector3d() is returned. 
            If False a list of 3 coordinates [number, number, number] is returned.
        Returns:
          The new vector.
        """
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
        """Calculates all the distances between the cities in self.cities.
    
        The distances are calculated in the initial order of self.cities and 
        saved in the structured dictionary of dictionaries self.cities_dists.
        The dictionary keys corrsepond to item indices of self.cities and the values
        to new dictionaries, with item indices as keys and distances as values.
        Each upper level item index forms pairs, with its lower level item indices,
        that represent the cities that the distance was measured between. 
    
        For example:
        {0: {1: distance from self.cities[0] to self.cities[1], 
             2: distance from self.cities[0] to self.cities[2]},
         1: {0: distance from self.cities[1] to self.cities[0], 
             2: distance from self.cities[1] to self.cities[2]},
         i: {0: distance from self.cities[i] to self.cities[0], 
             1: distance from self.cities[i] to self.cities[1]}}
        """
        for i in range(len(self.cities)):
            self.cities_dists[i] = dict()
            for j in range(len(self.cities)):
                if i != j:
                    x2 = (self.cities[i][0] - self.cities[j][0])**2
                    y2 = (self.cities[i][1] - self.cities[j][1])**2
                    z2 = (self.cities[i][2] - self.cities[j][2])**2
                    dist = math.sqrt(x2 + y2 + z2)
                    self.cities_dists[i][j] = dist
    
    
    def sum_distances(self, order):
        """Calculates the sum of distances between the cities in a given order.
        
        Args:
          order: A list of indices (integers) for each city in cities.
        Returns:
          The sum of distances (number).
        """
        dsum = 0
        for i in range(len(order)-1):
            curr_order = order[i]
            next_order = order[i+1]
            dist = self.cities_dists[curr_order][next_order]
            dsum += dist
        return dsum
    
    
    def calculate_fitness(self):
        """Calculates fitness values between 0 and 1.
        
        For each random population order, corresponding to cities indices,
        a total distance is calculated, remapped between 0 and 1, and
        saved as single fitness value per population order. 
        A small distance is translated to a big fitness value and vis versa. 
        """
        current_record = maxint
        new_fitness = []
        for i in range(len(self.population)):
            #dist = self.calculate_distance(self.cities, self.population[i])
            dist = self.sum_distances(self.population[i])
            if dist < self.record_distance:
                self.record_distance = dist
                self.best_ever_order = self.population[i]
            if dist < current_record:
                current_record = dist
                self.current_best_order = self.population[i]
            new_fitness.append(1 / (dist + 1)) # 0 < distance < 1 (+1 prevents dividing by 0) 
        self.fitness = new_fitness
    
    
    def normalize_fitness(self):
        """Remaps all fitness values to decimal probability percentages that add up to 1.0 (equals 100%)."""
        fsum = sum(self.fitness) # sum of all fitness values
        for i in range(len(self.fitness)):
            self.fitness[i] /= fsum
        
            
    def pick_one(self, vals, probs):
        """Picks the single fittest item from a list.
        
        Args:
          vals: A list of values or a list of lists of values.
          probs: A list of probabilities, each number between 0 and 1.
            The list length of probs must be equal to the list length of vals.
        Returns:
          The picked value or a copy of the picked value list.
        """
        idx = 0
        r = random.random()
        while r > 0:
            r -= probs[idx]
            idx += 1    
        idx -= 1
        if any(isinstance(i, list) for i in vals):
            return list(vals[idx])
        else:
            return vals[idx]
    

    def crossover(self, valsA, valsB):
        """Performs a modified single-point crossover on two lists of chromosomes.

        An index from valsA is randomly picked and designated as crossover point. 
        Bits to the right or left of that point from valsA, randomly form the beginning 
        of a new, recombined list that is then filled with unique values from valsB. 

        Args:
          valsA: A list of values representing a set of chromosomes.
          valsB: A list of values representing a set of chromosomes.
            The list length of valsB must be equal to the list length of valsA.
        Returns:
          A new, recombined chromosomes list.
        """
        idx = random.randint(0, len(valsA)-1)
        if random.random() < 0.5: # Keep front bits
            new_vals = valsA[:idx] 
        else: # Keep back bits
            new_vals = valsA[idx:] 
        for val in valsB:
            if val not in new_vals:
                new_vals.append(val)
        return new_vals
        
    
    def mutate(self, vals, rate):
        """Mutates a list of chromosome by performing a number of random swapps on two of its items.
        
        Args:
          vals: A list of chromosomes.
          rate: A mutation rate between 0 and 1.
            Determines the frequency of random shuffles, where 1
            means excessive shuffling, and 0 no shuffling at all.
        Returns:
          The mutated chromosomes list.
        """
        for _ in range(self.num_cities):
            if random.random() < rate:
                i = random.randint(0, len(vals)-1)
                #j = random.randint(0, len(vals)-1)
                j = (i + 1) % self.num_cities
                tmp = vals[i]
                vals[i] = vals[j]
                vals[j] = tmp
        return vals
        
    
    def next_generation(self):
        """Generates a fitter, new population for the next generation.
        
        The fittest order list is picked for each order list of the current 
        population and mutated. The resulting new population replaces the 
        current one for the next generation. 
        """
        new_population = []
        for i in range(len(self.population)):
            orderA = self.pick_one(self.population, self.fitness)
            orderB = self.pick_one(self.population, self.fitness)
            order = self.crossover(orderA, orderB)
            order = self.mutate(order, self.mutation_rate) # rate can be low if crossover is integrated
            new_population.append(order)
        self.population = new_population    
    
    
    def get_current_best(self):
        """Returns the cities in their current best order."""
        ordered_cities = [self.cities[i] for i in self.current_best_order]
        return ordered_cities
        
    
    def get_best_ever(self):
        """Returns the cities in their best order so far."""
        ordered_cities = [self.cities[i] for i in self.best_ever_order]
        return ordered_cities


###################################################################################


def update_component():
    """Updates this component, similar to using a Grasshopper timer"""
    # written by Anders Deleuran, andersholdendeleuran.com
    import Grasshopper as gh
    def call_back(e):
        """Defines a callback action"""
        ghenv.Component.ExpireSolution(False)
    # Get the Grasshopper document
    ghDoc = ghenv.Component.OnPingDocument() 
    # Schedule this component to expire
    ghDoc.ScheduleSolution(1, gh.Kernel.GH_Document.GH_ScheduleDelegate(call_back))
    

###################################################################################


# Initialize or reset count, and reset the Travelling Salesman algorithm
if "count" not in globals() or Reset:
    count = 0
    sticky.pop("TSPA", None)
    ghenv.Component.Message = "Reset"

# Run or pause the Travelling Salesman algorithm
if Run:
    if not "TSPA" in sticky.keys():
        # Initialize the travelling salesman algorithm
        if Algorithm == 0 or Algorithm == None: # Brute-force algorithm
            ts = TSPBruteForce(Cities)
        if Algorithm == 1: # Genetic algorithm
            ts = TSPGenetic(Cities)
        ts.setup()
        sticky["TSPA"] = ts
        
    else:
        # Run the travelling salesman algorithm
        ts = sticky["TSPA"]
        ts.update()
        sticky["TSPA"]= ts
    
    # Dynamically increment the counter value
    ## Brute-force algorithm incrementation
    if Algorithm == 0 or Algorithm == None: 
        if ts.count <= ts.permutations: 
            BestOrder = ts.get_best_ever()
            CurrentOrder = ts.get_current_best()
            update_component()
            if ts.num_cities < 12: # message with three decimal places
                ghenv.Component.Message = "Running... (%.3f%%)" %((100 * (ts.count / ts.permutations)))
            else: # message with number of permutations (since percentage too imprecise for huge integers)
                ghenv.Component.Message = "Running... (Gen. %d)" %(ts.count)
        else:
            BestOrder = ts.get_best_ever()
            ghenv.Component.Message = "Completed (100%)"
    
    ## Genetic algorithm incrementation
    if Algorithm == 1:
        BestOrder = ts.get_best_ever()
        CurrentOrder = ts.get_current_best()
        update_component()
        ghenv.Component.Message = "Running... (Gen. %d)" %(ts.count)

else:
    # Pause the travelling salesman algorithm
    if "TSPA" in sticky.keys():
        ts = sticky["TSPA"]
        ts.update()
        sticky["TSPA"] = ts
        BestOrder = ts.get_best_ever()
        CurrentOrder = ts.get_current_best()
        ghenv.Component.Message = "Paused"


InitialOrder = Cities # the same as 'Cities' input