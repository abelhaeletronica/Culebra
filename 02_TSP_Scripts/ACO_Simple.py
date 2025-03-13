"""
Ant Colony Optimization para Grasshopper
---------------------------------
Inputs:
    x0: Point - Posição do ninho
    x1: Point - Posição da comida
    x2: Integer - Número de formigas
    x3: Number - Taxa de evaporação
    x4: Number - Força do feromônio
    x5: Boolean - Reset
Outputs:
    a0: List of Points - Posições das formigas
    a1: Tree of Points - Trilhas de feromônios
    a2: List of Numbers - Força das trilhas
    a3: Integer - Número de formigas com comida
"""

import Rhino.Geometry as rg
import random
import math
from collections import defaultdict
import ghpythonlib.components as ghc
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path

class Ant:
    def __init__(self, pos, nest_pos, food_pos):
        self.position = pos
        self.velocity = rg.Vector3d(random.uniform(-1,1), random.uniform(-1,1), 0)
        self.nest_pos = nest_pos
        self.food_pos = food_pos
        self.has_food = False
        self.attraction_strength = 2.0
        self.pheromone_strength = 1.5
        self.wander_strength = 0.3
        self.max_speed = 5.0
        self.search_radius = 5.0

    def update(self, pheromone_trails):
        # Define o alvo com base no estado da formiga
        target = self.food_pos if not self.has_food else self.nest_pos
        
        # Força de atração para o objetivo
        direction = rg.Vector3d(
            target.X - self.position.X,
            target.Y - self.position.Y,
            target.Z - self.position.Z
        )
        if direction.Length > 0:
            direction.Unitize()
        attraction = direction * self.attraction_strength
        
        # Força dos feromônios
        pheromone_force = self._calculate_pheromone_force(pheromone_trails)
        
        # Força aleatória para exploração
        wander = rg.Vector3d(random.uniform(-1,1), random.uniform(-1,1), 0)
        wander.Unitize()
        wander *= self.wander_strength
        
        # Combinar forças
        total_force = attraction + pheromone_force + wander
        
        # Atualizar velocidade
        self.velocity += total_force
        if self.velocity.Length > self.max_speed:
            self.velocity.Unitize()
            self.velocity *= self.max_speed
        
        # Atualizar posição
        self.position = rg.Point3d(
            self.position.X + self.velocity.X,
            self.position.Y + self.velocity.Y,
            self.position.Z
        )
        
        # Verificar se encontrou comida ou voltou ao ninho
        if not self.has_food:
            if self.position.DistanceTo(self.food_pos) < 2.0:
                self.has_food = True
        else:
            if self.position.DistanceTo(self.nest_pos) < 2.0:
                self.has_food = False

    def _calculate_pheromone_force(self, pheromone_trails):
        force = rg.Vector3d(0,0,0)
        current_pos = (self.position.X, self.position.Y, self.position.Z)
        
        for trail_pos, strength in pheromone_trails.items():
            dx = trail_pos[0] - current_pos[0]
            dy = trail_pos[1] - current_pos[1]
            dz = trail_pos[2] - current_pos[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if distance < self.search_radius and distance > 0:
                direction = rg.Vector3d(dx/distance, dy/distance, dz/distance)
                force += direction * (strength * self.pheromone_strength * (1.0 - distance/self.search_radius))
        
        return force

# Variáveis globais para manter o estado entre execuções
if 'ants' not in globals():
    ants = []
    pheromone_trails = defaultdict(float)
    previous_reset = False

# Valores padrão para inputs
if not x0: x0 = rg.Point3d(0,0,0)  # Posição do ninho
if not x1: x1 = rg.Point3d(50,0,0)  # Posição da comida
if not x2: x2 = 50  # Número de formigas
if not x3: x3 = 0.1  # Taxa de evaporação
if not x4: x4 = 1.0  # Força do feromônio
if not x5: x5 = False  # Reset

# Garantir que os inputs são do tipo correto
nest_pos = x0 if isinstance(x0, rg.Point3d) else rg.Point3d(0,0,0)
food_pos = x1 if isinstance(x1, rg.Point3d) else rg.Point3d(50,0,0)
num_ants = int(x2) if x2 else 50
evap_rate = float(x3) if x3 else 0.1
pher_strength = float(x4) if x4 else 1.0
reset = bool(x5) if x5 is not None else False

# Inicializar ou resetar
if not ants or (reset and not previous_reset):
    ants = []
    for _ in range(num_ants):
        pos = rg.Point3d(
            nest_pos.X + random.uniform(-2,2),
            nest_pos.Y + random.uniform(-2,2),
            nest_pos.Z
        )
        ants.append(Ant(pos, nest_pos, food_pos))
    pheromone_trails.clear()

previous_reset = reset

# Atualizar simulação
for ant in ants:
    old_pos = (ant.position.X, ant.position.Y, ant.position.Z)
    ant.update(pheromone_trails)
    
    # Depositar feromônio
    if ant.has_food:
        pheromone_trails[old_pos] += pher_strength * 2
    else:
        pheromone_trails[old_pos] += pher_strength

# Evaporar feromônios
positions_to_remove = []
for pos, strength in pheromone_trails.items():
    pheromone_trails[pos] *= (1.0 - evap_rate)
    if pheromone_trails[pos] < 0.1:
        positions_to_remove.append(pos)

for pos in positions_to_remove:
    del pheromone_trails[pos]

# Preparar outputs
a0 = [ant.position for ant in ants]  # Posições das formigas

a1 = DataTree[object]()  # Pontos das trilhas
a2 = []  # Força das trilhas
path = GH_Path(0)
for pos, strength in pheromone_trails.items():
    point = rg.Point3d(pos[0], pos[1], pos[2])
    a1.Add(point, path)
    a2.append(strength)

a3 = sum(1 for ant in ants if ant.has_food)  # Número de formigas com comida 