import Rhino.Geometry as rg
import CulebraData
from CulebraData.Objects import Creeper
from CulebraData.Behavior import Controller
from Culebra_GH.Data_Structures import *

class AntBehavior:
    def __init__(self, ant_creeper, nest_pos, food_pos):
        self.creeper = ant_creeper
        self.nest_pos = nest_pos
        self.food_pos = food_pos
        self.has_food = False
        self.attraction_strength = 1.0
        self.pheromone_strength = 1.0
        self.wander_strength = 0.5
        
    def update(self, pheromone_trails):
        current_pos = self.creeper.attributes.GetLocation()
        
        # Calcular forças de atração
        if self.has_food:
            target_pos = self.nest_pos
        else:
            target_pos = self.food_pos
            
        # Força de atração para o objetivo (comida ou ninho)
        attraction_vector = target_pos - current_pos
        attraction_vector.Unitize()
        attraction_vector *= self.attraction_strength
        
        # Força dos feromônios
        pheromone_vector = self.calculate_pheromone_force(current_pos, pheromone_trails)
        
        # Comportamento de vagar (wandering)
        wander_vector = self.calculate_wander_force()
        
        # Combinar forças
        total_force = attraction_vector + (pheromone_vector * self.pheromone_strength) + (wander_vector * self.wander_strength)
        
        # Aplicar força resultante usando o Culebra
        self.creeper.behaviors.ApplyForce(total_force)
        
        # Verificar se chegou ao objetivo
        if not self.has_food:
            if current_pos.DistanceTo(self.food_pos) < 1.0:
                self.has_food = True
        else:
            if current_pos.DistanceTo(self.nest_pos) < 1.0:
                self.has_food = False
                
    def calculate_pheromone_force(self, current_pos, pheromone_trails):
        # Calcular força baseada nas trilhas de feromônios próximas
        pheromone_vector = rg.Vector3d(0,0,0)
        search_radius = 5.0  # Raio de busca para feromônios
        
        for pos, strength in pheromone_trails.items():
            point = rg.Point3d(pos[0], pos[1], pos[2])
            if current_pos.DistanceTo(point) < search_radius:
                direction = point - current_pos
                direction.Unitize()
                pheromone_vector += direction * strength
                
        if pheromone_vector.Length > 0:
            pheromone_vector.Unitize()
            
        return pheromone_vector
        
    def calculate_wander_force(self):
        # Usar o comportamento de vagar do Culebra
        wander_force = rg.Vector3d(0,0,0)
        
        # Parâmetros do wandering
        change = 0.3
        wander_radius = 2.0
        wander_distance = 4.0
        
        # Usar o comportamento de wandering do Culebra
        self.creeper.behaviors.Wander2D(change, wander_radius, wander_distance)
        
        # Obter a força resultante
        wander_force = self.creeper.attributes.GetSpeed()
        wander_force.Unitize()
        
        return wander_force 