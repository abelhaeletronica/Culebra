import Rhino.Geometry as rg
import ghpythonlib.components as ghc
import rhinoscriptsyntax as rs
import System.Drawing.Color as Color
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import math
import random
from System import Object

# Importações do Culebra
import CulebraData
from CulebraData.Objects import Creeper
from CulebraData.Data_Structures import *
from Culebra_GH.Data_Structures import *
from Culebra_GH.Behaviors import *
from ACO_Behavior import AntBehavior

class ACOAnt:
    def __init__(self, pos, nest_pos, food_pos):
        # Criar um Creeper do Culebra
        initial_speed = rg.Vector3d(random.uniform(-1,1), random.uniform(-1,1), 0)
        self.creeper = Creeper(pos, initial_speed, True, False)  # False para 2D
        
        # Configurar comportamento
        self.behavior = AntBehavior(self.creeper, nest_pos, food_pos)
        self.path = []
        self.pheromone_strength = 1.0
        
    def update(self, pheromone_trails):
        # Atualizar comportamento
        self.behavior.update(pheromone_trails)
        
        # Atualizar caminho
        current_pos = self.creeper.attributes.GetLocation()
        self.path.append(current_pos)
        
        # Limitar tamanho do caminho para economizar memória
        if len(self.path) > 1000:
            self.path = self.path[-1000:]
            
    @property
    def position(self):
        return self.creeper.attributes.GetLocation()
    
    @property
    def has_food(self):
        return self.behavior.has_food

class ACOEngine:
    def __init__(self):
        self.ants = []
        self.pheromone_trails = {}
        self.evaporation_rate = 0.1
        self.deposit_amount = 1.0
        
    def initialize_ants(self, num_ants, nest_pos, food_pos):
        self.ants = []
        for _ in range(num_ants):
            ant = ACOAnt(nest_pos.Clone(), nest_pos, food_pos)
            self.ants.append(ant)
    
    def update(self):
        # Atualizar formigas
        for ant in self.ants:
            ant.update(self.pheromone_trails)
        
        # Atualizar feromônios
        self.update_pheromones()
    
    def update_pheromones(self):
        # Evaporação
        for pos in self.pheromone_trails:
            self.pheromone_trails[pos] *= (1 - self.evaporation_rate)
            
            # Remover trilhas muito fracas
            if self.pheromone_trails[pos] < 0.01:
                del self.pheromone_trails[pos]
        
        # Depósito de novos feromônios
        for ant in self.ants:
            if len(ant.path) > 0:
                # Formigas com comida deixam trilha mais forte
                strength = self.deposit_amount * (2.0 if ant.has_food else 1.0)
                
                # Força do feromônio é inversamente proporcional ao tamanho do caminho
                strength = strength / len(ant.path)
                
                for pos in ant.path:
                    key = (pos.X, pos.Y, pos.Z)
                    if key not in self.pheromone_trails:
                        self.pheromone_trails[key] = 0
                    self.pheromone_trails[key] += strength

class ACOComponent(gh.GH_Component):
    def __init__(self):
        super(ACOComponent, self).__init__(
            "ACO Engine", "ACO",
            "Ant Colony Optimization usando Culebra",
            "Culebra_GH", "ACO")
        self.engine = ACOEngine()
        
    def RegisterInputParams(self, pManager):
        pManager.AddPointParameter("Nest", "N", "Posição do formigueiro", gh.GH_ParamAccess.item)
        pManager.AddPointParameter("Food", "F", "Posição da comida", gh.GH_ParamAccess.item)
        pManager.AddIntegerParameter("NumAnts", "A", "Número de formigas", gh.GH_ParamAccess.item, 50)
        pManager.AddNumberParameter("EvaporationRate", "E", "Taxa de evaporação dos feromônios", gh.GH_ParamAccess.item, 0.1)
        pManager.AddNumberParameter("PheromoneStrength", "P", "Força dos feromônios", gh.GH_ParamAccess.item, 1.0)
        pManager.AddBooleanParameter("Reset", "R", "Reiniciar simulação", gh.GH_ParamAccess.item, False)
        
    def RegisterOutputParams(self, pManager):
        pManager.AddPointParameter("AntPositions", "AP", "Posições atuais das formigas", gh.GH_ParamAccess.list)
        pManager.AddPointParameter("Trails", "T", "Trilhas de feromônios", gh.GH_ParamAccess.tree)
        pManager.AddNumberParameter("TrailStrengths", "TS", "Força das trilhas", gh.GH_ParamAccess.list)
        pManager.AddIntegerParameter("AntsWithFood", "AF", "Número de formigas carregando comida", gh.GH_ParamAccess.item)
        
    def SolveInstance(self, DA):
        # Obter inputs
        nest_pos = None
        food_pos = None
        num_ants = 50
        evap_rate = 0.1
        pher_strength = 1.0
        reset = False
        
        if not DA.GetData(0, nest_pos): return
        if not DA.GetData(1, food_pos): return
        DA.GetData(2, num_ants)
        DA.GetData(3, evap_rate)
        DA.GetData(4, pher_strength)
        DA.GetData(5, reset)
        
        # Atualizar parâmetros do engine
        self.engine.evaporation_rate = evap_rate
        self.engine.deposit_amount = pher_strength
        
        # Reiniciar se solicitado
        if reset or len(self.engine.ants) == 0:
            self.engine.initialize_ants(num_ants, nest_pos, food_pos)
        
        # Atualizar simulação
        self.engine.update()
        
        # Preparar outputs
        ant_positions = [ant.position for ant in self.engine.ants]
        trail_points = DataTree[Object]()
        trail_strengths = []
        
        for pos, strength in self.engine.pheromone_trails.items():
            point = rg.Point3d(pos[0], pos[1], pos[2])
            trail_points.Add(point, GH_Path(0))
            trail_strengths.append(strength)
        
        # Contar formigas com comida
        ants_with_food = sum(1 for ant in self.engine.ants if ant.has_food)
        
        # Definir outputs
        DA.SetDataList(0, ant_positions)
        DA.SetDataTree(1, trail_points)
        DA.SetDataList(2, trail_strengths)
        DA.SetData(3, ants_with_food)

if __name__ == "__main__":
    component = ACOComponent() 