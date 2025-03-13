import Rhino.Geometry as rg
import ghpythonlib.components as ghc
import rhinoscriptsyntax as rs
import System.Drawing.Color as Color
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import math
import random
from System import Object
from collections import defaultdict

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
    def __init__(self, nest_pos, food_pos, num_ants=50):
        self.nest_pos = nest_pos
        self.food_pos = food_pos
        self.num_ants = num_ants
        self.ants = []
        self.pheromone_trails = defaultdict(float)
        self.initialize_ants()
        
    def initialize_ants(self):
        self.ants = []
        for _ in range(self.num_ants):
            ant_pos = rg.Point3d(
                self.nest_pos.X + random.uniform(-2,2),
                self.nest_pos.Y + random.uniform(-2,2),
                self.nest_pos.Z
            )
            self.ants.append(AntBehavior(ant_pos, self.nest_pos, self.food_pos))
    
    def update(self, evaporation_rate=0.1, pheromone_strength=1.0):
        # Atualizar formigas
        for ant in self.ants:
            old_pos = (ant.position.X, ant.position.Y, ant.position.Z)
            ant.update(self.pheromone_trails)
            
            # Depositar feromônio
            if ant.has_food:
                self.pheromone_trails[old_pos] += pheromone_strength * 2  # Trilha mais forte quando carrega comida
            else:
                self.pheromone_trails[old_pos] += pheromone_strength
        
        # Evaporar feromônios
        positions_to_remove = []
        for pos, strength in self.pheromone_trails.items():
            self.pheromone_trails[pos] *= (1.0 - evaporation_rate)
            if self.pheromone_trails[pos] < 0.1:  # Remover trilhas fracas
                positions_to_remove.append(pos)
        
        for pos in positions_to_remove:
            del self.pheromone_trails[pos]

class ACOComponent(gh.GH_Component):
    def __init__(self):
        super(ACOComponent, self).__init__(
            "ACO Engine", "ACO",
            "Ant Colony Optimization usando Culebra",
            "Culebra_GH", "ACO")
        self.engine = None
        self.previous_reset = False
        
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
        
        # Inicializar ou resetar engine
        if not self.engine or reset and not self.previous_reset:
            self.engine = ACOEngine(nest_pos, food_pos, num_ants)
        self.previous_reset = reset
        
        # Atualizar simulação
        self.engine.update(evap_rate, pher_strength)
        
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