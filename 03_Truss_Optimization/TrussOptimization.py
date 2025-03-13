"""Otimização de Treliças usando Algoritmo de Colônia de Formigas (ACO)
    Inputs:
        Nodes: Lista de pontos 3D representando os nós da treliça
        Loads: Lista de cargas aplicadas (força e ponto de aplicação)
        Supports: Lista de apoios (tipo e localização)
        MaterialProperties: Propriedades do material (E, fy)
        CrossSections: Lista de seções transversais disponíveis
    Outputs:
        OptimizedMembers: Lista de barras otimizadas (conectividade)
        MemberSections: Seção transversal designada para cada barra
        TotalWeight: Peso total da estrutura
        Displacements: Deslocamentos nos nós
"""

import math
import random
from collections import defaultdict
import numpy as np

class Section:
    """Classe para representar seções transversais"""
    def __init__(self, name, area, inertia, density=7850):
        self.name = name
        self.area = area
        self.inertia = inertia
        self.density = density

class TrussAnt:
    def __init__(self, nodes, loads, supports):
        self.nodes = nodes
        self.loads = loads
        self.supports = supports
        self.path = []
        self.visited_nodes = set()
        self.total_weight = float('inf')
        self.members = []
        self.member_sections = {}
        self.displacements = None
        self.member_forces = {}
        
    def reset(self):
        """Reseta o estado da formiga para uma nova iteração"""
        self.path = []
        self.visited_nodes = set()
        self.total_weight = float('inf')
        self.members = []
        self.member_sections = {}
        self.displacements = None
        self.member_forces = {}
        
    def select_next_node(self, pheromone_trails, current_node):
        """Seleciona o próximo nó baseado em feromônios e heurística melhorada"""
        unvisited = [n for n in range(len(self.nodes)) if n not in self.visited_nodes]
        if not unvisited:
            return None
            
        # Calcular probabilidades
        probabilities = []
        for node in unvisited:
            trail_key = tuple(sorted([current_node, node]))
            pheromone = pheromone_trails.get(trail_key, 0.1)
            distance = self.calculate_distance(current_node, node)
            
            # 1. Heurística base (inverso da distância)
            distance_factor = 1.0 / (distance + 0.1)
            
            # 2. Fator de direção das cargas
            load_direction_factor = self.calculate_load_direction_factor(current_node, node)
            
            # 3. Fator de triangulação
            triangulation_factor = self.calculate_triangulation_factor(current_node, node)
            
            # 4. Fator de ângulo entre membros
            angle_factor = self.calculate_angle_factor(current_node, node)
            
            # 5. Fator de peso (estimativa baseada na força esperada)
            weight_factor = self.estimate_weight_factor(current_node, node)
            
            # Combinar todos os fatores
            heuristic = (distance_factor * load_direction_factor * 
                        triangulation_factor * angle_factor * weight_factor)
            
            probability = (pheromone ** 1.0) * (heuristic ** 2.0)
            probabilities.append((node, probability))
            
        # Selecionar próximo nó
        total = sum(p[1] for p in probabilities)
        if total <= 0:
            return random.choice(unvisited)
            
        r = random.random() * total
        cumsum = 0
        for node, prob in probabilities:
            cumsum += prob
            if cumsum >= r:
                return node
        return probabilities[-1][0]
        
    def calculate_distance(self, node1_idx, node2_idx):
        """Calcula distância entre dois nós"""
        n1 = self.nodes[node1_idx]
        n2 = self.nodes[node2_idx]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(n1, n2)))
        
    def calculate_load_direction_factor(self, node1, node2):
        """Calcula fator baseado no alinhamento com as cargas"""
        # Vetor do membro
        x1, y1, _ = self.nodes[node1]
        x2, y2, _ = self.nodes[node2]
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return 0.1
            
        # Normalizar vetor do membro
        dx /= length
        dy /= length
        
        # Verificar alinhamento com cargas
        alignment = 0.1  # valor base
        for load in self.loads:
            node = load['node']
            fx, fy, _ = load['force']
            if node == node1 or node == node2:
                # Produto escalar normalizado
                force_mag = math.sqrt(fx*fx + fy*fy)
                if force_mag > 0:
                    alignment += abs(dx*fx/force_mag + dy*fy/force_mag)
                    
        return 1.0 + alignment
        
    def calculate_triangulation_factor(self, node1, node2):
        """Calcula fator que favorece formação de triângulos equiláteros"""
        if len(self.members) < 1:
            return 1.0
            
        best_angle = 0.0
        for member in self.members:
            if node1 in member:
                other_node = member[0] if member[1] == node1 else member[1]
                angle = self.calculate_angle(other_node, node1, node2)
                # Favorece ângulos próximos a 60 graus (triângulos equiláteros)
                best_angle = max(best_angle, math.cos(abs(angle - math.pi/3)))
                
        return 1.0 + best_angle
        
    def calculate_angle_factor(self, node1, node2):
        """Calcula fator que penaliza ângulos muito agudos ou obtusos"""
        if len(self.members) < 1:
            return 1.0
            
        min_angle = math.pi
        for member in self.members:
            if node1 in member:
                other_node = member[0] if member[1] == node1 else member[1]
                angle = self.calculate_angle(other_node, node1, node2)
                min_angle = min(min_angle, abs(angle))
                
        # Penalizar ângulos menores que 30° ou maiores que 150°
        if min_angle < math.pi/6 or min_angle > 5*math.pi/6:
            return 0.5
        return 1.0
        
    def calculate_angle(self, node1, node2, node3):
        """Calcula o ângulo entre três nós"""
        x1, y1, _ = self.nodes[node1]
        x2, y2, _ = self.nodes[node2]
        x3, y3, _ = self.nodes[node3]
        
        # Vetores
        v1x = x1 - x2
        v1y = y1 - y2
        v2x = x3 - x2
        v2y = y3 - y2
        
        # Produto escalar
        dot = v1x*v2x + v1y*v2y
        
        # Magnitudes
        mag1 = math.sqrt(v1x*v1x + v1y*v1y)
        mag2 = math.sqrt(v2x*v2x + v2y*v2y)
        
        # Ângulo
        if mag1 == 0 or mag2 == 0:
            return 0
        cos_angle = dot/(mag1*mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Evitar erros numéricos
        return math.acos(cos_angle)
        
    def estimate_weight_factor(self, node1, node2):
        """Estima o fator de peso baseado na força esperada e comprimento"""
        # Distância do membro
        distance = self.calculate_distance(node1, node2)
        
        # Estimar força baseado na proximidade com cargas
        estimated_force = 0.0
        for load in self.loads:
            load_node = load['node']
            _, fy, _ = load['force']
            if load_node == node1 or load_node == node2:
                estimated_force += abs(fy)
            else:
                # Considerar efeito reduzido para nós próximos
                d_to_load = min(
                    self.calculate_distance(node1, load_node),
                    self.calculate_distance(node2, load_node)
                )
                if d_to_load > 0:
                    estimated_force += abs(fy) / d_to_load
                    
        # Estimar área necessária
        if estimated_force > 0:
            required_area = estimated_force / (355e6 / 1.1)  # Tensão admissível
            # Quanto menor a área necessária, melhor
            return 1.0 / (required_area * distance + 0.1)
        return 1.0
        
    def analyze_truss(self):
        """Análise simplificada da treliça usando método dos deslocamentos"""
        num_nodes = len(self.nodes)
        num_dof = 2 * num_nodes  # 2 graus de liberdade por nó (2D)
        
        # Matriz de rigidez global
        K = np.zeros((num_dof, num_dof))
        
        # Vetor de forças
        F = np.zeros(num_dof)
        
        # Montar matriz de rigidez global
        for i, (node1, node2) in enumerate(self.members):
            # Propriedades do elemento
            section = self.member_sections.get((node1, node2), Section("default", 1e-4, 1e-8))
            E = 210e9  # Módulo de elasticidade do aço
            A = section.area
            
            # Coordenadas dos nós
            x1, y1, _ = self.nodes[node1]
            x2, y2, _ = self.nodes[node2]
            
            # Comprimento e cossenos diretores
            L = self.calculate_distance(node1, node2)
            c = (x2 - x1) / L
            s = (y2 - y1) / L
            
            # Matriz de rigidez do elemento no sistema global
            k = np.array([
                [c*c, c*s, -c*c, -c*s],
                [c*s, s*s, -c*s, -s*s],
                [-c*c, -c*s, c*c, c*s],
                [-c*s, -s*s, c*s, s*s]
            ]) * (E * A / L)
            
            # Índices dos graus de liberdade
            dof = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
            
            # Adicionar à matriz global
            for i1, i_global in enumerate(dof):
                for j1, j_global in enumerate(dof):
                    K[i_global, j_global] += k[i1, j1]
        
        # Aplicar cargas
        for load in self.loads:
            node = load['node']
            fx, fy, _ = load['force']
            F[2*node] += fx
            F[2*node+1] += fy
        
        # Aplicar condições de contorno
        free_dof = []
        for i in range(num_nodes):
            is_support = False
            for support in self.supports:
                if support['node'] == i:
                    is_support = True
                    break
            if not is_support:
                free_dof.extend([2*i, 2*i+1])
        
        # Resolver sistema
        K_red = K[np.ix_(free_dof, free_dof)]
        F_red = F[free_dof]
        
        try:
            d_red = np.linalg.solve(K_red, F_red)
            
            # Recuperar deslocamentos completos
            d = np.zeros(num_dof)
            for i, dof in enumerate(free_dof):
                d[dof] = d_red[i]
            
            self.displacements = d
            
            # Calcular forças nos elementos
            for i, (node1, node2) in enumerate(self.members):
                # Propriedades do elemento
                section = self.member_sections.get((node1, node2), Section("default", 1e-4, 1e-8))
                E = 210e9
                A = section.area
                
                # Coordenadas e deslocamentos
                x1, y1, _ = self.nodes[node1]
                x2, y2, _ = self.nodes[node2]
                d1 = d[2*node1:2*node1+2]
                d2 = d[2*node2:2*node2+2]
                
                # Comprimento e cossenos diretores
                L = self.calculate_distance(node1, node2)
                c = (x2 - x1) / L
                s = (y2 - y1) / L
                
                # Força axial
                force = (E * A / L) * np.array([c, s, -c, -s]) @ np.concatenate([d1, d2])
                self.member_forces[(node1, node2)] = force
            
            return True
            
        except np.linalg.LinAlgError:
            return False
        
    def check_stability(self):
        """Verifica estabilidade da treliça"""
        # Verificação básica de conectividade
        if len(self.members) < len(self.nodes) - 1:
            return False
            
        # Verificar se todos os nós com carga estão conectados
        load_nodes = set(load['node'] for load in self.loads)
        connected_nodes = set()
        for n1, n2 in self.members:
            connected_nodes.add(n1)
            connected_nodes.add(n2)
        if not all(node in connected_nodes for node in load_nodes):
            return False
            
        # Verificação simplificada baseada em análise estrutural
        if not self.analyze_truss():
            return False
            
        # Verificar se os deslocamentos são razoáveis
        if self.displacements is not None:
            if np.max(np.abs(self.displacements)) > 0.2:  # Aumentado para 20cm
                return False
                
        # Verificar forças nos elementos
        if self.member_forces:
            for member, force in self.member_forces.items():
                if abs(force) > 2e6:  # Aumentado para 2000 kN
                    return False
                    
        return True
        
    def calculate_weight(self):
        """Calcula peso total da estrutura"""
        total_weight = 0
        for member in self.members:
            section = self.member_sections.get(member, Section("default", 1e-4, 1e-8))
            length = self.calculate_distance(member[0], member[1])
            total_weight += length * section.area * section.density
        return total_weight

class TrussOptimizer:
    def __init__(self, nodes, loads, supports, material):
        self.nodes = nodes
        self.loads = loads
        self.supports = supports
        self.material = material
        self.best_solution = None
        self.best_weight = float('inf')
        
        # Parâmetros do ACO ajustados para mais nós
        self.num_ants = 30  # Aumentado para 30
        self.evaporation_rate = 0.05  # Reduzido para 0.05
        self.alpha = 1.0  # Importância do feromônio
        self.beta = 3.0   # Aumentada importância da heurística
        
        # Criar seções padrão com mais opções
        areas = [100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500]  # mm²
        self.sections = []
        for i, area in enumerate(areas):
            self.sections.append(Section(f"S{i+1}", area/1e6, (area/1e6)**2/12))
            
        self.pheromone_trails = defaultdict(float)
        
    def initialize(self):
        """Inicializa a otimização"""
        self.ants = [TrussAnt(self.nodes, self.loads, self.supports) 
                    for _ in range(self.num_ants)]
        
    def optimize(self, max_iterations=10000):
        """Executa o processo de otimização"""
        self.initialize()
        
        for iteration in range(max_iterations):
            # Mover formigas
            for ant in self.ants:
                self.construct_solution(ant)
                
            # Atualizar feromônios
            self.update_pheromones()
            
            # Verificar melhor solução
            self.update_best_solution()
            
            print(f"Iteração {iteration+1}: Melhor peso = {self.best_weight:.2f}")
            
        return self.best_solution
        
    def construct_solution(self, ant):
        """Constrói uma solução válida para a treliça"""
        ant.reset()
        
        # Começar dos pontos de apoio
        start_node = self.get_start_node()
        ant.visited_nodes.add(start_node)
        
        # Primeiro, conectar todos os apoios
        support_nodes = [s['node'] for s in self.supports]
        for node in support_nodes[1:]:  # Pula o primeiro nó (já visitado)
            ant.visited_nodes.add(node)
            ant.members.append((start_node, node))
            
        # Depois, conectar os nós com carga
        load_nodes = [l['node'] for l in self.loads]
        for node in load_nodes:
            if node not in ant.visited_nodes:
                # Conectar aos dois nós visitados mais próximos
                closest_nodes = sorted(
                    [n for n in ant.visited_nodes],
                    key=lambda x: ant.calculate_distance(x, node)
                )[:2]
                ant.visited_nodes.add(node)
                for closest in closest_nodes:
                    ant.members.append((closest, node))
                
        # Adicionar membros adicionais para triangulação
        unvisited = set(range(len(self.nodes))) - ant.visited_nodes
        while unvisited:
            node = random.choice(list(unvisited))
            # Conectar aos três nós visitados mais próximos
            visited_nodes = sorted(
                [n for n in ant.visited_nodes],
                key=lambda x: ant.calculate_distance(x, node)
            )[:3]
            
            ant.visited_nodes.add(node)
            for closest in visited_nodes:
                ant.members.append((closest, node))
            unvisited.remove(node)
            
        # Adicionar membros extras para rigidez
        for _ in range(3):  # Aumentado para 3 membros extras
            n1 = random.choice(list(ant.visited_nodes))
            possible_n2 = [n2 for n2 in ant.visited_nodes
                          if n2 != n1 and (n1, n2) not in ant.members and (n2, n1) not in ant.members]
            if possible_n2:
                n2 = min(possible_n2, key=lambda x: ant.calculate_distance(n1, x))
                ant.members.append((n1, n2))
                
        # Designar seções para as barras
        if self.is_valid_truss(ant):
            self.assign_sections(ant)
            
    def get_start_node(self):
        """Seleciona um nó inicial (preferencialmente um apoio)"""
        if self.supports:
            return self.supports[0]['node']
        return 0
        
    def is_valid_truss(self, ant):
        """Verifica se a treliça é válida"""
        # Verificar se todos os nós estão conectados
        if len(ant.visited_nodes) != len(self.nodes):
            return False
            
        # Verificar estabilidade
        return ant.check_stability()
        
    def assign_sections(self, ant):
        """Designa seções transversais para as barras"""
        if not ant.member_forces:
            return
            
        for member, force in ant.member_forces.items():
            # Selecionar seção baseada na força axial
            required_area = abs(force) / (self.material['fy'] / 1.1)  # Com fator de segurança
            
            # Encontrar a menor seção adequada
            selected_section = self.sections[-1]  # Maior seção como padrão
            for section in self.sections:
                if section.area >= required_area:
                    selected_section = section
                    break
                    
            ant.member_sections[member] = selected_section
            
        # Calcular peso total
        ant.total_weight = ant.calculate_weight()
        
    def update_pheromones(self):
        """Atualiza trilhas de feromônio"""
        # Evaporação
        for trail in self.pheromone_trails:
            self.pheromone_trails[trail] *= (1.0 - self.evaporation_rate)
            
        # Depósito de novo feromônio
        for ant in self.ants:
            if ant.total_weight < float('inf'):
                deposit = 1.0 / ant.total_weight
                for member in ant.members:
                    trail_key = tuple(sorted(member))
                    self.pheromone_trails[trail_key] += deposit
                    
    def update_best_solution(self):
        """Atualiza a melhor solução encontrada"""
        for ant in self.ants:
            if ant.total_weight < self.best_weight:
                self.best_weight = ant.total_weight
                self.best_solution = {
                    'members': ant.members[:],
                    'sections': ant.member_sections.copy(),
                    'weight': ant.total_weight,
                    'forces': ant.member_forces.copy() if ant.member_forces else None,
                    'displacements': ant.displacements.copy() if ant.displacements is not None else None
                }

def plot_truss(nodes, members, forces=None, loads=None, supports=None):
    """Plota a treliça com forças nos elementos (se fornecidas)"""
    try:
        import matplotlib.pyplot as plt
        
        # Criar figura com tamanho específico
        plt.figure(figsize=(12, 8))
        
        # Plotar nós
        x = [node[0] for node in nodes]
        y = [node[1] for node in nodes]
        plt.scatter(x, y, c='black', s=100, zorder=5, label='Nós')
        
        # Plotar elementos
        max_force = max(abs(f) for f in forces.values()) if forces else 1.0
        for i, (n1, n2) in enumerate(members):
            x_coords = [nodes[n1][0], nodes[n2][0]]
            y_coords = [nodes[n1][1], nodes[n2][1]]
            
            if forces is not None and (n1, n2) in forces:
                force = forces[(n1, n2)]
                if abs(force) < 1.0:  # Força próxima a zero
                    color = 'gray'
                    label = 'Elementos de estabilização' if i == 0 else ""
                elif force > 0:
                    color = 'red'
                    label = 'Elementos em tração' if i == 0 else ""
                else:
                    color = 'blue'
                    label = 'Elementos em compressão' if i == 0 else ""
                width = max(1, min(4, 3 * abs(force) / max_force))
            else:
                color = 'black'
                width = 1
                label = 'Elementos' if i == 0 else ""
                
            plt.plot(x_coords, y_coords, color=color, linewidth=width, label=label, zorder=2)
            
        # Adicionar números dos nós
        for i, (x, y, _) in enumerate(nodes):
            plt.annotate(f'N{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold', zorder=6)
            
        # Adicionar cargas
        if loads:
            for load in loads:
                node = load['node']
                fx, fy, _ = load['force']
                x, y = nodes[node][0], nodes[node][1]
                if fy < 0:  # Carga vertical para baixo
                    plt.arrow(x, y+0.5, 0, -0.3, head_width=0.2, head_length=0.2, 
                             fc='green', ec='green', zorder=4, label='Cargas' if node == loads[0]['node'] else "")
                
        # Adicionar apoios
        if supports:
            for support in supports:
                node = support['node']
                x, y = nodes[node][0], nodes[node][1]
                if support['type'] == 'fixed':
                    plt.plot([x-0.3, x+0.3], [y-0.3, y-0.3], 'k-', linewidth=2, zorder=3, 
                            label='Apoio fixo' if node == supports[0]['node'] else "")
                    plt.plot([x-0.3, x+0.3], [y-0.3, y-0.1], 'k-', linewidth=1, zorder=3)
                else:  # roller
                    circle = plt.Circle((x, y-0.2), 0.1, fc='white', ec='black', zorder=3,
                                      label='Apoio móvel' if node == supports[-1]['node'] else "")
                    plt.gca().add_patch(circle)
                
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title('Treliça Otimizada\nPeso Total: {:.2f} kg'.format(solution['weight']))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib não está instalado. Não é possível gerar o gráfico.")

# Exemplo de uso:
if __name__ == "__main__":
    # Definir dados de entrada (versão expandida com 10 nós)
    nodes = [
        (0, 0, 0),      # Nó 0 - Apoio fixo
        (3, 0, 0),      # Nó 1
        (6, 0, 0),      # Nó 2
        (9, 0, 0),      # Nó 3 - Apoio móvel
        (1.5, 2, 0),    # Nó 4
        (4.5, 2.5, 0),  # Nó 5
        (7.5, 2, 0),    # Nó 6
        (3, 4, 0),      # Nó 7
        (6, 4.5, 0),    # Nó 8
        (4.5, 6, 0),    # Nó 9
    ]
    
    loads = [
        {'node': 5, 'force': (0, -10000, 0)},   # 10kN vertical no nó 5
        {'node': 6, 'force': (0, -15000, 0)},   # 15kN vertical no nó 6
        {'node': 8, 'force': (0, -12000, 0)},   # 12kN vertical no nó 8
        {'node': 9, 'force': (0, -8000, 0)},    # 8kN vertical no nó 9
    ]
    
    supports = [
        {'node': 0, 'type': 'fixed'},
        {'node': 3, 'type': 'roller'}
    ]
    
    material = {
        'E': 210e9,    # Módulo de elasticidade (Pa)
        'fy': 355e6,   # Tensão de escoamento (Pa)
        'density': 7850 # Densidade (kg/m³)
    }
    
    # Criar e executar otimizador
    optimizer = TrussOptimizer(nodes, loads, supports, material)
    solution = optimizer.optimize(max_iterations=10000)
    
    if solution:
        print("\nMelhor solução encontrada (ACO):")
        print(f"Peso total: {solution['weight']:.2f} kg")
        print("Membros:", solution['members'])
        
        if solution['forces'] is not None:
            print("\nForças nos elementos (N):")
            for member, force in solution['forces'].items():
                print(f"Barra {member}: {force:.2f}")
                
        # Plotar a treliça
        plot_truss(nodes, solution['members'], solution['forces'], loads, supports)
    else:
        print("\nNão foi possível encontrar uma solução válida.") 