"""Otimização de Treliças usando Força Bruta (versão simplificada)"""

import itertools
import numpy as np
from TrussOptimization import Section, TrussAnt

class TrussBruteForce:
    def __init__(self, nodes, loads, supports, material_properties):
        self.nodes = nodes[:6]  # Limitado aos primeiros 6 nós
        self.loads = [load for load in loads if load['node'] < 6]
        self.supports = supports
        self.material = material_properties
        self.best_solution = None
        self.best_weight = float('inf')
        
    def optimize(self, max_combinations=10000):
        """Testa todas as combinações possíveis de barras"""
        possible_members = list(itertools.combinations(range(len(self.nodes)), 2))
        num_members = len(possible_members)
        
        # Criar objeto para análise
        ant = TrussAnt(self.nodes, self.loads, self.supports)
        
        count = 0
        print(f"Total de combinações possíveis: {2**num_members}")
        print(f"Testando as primeiras {max_combinations} combinações...")
        
        # Testar combinações
        for num_bars in range(len(self.nodes), len(possible_members) + 1):
            for members in itertools.combinations(possible_members, num_bars):
                if count >= max_combinations:
                    break
                    
                count += 1
                if count % 100 == 0:
                    print(f"Testando combinação {count}...")
                
                # Resetar objeto de análise
                ant.reset()
                ant.members = list(members)
                ant.visited_nodes = set(n for member in members for n in member)
                
                # Verificar se é uma estrutura válida
                if ant.check_stability():
                    # Designar seções
                    if ant.member_forces:
                        for member, force in ant.member_forces.items():
                            required_area = abs(force) / (self.material['fy'] / 1.1)
                            section = Section(f"S{member}", required_area, required_area**2/12)
                            ant.member_sections[member] = section
                        
                        # Calcular peso
                        weight = ant.calculate_weight()
                        
                        # Atualizar melhor solução
                        if weight < self.best_weight:
                            self.best_weight = weight
                            self.best_solution = {
                                'members': ant.members[:],
                                'sections': ant.member_sections.copy(),
                                'weight': weight,
                                'forces': ant.member_forces.copy(),
                                'displacements': ant.displacements.copy() if ant.displacements is not None else None
                            }
                            print(f"\nNova melhor solução encontrada! Peso: {weight:.2f} kg")
        
        return self.best_solution

if __name__ == "__main__":
    # Definir dados de entrada (versão simplificada com 6 nós)
    nodes = [
        (0, 0, 0),     # Nó 0 - Apoio fixo
        (3, 0, 0),     # Nó 1
        (7, 0, 0),     # Nó 2
        (1.5, 2, 0),   # Nó 3
        (4.5, 3, 0),   # Nó 4
        (6, 2.5, 0),   # Nó 5
    ]
    
    loads = [
        {'node': 4, 'force': (0, -15000, 0)},   # 15kN vertical no nó 4
        {'node': 5, 'force': (0, -10000, 0)},   # 10kN vertical no nó 5
    ]
    
    supports = [
        {'node': 0, 'type': 'fixed'},
        {'node': 2, 'type': 'roller'}
    ]
    
    material = {
        'E': 210e9,    # Módulo de elasticidade (Pa)
        'fy': 355e6,   # Tensão de escoamento (Pa)
        'density': 7850 # Densidade (kg/m³)
    }
    
    # Criar e executar otimizador
    optimizer = TrussBruteForce(nodes, loads, supports, material)
    solution = optimizer.optimize(max_combinations=10000)
    
    if solution:
        print("\nMelhor solução encontrada (Força Bruta):")
        print(f"Peso total: {solution['weight']:.2f} kg")
        print("Membros:", solution['members'])
        
        if solution['forces'] is not None:
            print("\nForças nos elementos (N):")
            for member, force in solution['forces'].items():
                print(f"Barra {member}: {force:.2f}")
                
        # Importar função de plotagem
        from TrussOptimization import plot_truss
        plot_truss(nodes, solution['members'], solution['forces'], loads, supports)
    else:
        print("\nNão foi possível encontrar uma solução válida.") 