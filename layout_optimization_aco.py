import numpy as np
import matplotlib.pyplot as plt
import random

class LayoutACO:
    def __init__(self, num_elements, space_size, num_ants, num_iterations):
        self.num_elements = num_elements
        self.space_size = space_size
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.pheromone = np.ones((num_elements, num_elements))
        self.best_layout = None
        self.best_distance = float('inf')

    def initialize_elements(self):
        """Inicializa a posição dos elementos aleatoriamente no espaço."""
        return np.random.rand(self.num_elements, 2) * self.space_size

    def calculate_distance(self, layout):
        """Calcula a distância total percorrida entre os elementos."""
        distance = 0
        for i in range(self.num_elements - 1):
            for j in range(i + 1, self.num_elements):
                distance += np.linalg.norm(layout[i] - layout[j])
        return distance

    def update_pheromones(self, layouts, distances):
        """Atualiza as trilhas de feromônio com base nas distâncias calculadas."""
        self.pheromone *= 0.9  # Evaporação
        for layout, distance in zip(layouts, distances):
            for i in range(self.num_elements - 1):
                for j in range(i + 1, self.num_elements):
                    self.pheromone[i, j] += 1.0 / distance

    def optimize(self):
        """Executa o processo de otimização do layout."""
        for iteration in range(self.num_iterations):
            layouts = [self.initialize_elements() for _ in range(self.num_ants)]
            distances = [self.calculate_distance(layout) for layout in layouts]
            self.update_pheromones(layouts, distances)

            # Encontrar o melhor layout
            min_distance = min(distances)
            if min_distance < self.best_distance:
                self.best_distance = min_distance
                self.best_layout = layouts[distances.index(min_distance)]

            print(f"Iteração {iteration + 1}: Melhor distância = {self.best_distance:.2f}")

    def plot_layout(self):
        """Plota o layout otimizado."""
        if self.best_layout is not None:
            plt.scatter(self.best_layout[:, 0], self.best_layout[:, 1], c='blue')
            for i in range(self.num_elements):
                plt.annotate(f'E{i}', (self.best_layout[i, 0], self.best_layout[i, 1]))
            plt.title(f'Layout Otimizado - Distância Total: {self.best_distance:.2f}')
            plt.xlim(0, self.space_size)
            plt.ylim(0, self.space_size)
            plt.grid(True)
            plt.show()

# Exemplo de uso
if __name__ == "__main__":
    num_elements = 10
    space_size = 100
    num_ants = 20
    num_iterations = 100

    aco = LayoutACO(num_elements, space_size, num_ants, num_iterations)
    aco.optimize()
    aco.plot_layout() 