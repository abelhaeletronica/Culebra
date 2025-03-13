# Projetos de Otimização

Este repositório contém três projetos principais relacionados a otimização:

## 1. Culebra (`01_Culebra/`)
Plugin para Grasshopper que implementa diversos algoritmos de otimização, incluindo:
- Otimização por Colônia de Formigas (ACO)
- Algoritmos Genéticos
- Otimização por Enxame de Partículas (PSO)

## 2. Scripts TSP (`02_TSP_Scripts/`)
Implementações do Problema do Caixeiro Viajante (TSP) usando diferentes abordagens:
- Algoritmo de Força Bruta
- Algoritmo Genético
- Otimização por Colônia de Formigas
Para mais detalhes, consulte o [README do TSP](02_TSP_Scripts/README.md).

## 3. Otimização de Treliças (`03_Truss_Optimization/`)
Implementação de otimização estrutural de treliças usando ACO:
- Otimização topológica
- Análise estrutural
- Dimensionamento de seções
Para mais detalhes, consulte o [README da Otimização de Treliças](03_Truss_Optimization/README.md).

## Estrutura do Repositório
```
.
├── 01_Culebra/           # Plugin Culebra para Grasshopper
├── 02_TSP_Scripts/       # Scripts do Problema do Caixeiro Viajante
└── 03_Truss_Optimization/# Otimização de Treliças usando ACO
```

## Requisitos
- Python 3.8+
- NumPy
- Matplotlib
- Grasshopper (para o plugin Culebra) 