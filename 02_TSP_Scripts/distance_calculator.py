import math

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

# Calcular distância na ordem original
print(f"\nDistância total (ordem original): {calculate_total_distance(test_cities):.2f}")

# Algumas ordens diferentes para teste
orders = [
    [0, 1, 2, 3, 4, 5, 6, 7],  # ordem original
    [0, 7, 6, 5, 4, 3, 2, 1],  # ordem reversa
    [0, 2, 4, 6, 1, 3, 5, 7],  # alternando índices
]

print("\nTestando diferentes ordens:")
for i, order in enumerate(orders):
    path = [test_cities[idx] for idx in order]  # Corrigido: usando idx ao invés de i
    print(f"Ordem teste {i+1}: {order}")
    print(f"Distância total: {calculate_total_distance(path):.2f}\n") 