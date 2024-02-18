import numpy as np
import time

# Функция приспособленности
def fitness(x, y):
    return np.sin(x**2 + y**2)

# Инициализация популяции
def initialize_population(pop_size):
    return np.random.uniform(-3, 3, (pop_size, 2))

# Селекция
def select(population, scores, num_parents):
    parents = np.zeros((num_parents, population.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(scores == np.max(scores))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = population[max_fitness_idx, :]
        scores[max_fitness_idx] = -99999999 # Это чтобы не выбирать одного и того же родителя дважды
    return parents

# Кроссовер
def crossover(parents, offspring_size):
    offspring = np.zeros(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

# Мутация
def mutation(offspring_crossover, mutation_probability):
    for idx in range(offspring_crossover.shape[0]):
        if np.random.rand() < mutation_probability:
            random_value = np.random.uniform(-0.1, 0.1, 1)
            offspring_crossover[idx, 1] += random_value
    return offspring_crossover

def genetic_algorithm(pop_size, num_generations, num_parents_mating, mutation_probability):
    start_time = time.time()
    population = initialize_population(pop_size)
    for generation in range(num_generations):
        scores = fitness(population[:, 0], population[:, 1])
        parents = select(population, scores, num_parents_mating)
        offspring_crossover = crossover(parents, (pop_size - parents.shape[0], 2))
        offspring_mutation = mutation(offspring_crossover, mutation_probability)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
        max_fitness = np.max(fitness(population[:, 0], population[:, 1]))
        print(f"Поколение {generation}, Лучшее значение: {max_fitness}")
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time} секунд")

# Параметры для эксперимента
parameters = [
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 20, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 30, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 40, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 50, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01},

    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 40, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 60, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 70, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 80, "num_parents_mating": 5, "mutation_probability": 0.01},
    
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 2, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 3, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 4, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 6, "mutation_probability": 0.01},

    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01},
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.05},
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.1},
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.2},
    {"pop_size": 10, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.4},
]

for params in parameters:
    print(f"Эксперимент с параметрами: {params}")
    genetic_algorithm(params["pop_size"], params["num_generations"], params["num_parents_mating"], params["mutation_probability"])
    print("-" * 50)
