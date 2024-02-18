import numpy as np
import time
import matplotlib.pyplot as plt
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
            offspring_crossover[idx, np.random.choice([0, 1])] += random_value

    return offspring_crossover

def genetic_algorithm(pop_size, num_generations, num_parents_mating, mutation_probability):
    population = initialize_population(pop_size)
    max_fitness = None
    for generation in range(num_generations):
        scores = fitness(population[:, 0], population[:, 1])
        parents = select(population, scores, num_parents_mating)
        offspring_crossover = crossover(parents, (pop_size - parents.shape[0], 2))
        offspring_mutation = mutation(offspring_crossover, mutation_probability)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
        max_fitness = np.max(fitness(population[:, 0], population[:, 1]))
    return max_fitness


def avg(data):
    data = np.array(data)  # Преобразование в массив NumPy
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Вычисляем IQR
    IQR = Q3 - Q1

    # Определяем границы для выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Фильтруем выбросы
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return np.mean(filtered_data)

# Параметры для эксперимента по изменению размера популяции
parameters_pop_size = [
    {"pop_size": i, "num_generations": 20, "num_parents_mating": 5, "mutation_probability": 0.01} for i in range(5,50,2)
]

# Параметры для эксперимента по изменению Количество поколений 
parameters_num_generations = [
    {"pop_size": 50, "num_generations": i, "num_parents_mating": 5, "mutation_probability": 0.01} for i in range(2,50,2)
]

# Параметры для эксперимента по изменению количество родителей 
parameters_num_parents_mating = [
    {"pop_size": 50, "num_generations": 50, "num_parents_mating": i, "mutation_probability": 0.01} for i in range(2,50,2)
]

# Параметры для эксперимента по изменению шанс мутации 
parameters_mutation_probability= [
    {"pop_size": 50, "num_generations": 50, "num_parents_mating": 5, "mutation_probability": i/100} for i in range(1,100,5)
]
names={
    'pop_size': 'размер популяции',
    'num_generations': 'количество поколений',
    'num_parents_mating': 'количество родителей',
    'mutation_probability': 'вероятность мутации'
}
name_parameters=['pop_size','num_generations','num_parents_mating','mutation_probability']
for name in name_parameters:
    if name =='pop_size':
        parameters=parameters_pop_size
    elif name=='num_generations':
        parameters=parameters_num_generations
    elif name=='num_parents_mating':
        parameters=parameters_num_parents_mating
    elif name=='mutation_probability':
        parameters=parameters_mutation_probability

    execution_times = []
    max_fitnesses = []
    pop_sizes = [params[f'{name}'] for params in parameters]

    for params in parameters:
        arr_time=[]
        arr_result=[]
        for i in range(100):
            start_time = time.time()
            result = genetic_algorithm(params["pop_size"], params["num_generations"], params["num_parents_mating"], params["mutation_probability"])
            end_time = time.time()
            arr_time.append(end_time-start_time)
            arr_result.append(result)
        
        execution_times.append(avg(arr_time))
        max_fitnesses.append(avg(arr_result))

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Создаем фигуру и оси для двух подграфиков

# График времени выполнения
    axs[0].plot(pop_sizes, execution_times, marker='o', linestyle='-', color='b')
    axs[0].set_title(f'Зависимость времени выполнения от {names[name]}')
    axs[0].set_xlabel(f'{names[name]}')
    axs[0].set_ylabel('Время выполнения, секунды')
    axs[0].grid(True)

# График максимальной приспособленности
    axs[1].plot(pop_sizes, max_fitnesses, marker='x', linestyle='-', color='r')
    axs[1].set_title(f'Зависимость максимальной приспособленности от {names[name]}')
    axs[1].set_xlabel(f'{names[name]}')
    axs[1].set_ylabel('Максимальная приспособленность')
    axs[1].grid(True)

    plt.tight_layout()  # Автоматически корректирует подграфики, чтобы они не перекрывались
    plt.show()
    plt.savefig(f'{names[name]}.png')
