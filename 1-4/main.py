'''
Давайте разработаем задачу, где генетический алгоритм (ГА) может быть использован для оптимизации, и напишем соответствующий код. Задача будет заключаться в оптимизации портфеля инвестиций, чтобы максимизировать ожидаемый доход при заданном уровне риска. Это классическая задача оптимизации, где генетические алгоритмы могут показать хорошие результаты, особенно когда пространство поиска велико и нелинейно.

Задача
Имеется набор инвестиционных активов (акции, облигации и т.д.), каждый из которых имеет свой ожидаемый доход и уровень риска. Необходимо выбрать состав портфеля таким образом, чтобы максимизировать ожидаемый доход при заданном максимально допустимом уровне риска.

Параметры
Ожидаемый доход актива определяется как процентное увеличение его стоимости.
Уровень риска актива может быть определен как стандартное отклонение ожидаемого дохода.
Максимально допустимый уровень риска портфеля задан заранее.
Генетический алгоритм
Популяция: наборы портфелей с различным составом активов.
Фитнес-функция: соотношение ожидаемого дохода к риску портфеля, где портфели с доходностью выше и риском ниже получают более высокий фитнес.
Операторы генетического алгоритма: кроссовер (смешивание двух портфелей для создания нового) и мутация (случайное изменение состава портфеля).
Реализация
Сначала определим базовую структуру данных для актива и функции для оценки портфеля. Затем реализуем основные компоненты генетического алгоритма: создание начальной популяции, оценку фитнеса, селекцию, кроссовер и мутацию.
'''
import numpy as np
import random

# Определение актива
class Asset:
    def __init__(self, name, return_rate, risk):
        self.name = name
        self.return_rate = return_rate  # Ожидаемый доход
        self.risk = risk  # Уровень риска

# Создание начальной популяции
def create_initial_population(size, assets, portfolio_size):
    population = []
    for _ in range(size):
        portfolio = random.sample(assets, portfolio_size)
        population.append(portfolio)
    return population

# Функция фитнеса
def fitness(portfolio, max_risk):
    total_return = sum(asset.return_rate for asset in portfolio)
    total_risk = np.sqrt(sum(asset.risk ** 2 for asset in portfolio))
    if total_risk > max_risk:
        return 0  # Портфель слишком рискованный
    return total_return / total_risk  # Чем выше значение, тем лучше

# Селекция
def select(population, fitness_scores, num_parents):
    parents = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in parents[:num_parents]]

# Кроссовер
def crossover(parent1, parent2):
    child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
    return child

# Мутация
def mutate(portfolio, assets, mutation_rate):
    if random.random() < mutation_rate:
        replace_index = random.randint(0, len(portfolio) - 1)
        portfolio[replace_index] = random.choice(assets)
    return portfolio

# Главный цикл алгоритма
def genetic_algorithm(assets, population_size, portfolio_size, max_risk, generations, mutation_rate):
    population = create_initial_population(population_size, assets, portfolio_size)
    for _ in range(generations):
        fitness_scores = [fitness(portfolio, max_risk) for portfolio in population]
        parents = select(population, fitness_scores, population_size // 2)
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, assets, mutation_rate)
            next_generation.append(child)
        population = next_generation
    return population

# Пример активов
assets = [Asset(f"Asset {i}", random.uniform(0.05, 0.2), random.uniform(0.01, 0.05)) for i in range(10)]

# Жадный метод оптимизации
def greedy_optimization(assets, max_risk):
    sorted_assets = sorted(assets, key=lambda x: x.return_rate / x.risk, reverse=True)
    portfolio = []
    total_risk = 0
    for asset in sorted_assets:
        if total_risk + asset.risk <= max_risk:
            portfolio.append(asset)
            total_risk += asset.risk
        else:
            break
    return portfolio

# Измерение времени выполнения и качества решения
import time

# Генетический алгоритм
start_time_ga = time.time()
best_portfolios_ga = genetic_algorithm(assets, 100, 5, 0.1, 50, 0.2)
execution_time_ga = time.time() - start_time_ga
best_fitness_ga = max(fitness(portfolio, 0.1) for portfolio in best_portfolios_ga)

# Жадный алгоритм
start_time_greedy = time.time()
best_portfolio_greedy = greedy_optimization(assets, 0.1)
execution_time_greedy = time.time() - start_time_greedy
best_fitness_greedy = fitness(best_portfolio_greedy, 0.1)

# Вывод результатов
print(f"GA Execution Time: {execution_time_ga:.4f}s, Best Fitness: {best_fitness_ga}")
print(f"Greedy Execution Time: {execution_time_greedy:.4f}s, Best Fitness: {best_fitness_greedy}")

