#%%
import numpy as np
from copy import deepcopy
#%%
weights = np.random.uniform(0,5,1000)
#%%
values = np.random.randint(0, 100, 1000)
#%%
def knapsack_cost(v, max_weight):
    total_weight = np.sum(weights[v.chromosome])
    total_value = np.sum(values[v.chromosome])

    if total_weight <= max_weight:
        return -total_value  # Maximize value
    else:
        # Increase penalty based on weight excess
        penalty_factor = 10 * (total_weight - max_weight)
        return -total_value - penalty_factor
#%%
class treasure_problem:
  def __init__(self):
     self.number_of_genes = 1000
     self.max_weight = 60 #Store max_weight as an attribute
     #Assign a lambda function that calls knapsack_cost with both arguments
     self.cost_function = lambda v: knapsack_cost(v, self.max_weight)
#%%
class treasure_individual:
  # This class defines the individual for the genetic algorithm. We hope to find the individual which solves our problem
  chromosome = None

  def __init__(self, prob):    #  This is the constructor for the individual and as such needs the problem to mould the individuals to it
    #Create a random individual.
    self.chromosome =np.random.choice([True,False],prob.number_of_genes,p=[0.01, 0.99])
    self.cost = prob.cost_function(self)


  def crossover(self, other_parent):
        # Choose two random crossover points
        chromosome_length = len(self.chromosome)  # Store length to avoid repeated calls
        crossover_point1 = np.random.randint(0, chromosome_length - 1)  # Ensure crossover_point1 is not the last index
        crossover_point2 = np.random.randint(crossover_point1 + 1, chromosome_length)

        # Create the offspring
        offspring1 = deepcopy(self)
        offspring2 = deepcopy(other_parent)

        # Perform the crossover
        offspring1.chromosome[crossover_point1:crossover_point2] = other_parent.chromosome[crossover_point1:crossover_point2]
        offspring2.chromosome[crossover_point1:crossover_point2] = self.chromosome[crossover_point1:crossover_point2]

        return offspring1, offspring2

  def mutate(self, mutation_rate, weights):
    mutated_chromosome = deepcopy(self.chromosome)
    for i in range(len(mutated_chromosome)):
        if mutated_chromosome[i] and np.random.rand() < mutation_rate:  # Only mutate True values
            # Define neighborhood indices (e.g., one position to the left and right)
            neighbors = [i - 1, i + 1]
            # Handle boundary conditions
            neighbors = [n for n in neighbors if 0 <= n < len(mutated_chromosome)]

            # Find lighter neighbors that are False
            lighter_neighbors = [n for n in neighbors
                                  if not mutated_chromosome[n] and weights[n] < weights[i]]

            if lighter_neighbors:  # If lighter neighbors exist
                # Randomly select a lighter neighbor for the swap
                swap_index = np.random.choice(lighter_neighbors)

                # Perform the swap
                mutated_chromosome[i] = False
                mutated_chromosome[swap_index] = True
            else:
              mutated_chromosome[i] = False

    self.chromosome = mutated_chromosome
    return self



#%%
ind1 = treasure_individual(treasure_problem())
#%%
ind1.cost
#%%

#%% md
# To get the Weight of the individual, need to sum the weights of the true values of the individual chromosome.
#%%
# Calculate the weight of ind1
ind1_weight = np.sum(weights[ind1.chromosome])

print(f"The weight of ind1 is: {ind1_weight}")
#%% md
# To get the Total Value of an individual, You need to sum the values of the array that has True values.
# 
#%%
# Calculate the Value of ind1
ind1_value = np.sum(values[ind1.chromosome])

print(f"The value of ind1 is: {ind1_value}")
#%% md
# Total Weight and Value of all Items
# 
#%%
# Total value of all items:
total_value = np.sum(values)
print(f"The total value of all items is: {total_value}")

#Total Weight of all items:
total_weight = np.sum(weights)
print(f"The total weight of all items is: {total_weight}")
#%%
class parameters:
  def __init__(self):
    self.population_size = 1000
    self.gene_mutation_rate = 0.8
    self.gene_mutation_range = 0.75
    self.birth_rate_per_generation = 1
    self.max_number_of_generations = 100
    self.weights = weights
    self.values = values
    self.max_weight = 60
#%%
def choose_parents(population_size):
  index1_parent = np.random.randint(0,population_size)
  index2_parent = np.random.randint(0,population_size)
  if index1_parent == index2_parent:
    return choose_parents(population_size)
  return index1_parent, index2_parent
#%%
def run_genetic(prob, params):
  # Read Variables
  population_size = params.population_size
  rate_of_gene_mutation = params.gene_mutation_rate
  # range_of_gene_mutation = params.gene_mutation_range
  # explore_crossover = params.explore_crossover_range
  cost_function = prob.cost_function
  number_of_children_per_generation = params.birth_rate_per_generation * population_size
  max_number_of_generations = params.max_number_of_generations
  # acceptable_cost = prob.acceptable_cost
  max_weight = params.max_weight
  weights = params.weights
  values = params.values

  # Create Our Population
  population = []
  best_solution = treasure_individual(prob)
  best_solution.cost = -100000


  for i in range(population_size):
    new_individual = treasure_individual(prob)
    if new_individual.cost > best_solution.cost:
      best_solution = deepcopy(new_individual)
    population.append(new_individual)



  # Start Loop
  for i in range(max_number_of_generations):
    #Start generation loop
    children = []
    while (len(children) < number_of_children_per_generation):
      #choose Parents
      parent1_index, parent2_index = choose_parents(population_size)

      parent1 = population[parent1_index]
      parent2 = population[parent2_index]

      # Create children
      child1, child2 = parent1.crossover(parent2)
      child1.mutate(rate_of_gene_mutation, weights)
      child2.mutate(rate_of_gene_mutation, weights)

      child1.cost = cost_function(child1)
      child2.cost = cost_function(child2)

      # add children to population
      children.append(child1)
      children.append(child2)

    #add children
    population += children

    # sort population
    population = sorted(population,key=lambda x: x.cost)

    # cull population
    population = population[:population_size]

    # check solution
    if population[0].cost > best_solution.cost:
      best_solution = deepcopy(population[0])

    print(best_solution.cost)

    # Calculate and print average weight of the population
    total_population_weight = sum(np.sum(params.weights[individual.chromosome]) for individual in population)
    average_weight = total_population_weight / len(population)
    print(f"Average weight of population: {average_weight}")

    # if best_solution.cost < acceptable_cost:
    #   break

  return (population, best_solution)
#%%
problem1 = treasure_problem()
params1 = parameters()
#%%

#%%

#%%
pop, best = run_genetic(problem1,params1)
#%%
best.chromosome
#%%
best.cost
#%%
best.weight
#%% md
# 