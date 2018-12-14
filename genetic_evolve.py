import numpy as np 
# import tensorflow as tf 
# import matplotlib.pyplot as plt
import time
import random
import copy
from joblib import Parallel, delayed
import multiprocessing
from pprint import pprint
from tabulate import tabulate
#%%


class Genetic_Opt(object):
	def samplefun(self):
		return 1

	def __init__(self, population_size=5, ngenes=3, datatype = [float, int, int], lbound = [0, 0, 0], ubound = [1, 1, 1], fitfun=samplefun, max_mutation_rate=.2, max_mutation_size=5,history_filename='history.csv'):
		self.pop_size = int(population_size)
		self.ngenes = ngenes

		self.population = np.zeros((self.pop_size, self.ngenes))
		self.fit_fun = fitfun
		self.max_mut_rate = max_mutation_rate
		self.max_mut_size = max_mutation_size
		self.fitness = np.zeros((self.pop_size,1))
		self.R = 5
		self.best_fitness = 0
		self.history_filename = history_filename
		self.datatype = datatype
		self.lbound = lbound
		self.ubound = ubound
		self.gen = 0
		if ngenes != len(datatype):
			print("Warning! datatype length not equal to gene length!")

	def initial_population(self, seed_chrome):
		self.population[0,:] = seed_chrome

		for i in range(1,len(self.population)):
			self.population[i,:] = self.mutate(seed_chrome)


	def mutate(self, chromo):
		#inputs: chromosome
		#outputs: mutated  chromosome
		# nmutate_chromes = int(self.max_mut_rate * self.ngenes); #number of chromosomes to mutate
		chromosome = chromo
		nmutate_chromes = random.randint(0, self.ngenes)
		mutations = np.random.normal(0,.2,size=nmutate_chromes) #create mutations
		if self.gen == 0:
			max_mrate = 2.0
		else:
			max_mrate = self.max_mut_rate
		mutations = mutations * max_mrate #normalize mutations to maximum mutation rate
		mutations = np.append(mutations, np.zeros((self.ngenes - nmutate_chromes,1)))
		# mutations = mutations * self.max_mut_size * 2
		# mutations = mutations.astype(int)
		mutations = mutations + 1.0
		


		random.shuffle(mutations)

		chromosome = chromosome * mutations

		# print(chromosome)
		#Ensure gene is above 0
		# for i in np.where(chromosome<1)[:]:
		# 	chromosome[i] = 1

		#Enforce datatype

		for i in range(0, len(self.datatype)):
			chromosome[i] = self.datatype[i](chromosome[i])

			if chromosome[i] < self.lbound[i]:
				chromosome[i] = self.lbound[i]
			if chromosome[i] > self.ubound[i]:
				chromosome[i] = self.ubound[i]

			# if datatype[i] ==  'i':
			# 	chromosome[i] = int(chromosome[i])
			# elif datatype[i] == 'f':
			# 	chromosome[i] = float(chromosome[i])

		return chromosome

	def crossover(self, inchrome1, inchrome2):
		chrome1 = copy.copy(inchrome1)
		chrome2 = copy.copy(inchrome2)

		crosspoint = random.randint(1, self.ngenes-1)
		output1 = inchrome1
		output1[0:crosspoint] = chrome2[0:crosspoint]

		output2 = inchrome2
		output2[0:crosspoint] = chrome1[0:crosspoint]		

		return output1, output2

	def run_fitfun(self,i):
		# for i in range(0, len(self.population)):
		fitness = self.fit_fun(self.population[i,:])
		return fitness

	def sort_population(self):

		self.fitness = np.zeros((self.pop_size,1))

		# self.run_fitfun()

		num_cores = multiprocessing.cpu_count()
		# print(self.population[1,:])

		# for i in range(len(self.population)):
		# 	self.fitness[i] = self.run_fitfun(i)


		self.fitness = Parallel(n_jobs=num_cores)(delayed(self.run_fitfun)(i) for i in range(len(self.population)))
		self.fitness = np.array(self.fitness)

		# self.fitness = np.array(self.fitness).reshape((self.pop_size,3))

		self.full_list = np.hstack((self.population, np.array(self.fitness)))

		self.full_list = self.full_list[self.full_list[:,-1].argsort()]
		# self.full_list = np.flipud(self.full_list)


		self.population = self.full_list[:, 0:-3]
		# print('population')
		# print(self.population)

		self.fitness = np.transpose(self.full_list[:, -1])


		self.R = abs(self.best_fitness-self.fitness[0])


	def selection(self):
		 self.sort_population()

	def breed(self):


		new_pop = self.population
		for i in range(3,int((self.pop_size-1)/2)):
			repro_prob = np.multiply(np.flipud(range(self.pop_size)),  np.random.rand(self.pop_size))
			A = np.zeros((self.pop_size,2))
			A[:,0] = repro_prob
			A[:,1] = range(self.pop_size)
			A = np.flipud(A[A[:,0].argsort()])
			chrome1 = self.population[int(A[0,1]),:]
			chrome2 = self.population[int(A[1,1]),:]
			new_chrome1, new_chrome2 = self.crossover(chrome1, chrome2)
			new_pop[2*i-1,:] = new_chrome1 
			new_pop[2*i,:] = new_chrome2

		self.population = new_pop

	def mutation(self):
		for i in range(3,(self.pop_size-1)):
			chromosome = self.population[i,:]
			# chromosome = NNCheck(chromosome)
			if random.random()<.5:
				self.population[i,:] = self.mutate(chromosome)
			while self.check_duplicates(chromosome,i):
				self.population[i,:] = self.mutate(chromosome)


	# def NNCheck(self,chromosome):
	# 	chromo = chromosome
	# 	nlayers = int(chromo[1])
	# 	for i in range(2+nlayers, len(chromo)):
	# 		chromo[i] = 0
	# 	return chromo


	def check_duplicates(self, chromosome,index):
		chromo = chromosome
		nlayers = int(chromo[1])
		for j in range(0,(self.pop_size-1)):
			if np.allclose(self.population[j,0:nlayers + 2],chromo[0:nlayers+2], rtol=1e-05) and j != index:
				return True

		return False

	def print_to_file(self,gen):
		# with open(self.history_filename, "a") as myfile:

		# print(self.full_list)
		# print(gen)
		# 	print full_list

		# 	myfile.write(full_list[1,:])
		f_handle = open(self.history_filename, 'a')
		# f_handle.write("Gen: %d \n" % gen)
		a = np.ones(len(self.full_list))
		a = a[:, None]

		np.set_printoptions(precision=1)
		printlist = np.hstack((np.array(a*gen), self.full_list))
		print(tabulate(printlist.tolist(),headers=['Gen', 'LR', 'NL', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'NN6', 'MSE', 'time', 'f']))
		np.savetxt(f_handle, printlist, fmt='%d %.2e %d %d %d %d %d %d %d %.3e %.3e %.3e')
		f_handle.close()

	def evolve(self, ngenerations):
		#Run simulation

		tol = 1e-2

		gen = 0
		self.mutation()

		while gen < ngenerations or self.R < tol:
			self.selection()
			self.print_to_file(gen)
			gen = gen+1
			self.breed()
			self.mutation()

def fun_fun(X):
	time.sleep(2)

	return -X@X * .1


def main():
	num_cores = multiprocessing.cpu_count()
	evolution = Genetic_Opt(population_size=num_cores * 2 + 1, ngenes=3, datatype = [float, int,int,int, int], lbound = [0.000001, 1, 1], fitfun=fun_fun, max_mutation_rate=.2, max_mutation_size=5,history_filename='history.csv')
	evolution.initial_population([.001, 2, 3])
	evolution.evolve(50)




