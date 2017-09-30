import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf 

class PSO():
    pass 

class Firefly():
    '''
    Remember to set fitness function to connect with this dummy fitness.
    Example:
    In main.py:
        def fitness(array):
            result = caculation
            return result

        def main():
            Optimizer = EvolutionAlgorithm.Firefly(10, fitness)
            Optimizer.run()
    '''
    def __init__(self,
                 GeneNo , 
                 fitness, 
                 PopSize = 40, 
                 Iteration = 5000,  
                 alpha = 0.01,  
                 beta = 1,  
                 gamma = 0.1,
                 huntered_rate = 0.0 ):
        self.fitness = fitness # a function
        self.fitness_cnt = 0
        self.PopSize = PopSize
        self.Iteration = Iteration 
        self.GeneNo = GeneNo
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma # absorb coefficient
        self.huntered_rate = huntered_rate
        self.Pop = { 'PopSize' : self.PopSize,
                     'GeneNo': self.GeneNo,
                     'CurrentGeneration': 0,
                     'inds' : [ self.gen_ind(GeneNo) for i in range(PopSize)]
                   }

        plt.figure()
    pass 

    def run(self):
        self.update_fitness()
        self.BestInd = self.find_the_best(self.Pop).copy()
        print(self.BestInd['fitness'])
        for c_iter in range(1,self.Iteration + 1):
            self.update_alpha(c_iter)
            self.update_position()
            #self.update_position_with_hunting(c_iter)
            self.update_fitness()
            CBestInd = self.find_the_best(self.Pop)
            if self.BestInd['fitness'] > CBestInd['fitness'] :
                self.BestInd = CBestInd.copy()
            pass 
            print(self.BestInd['fitness'])
        
        pass
    pass 
    
    def find_the_best(self, pop):
        inds = pop['inds']
        best_ind = inds[0]
        for i in inds:
            if i['fitness'] < best_ind['fitness'] :
                best_ind = i
            pass
        pass
        return best_ind 
    pass

    def update_position(self):
        inds = self.Pop['inds'].copy()

        # making distance matrix
        distance_matrix = []
        fitness_matrix = np.array([])
        matrix_q = np.array( [ inds[j]['Genes'] for j in range(0,len(inds)) ] )

        for i in range(0,len(inds)):
            matrix_s = np.array( [j for j in inds[i]['Genes']] )
            distance_matrix.append( [np.sqrt(np.sum(i)) for i in (matrix_q-matrix_s)**2] ) # distance
            fitness_matrix = np.append(fitness_matrix, inds[i]['fitness']) # fitness (I), using exponential to change the finess toward to small
        pass 

        # fitness, multiplicative inverse
        fitness_matrix = np.power(fitness_matrix, -1)
        #fitness_matrix = (fitness_matrix - fitness_matrix.min()) / (fitness_matrix.max() - fitness_matrix.min())


        # making attractive matrix (beta)
        attractive_matrix = (
                             fitness_matrix * # beta
                             np.exp( ( (self.gamma)*(np.power(distance_matrix,2)) )*-1 )
                            )

        # making delta-x matrix
        delta_x = np.array([ (matrix_q[i] - matrix_q) * 
                              attractive_matrix[i].reshape(-1,1) # beta
                              for i in range(0, len(matrix_q)) ]) # distance * attractive
        delta_x +=  ( (np.random.uniform(0, 1, len(delta_x.flatten())) - 0.5 ) *  # epsilon
                       np.random.uniform(0, self.alpha, len(delta_x.flatten())) # alpha
                    ).reshape(delta_x.shape) # alpha * epsilon
        for i in delta_x :
            matrix_q += i
        pass
        #print(matrix_q.shape)

        # ##### visualization the particel
        # plt.scatter(matrix_q.T[0],matrix_q.T[1])
        # plt.draw()
        # plt.pause(0.005)
        # plt.gcf().clear()
        # #####

        np.clip(matrix_q, 0, 1)

        # assign the updated position
        self.Pop['inds'] = [ 
                            {
                            'Genes' : i,
                            'fitness' : np.NAN ,
                            'modify' : True
                            } for i in matrix_q
                           ]
        self.Pop['inds'][np.random.randint(len(matrix_q))] = self.BestInd.copy() # aristogenics

    pass

    def update_position_with_hunting(self, NGen):
        #self.gamma = self.update_gamma(NGen)

        inds = self.Pop['inds'].copy()

        # making distance matrix
        distance_matrix = []
        fitness_matrix = np.array([])
        matrix_q = np.array( [ inds[j]['Genes'] for j in range(0,len(inds)) ] )

        for i in range(0,len(inds)):
            matrix_s = np.array( [j for j in inds[i]['Genes']] )
            distance_matrix.append( [np.sqrt(np.sum(i)) for i in (matrix_q-matrix_s)**2] ) # distance
            fitness_matrix = np.append(fitness_matrix, inds[i]['fitness']) # fitness (I), using exponential to change the finess toward to small
        pass 

        # fitness, multiplicative inverse
        fitness_matrix = np.power(fitness_matrix, -1)
        #fitness_matrix = (fitness_matrix - fitness_matrix.min()) / (fitness_matrix.max() - fitness_matrix.min())


        # making attractive matrix (beta)
        attractive_matrix = (
                             fitness_matrix * # beta
                             np.exp( ( (self.gamma)*(np.power(distance_matrix,2)) )*-1 )
                            )

        # making delta-x matrix
        if (np.random.random() < self.huntered_rate) : 
            delta_x = np.array([ 
                                (matrix_q - matrix_q[i]) * # reversed !!
                                attractive_matrix[i].reshape(-1,1) # beta
                                for i in range(0, len(matrix_q)) ]) # distance * attractive
        else:
            delta_x = np.array([ 
                                (matrix_q[i] - matrix_q) * 
                                (1/(1 - self.huntered_rate)) *  # dispersing rate
                                attractive_matrix[i].reshape(-1,1) # beta
                                for i in range(0, len(matrix_q)) ]) # distance * attractive
        pass

        delta_x +=  ( (np.random.uniform(0, 1, len(delta_x.flatten())) - 0.5 ) *  # epsilon
                       np.random.uniform(0, self.alpha, len(delta_x.flatten())) # alpha
                    ).reshape(delta_x.shape) # alpha * epsilon
        for i in delta_x :
            matrix_q += i
        pass
        #print(matrix_q.shape)

        ##### visualization the particel
        plt.scatter(matrix_q.T[0],matrix_q.T[1])
        plt.draw()
        plt.pause(0.005)
        plt.gcf().clear()
        #####

        np.clip(matrix_q, 0, 1)

        # assign the updated position
        self.Pop['inds'] = [ 
                            {
                            'Genes' : i,
                            'fitness' : np.NAN ,
                            'modify' : True
                            } for i in matrix_q
                           ]
        self.Pop['inds'][np.random.randint(len(matrix_q))] = self.BestInd.copy() # aristogenics

    pass

    def update_alpha(self, NGen):
        self.alpha *= (self.Iteration - NGen)/self.Iteration 
    pass

    def update_gamma(self, NGen):
        self.gamma = np.exp((self.Iteration/2) - NGen)
    pass

    def update_fitness(self):
        for ind in self.Pop['inds']:
            if ind['modify'] == True:
                ind['fitness'] = self.fitness(ind['Genes'])
                self.fitness_cnt += 1
                ind['modify'] = False
            pass 
        pass 
    pass

    def gen_ind(self, GeneNo):
        ind = {
               'Genes' : np.array( [ i for i in np.random.rand(GeneNo)] ),
               'fitness' : np.NAN ,
               'modify' : True
              }
        return ind 
    pass    

    def fitness():
        '''
        This section depends on the problem
        '''
        print("PLEASE DEFINE THE FITNESS FUNCTION!!!!")
        exit()
    pass

pass 


class sample_fitness():
    def __init__(self):
        pass

    def Rastrigin(self, solution):
        '''
        f(0,0) = 0 (minimum)
        search domain : -5.12~5.12
        '''
        A = 10
        return A * len(solution) + np.sum([ i ** 2 - A*np.cos(2*np.pi*i) for i in solution])

pass
