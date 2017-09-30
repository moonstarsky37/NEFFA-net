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
                 alpha = 1.0e-3,  
                 beta = 0.5,  
                 gamma = 1,
                 move_constant = np.exp(1),
                 trap_limit = 5,
                 w = 0.2,
                 hunted_release = 3,
                 huntered_rate = 0.01 ):
        self.fitness = fitness # a function
        self.fitness_cnt = 0
        self.trap = 0
        self.PopSize = PopSize
        self.Iteration = Iteration 
        self.GeneNo = GeneNo
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma # absorb coefficient
        self.move_constant = move_constant # inference from PSO
        self.movement = np.zeros([PopSize, GeneNo]) # inference from pso
        self.w = w
        self.huntered_rate = huntered_rate
        self.trap_limit = trap_limit
        self.hunted_release = hunted_release
        self.Pop = { 'PopSize' : self.PopSize,
                     'GeneNo': self.GeneNo,
                     'CurrentGeneration': 0,
                     'inds' : [ self.gen_ind(GeneNo) for i in range(PopSize)]
                   }

        plt.figure()
    pass 

    def run(self):
        # initializing
        self.update_fitness()
        self.BestInd = self.Pop['inds'][0].copy() # random assign for initializing
        
        # evolution
        for c_iter in range(1,self.Iteration + 1):
            self.update_alpha(c_iter)
            #print(self.alpha)
            self.update_gamma(c_iter)
            #self.update_position()
            self.update_position_with_NaturalEnemies(c_iter)
            self.update_fitness()
            self.aristogenics()
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
                best_ind = i.copy()
                
                # enhancing the nature enemy attack if the population do not go far
                if ( (self.BestInd['fitness'] - best_ind['fitness'])/self.BestInd['fitness'] ) > 1e-8:
                    self.trap = -self.trap_limit
                pass
                
            pass
        pass
        self.trap += 1
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

    def update_position_with_NaturalEnemies(self, NGen):
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
        if (self.huntered_rate * np.exp(self.trap - 1)) < np.random.random():
        #if (self.huntered_rate * self.trap) < np.random.random():
            delta_x = np.array([ 
                                self.move_constant * 
                                (matrix_q[i] - matrix_q) *
                                attractive_matrix[i].reshape(-1,1) # beta
                                for i in range(0, len(matrix_q)) 
                            ]) # distance * attractive
        else:
            delta_x = np.array([ 
                                self.move_constant * 
                                ( matrix_q - matrix_q[i]) * 
                                (10/self.huntered_rate) *  # dispersing rate, move back rapidly
                                attractive_matrix[i].reshape(-1,1) # beta
                                for i in range(0, len(matrix_q)) 
                            ]) 
            print("!!")
            self.trap = -self.trap_limit
        pass

        delta_x +=  ( (np.random.uniform(0, 1, len(delta_x.flatten())) - 0.5 ) *  # epsilon
                       np.random.uniform(0, self.alpha, len(delta_x.flatten())) # alpha
                    ).reshape(delta_x.shape) # alpha * epsilon
        
        self.movement *= self.w 
        for i in delta_x :
            #matrix_q += i
            self.movement += i
        pass
        matrix_q += self.movement 

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

        

    pass

    def aristogenics(self): # aristogenics using wheel
        inds = self.Pop['inds'].copy()
        fitness_matrix = np.array([])
        for i in range(0,len(inds)):
            fitness_matrix = np.append(fitness_matrix, inds[i]['fitness']) # fitness (I), using exponential to change the finess toward to small
        pass 
        fitness_matrix = (fitness_matrix - fitness_matrix.min()) / (fitness_matrix.max()-fitness_matrix.min())
        fitness_matrix = np.exp(fitness_matrix * -1)
        wheel = np.array([ (fitness_matrix * 100) / np.sum(fitness_matrix) ]).astype(int)

        wheel = np.hstack(np.array([ [i for j in range(0,wheel[0][i])]  for i in range(0,len(wheel[0]))]))
        np.random.shuffle(wheel)
        wheel = wheel.astype(int)

        try:
            self.Pop['inds'][wheel[0]] = self.BestInd.copy() 
        except:
            print("convergenced, get into early stop step ...")
            exit()
    pass

    def update_alpha(self, NGen):
        self.alpha *= np.exp(-(self.Iteration - NGen) / self.Iteration) 
    pass

    def update_gamma(self, NGen):
        self.gamma *= (self.Iteration/(np.exp(NGen)-2.17828))
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
               'Genes' : np.array( [ i for i in np.random.random(GeneNo)] ),
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


