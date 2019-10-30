import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
import threading
from threading import Thread, Lock
def readFile(fileName):
    """
    read file with name: fileName and return ia, ja, aij
    """
    try:
        f = open(fileName)
        l = f.readline().split(' ')
        if ('n' not in l):
            raise Exception('Неверный формат файла, не задано n')
        else:
            n = int(l[l.index('n') + 2])
        if ('n' not in l):
            raise Exception('Неверный формат файла, не задано nnz')
        else:
            nnz = int(l[l.index('nnz') + 2])

        l = int(f.readline().split(' ')[-2])
        i = f.readline()
        ia = []
        while i[:6] != 'VECTOR':
            ia+= [int(j) for j in i.split(' ')[:-1] if j != '']
            i = f.readline()
        if len(ia) != l:
            raise Exception('Неверный задано IA size')

        l = int(i.split(' ')[-2])
        i = f.readline()
        ja = []
        while i[:6] != 'VECTOR':
            ja+= [int(j) for j in i.split(' ')[:-1] if j != '']
            i = f.readline()
        if len(ja) != l:
            raise Exception('Неверный задано JA size')

        l = int(i.split(' ')[-2])
        i = f.readline()
        aij = []
        while i != '':
            aij+= [float(j) for j in i.split(' ')[:-1] if j != '']
            i = f.readline()
        if len(aij) != l:
            raise Exception('Неверный задано AIJ size')
    except Exception as ex:
        print(ex)

    finally:
        f.close
    return ia, ja, aij
    
class SLAE:
    """docstring"""
    
    def __init__(self, ia, ja, aij, num_threads):
        """Constructor"""
        self.csr = csr_matrix((aij, ja, ia), dtype=float)
        #self.b = [1,0,0,0,0]
        self.b = np.random.sample(self.csr.shape[1])
        self.betta = self.b / self.csr.diagonal()
        
        self.alpha = self.csr.transpose().multiply(-1 / self.csr.diagonal()).transpose()
        self.alpha.setdiag(0)
        self.alpha.eliminate_zeros()
        
        self.x = self.betta.copy()
        self.oldx = self.betta.copy()
        #-----generating threads-------
        self.num_threads = num_threads
        self.iteration_number = 0
        self.stack_avaliable_indexes = []
        self.neuviazka = float('inf')
        self.mutex = Lock()
        self.semaphore = threading.Semaphore(0)


    def parallel_MPI(self):
        "parallel MPI iteration"
        jobs = []
        for i in range(self.num_threads):
            job = myThread(self)
            jobs.append(job)
            job.configure(i)
        

        for job in jobs:
            job.start()
        while self.neuviazka > 1e-10:
            self.neuviazka = 0
            self.iteration_number += 1
            self.stack_avaliable_indexes = [x for x in range(0, self.x.size)]
            for job in jobs:
                job.resume()
            
            for job in jobs:
                self.semaphore.acquire(blocking=True)
        
            """ while False in [job.paused for job in jobs]:
                time.sleep(0.1)
                print([job.paused for job in jobs])
             """
            #time.sleep(0.05)
            self.oldx = self.x.copy()
        for job in jobs:
            job.kill()
    
#     def printt(self):
#         """
#         Stop the car
#         """
#         try:
#             if(self.X):
#                 print(True)
#         except AttributeError:
#             print(False)
#         return self.csr.toarray()
    
    def forward(self):
        """
        do next iteration, return accuracy
        """
        self.iteration_number += 1
        x = self.x
        self.x = self.alpha * self.x + self.betta
        t = x - self.x

        return (t * t).sum()

    def forward_one_component(self, i):
        """next iteration of one component, i - number of component"""
        self.x[i] = self.alpha.getrow(i) * self.oldx + self.betta[i]
        return abs(self.oldx[i] - self.x[i])

class myThread (threading.Thread):
    def __init__(self, slae):
        threading.Thread.__init__(self)
        self.slae = slae
        self.i = -1
        self.lock = threading.Semaphore(0)
        self.paused = False
        self.Alive = True

    def configure(self, i):
        self.i = i

    def run(self):
        self.pause()
        while(self.Alive):
            self.slae.mutex.acquire()
            if len(self.slae.stack_avaliable_indexes) == 0:
                self.slae.mutex.release()
                self.slae.semaphore.release()
                self.pause()
                
                continue
            i = self.slae.stack_avaliable_indexes.pop()
            self.slae.mutex.release()
            ###
            neuviazka = self.slae.forward_one_component(i)
            ###
            self.slae.mutex.acquire()
            self.slae.neuviazka = max(neuviazka, self.slae.neuviazka)
            self.slae.mutex.release()
            ###
        
    def pause(self):
        self.paused = True
        self.lock.acquire()

    def resume(self):
        #self.slae.semaphore.release() 
        self.paused = False
        self.lock.release()


    def kill(self):
        self.Alive = False


def main():
    ia, ja, aij = readFile("matrix/matrix2_n100.mycsr")
    #last argument - amout of processes
    sl = SLAE(ia, ja, aij, 20)

    start_recur = time.time()

    lis = []
    i = sl.forward()
    #------------------------------------
    print("with parallel")
    sl.parallel_MPI()
    end_recur = time.time()
    print(end_recur - start_recur)
    print("\nsolution:")
    print(sl.x)
    #-------------------------------------
    print("without parallel")
    start_recur = time.time()
    while i > 1e-10:
        i = sl.forward()
        lis.append(i)
    
    end_recur = time.time()
    print(end_recur - start_recur)
    print("\nsolution:")
    print(sl.x)

    step = len(lis) // 100
    #plt.scatter(x=np.arange(len(lis[::step])) * step, y=lis[::step])
    #plt.show()

if __name__ == "__main__":
    main()