#import pandas as pd
import numpy as np
from numpy import random
import pyspark
from timeit import default_timer as timer


sc = pyspark.SparkContext()

start = timer()

def parseLine(line):
    line = line.split(',')
    line = line[0]+','+line[1]
    return(line)

def cartesianProd(us):
    to_return = []
    for z in range(nb_z.value):
        to_return += [us+','+str(z)]
    return(to_return)

nb_z = sc.broadcast(3)
nb_iter = 3
document = sc.textFile("hdfs:///user/hadoop/plsi/input/ratings_1m.csv")
#Removing the header
header = document.first()
document = document.filter(lambda x: x != header)

doc = document.map(parseLine)

'''
#We skip the train test split part: We did not split into training and test set for the scaling part of the study

###### Initialisation 
# creation of the tuples (u,s,z ; q*)
def splitTrainTest(document, alpha=0.1, seed=42):
    train_test = document.randomSplit([1-alpha, alpha], seed=seed)
    train, test = train_test[0], train_test[1]
    # We need to check that film 's' in the test set is also in the train set 
    # This is mandatory for evaluating our model
    s_train = train.map(lambda x : x.split(',')[1])
    s_test = test.map(lambda x : x.split(',')[1])
    # Here we can't just consider the rdd s_test.intersection(s_train) because then we need to filter 
    # the test set and a common reference needs to be shared between all the computers in the cluster.
    s_common = sc.broadcast(s_test.intersection(s_train).collect())
    # Same for a user 's', we need to check that if he is in the test set, then he is in the train set.
    u_train = train.map(lambda x : x.split(',')[0])
    u_test = test.map(lambda x : x.split(',')[0])
    u_common = sc.broadcast(u_test.intersection(u_train).collect())
    # filter the test set : If an element from the test set is not in the train set, we delete it
    test = test.map(lambda x : (x.split(',')[0], x.split(',')[1]))\
               .filter(lambda x : x[1] in s_common.value)\
               .filter(lambda x : x[0] in u_common.value)\
               .map(lambda x : (x[0]+','+x[1]))        
    return([train,test])

train = splitTrainTest(doc, alpha=0.1)[0]
test = splitTrainTest(doc, alpha=0.1)[1]
'''

#Loglikelihood functon
def logLikelihood(Psu,train):
    T = sc.broadcast(train.count())
    to_join = train.map(lambda x : (x,1))
    Psu_filter = to_join.join(Psu)
    Psu_filter = Psu_filter.map(lambda x : (x[0],x[1][1]))
    return(-1/T.value * Psu_filter.map(lambda x : np.log(x[1])).reduce(lambda x, y: x + y))



q = document.flatMap(cartesianProd).map(lambda usz : (usz,random.rand()))
num_part = q.getNumPartitions()
#q.collect()[:3]

epsilon = .005
curr_logLik = 0
prev_logLik = 0
nb_iter = 0
logLiks = []
times = []

print('\n \n \n Starting the loop !! \n \n \n ')

for i in range(1):
    nb_iter += 1
    # *************************** M-step **************************
    # ********* computation of the N(s,z) & N(z) functions *********
    # return (s,z, N(s,z))
    iter_start = timer()
    Nsz = q.map(lambda Q : (Q[0].split(',')[1]+','+Q[0].split(',')[2],Q[1])).reduceByKey(lambda x,y : x+y)
    Nz = Nsz.map(lambda N : (N[0].split(',')[1], N[1])).reduceByKey(lambda x,y : x+y)
    Nsz = Nsz.map(lambda x : (x[0].split(',')[1], (x[0].split(',')[0],x[1])))
    tmpN = Nsz.join(Nz).coalesce(num_part) #ICI
    Nsz_normalized = tmpN.map(lambda x : (x[1][0][0]+','+x[0], x[1][0][1]/x[1][1]))   
    # ********* computation of the p(z|u) function *********
    Puz = q.map(lambda Q : (Q[0].split(',')[0]+','+Q[0].split(',')[2],Q[1])).reduceByKey(lambda x,y : x+y)
    Pu = Puz.map(lambda p : (p[0].split(',')[0], p[1])).reduceByKey(lambda x,y : x+y)
    Puz = Puz.map(lambda x : (x[0].split(',')[0], (x[0].split(',')[1],x[1])))
    tmpP = Puz.join(Pu).coalesce(num_part) #ICI
    Puz = tmpP.map(lambda x : (x[0]+','+x[1][0][0], x[1][0][1]/x[1][1]))
    # *************************** E-step **************************
    tmpQ = q.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[2] , x[0].split(',')[1]))
    tmpQ = tmpQ.join(Puz).coalesce(num_part) #ICI
    tmpQ = tmpQ.map(lambda x : (x[1][0]+','+x[0].split(',')[1] ,\
                                 (x[0].split(',')[0],x[1][1])))
    tmpQ = tmpQ.join(Nsz_normalized).coalesce(num_part) #ICI
    tmpQ = tmpQ.map(lambda x : (x[1][0][0]+','+x[0],\
                                 x[1][0][1]*x[1][1]))
    ####
    #Use this to compute loglikelihood later
    sumTmpQ = tmpQ.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[1],x[1])).reduceByKey(lambda x,y : x+y)
    tmpQ = tmpQ.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[1],\
                                 (x[0].split(',')[2],x[1])))
    tmpQ = tmpQ.join(sumTmpQ).coalesce(num_part) #ICI
    q = tmpQ.map(lambda x : (x[0]+','+x[1][0][0], x[1][0][1]/x[1][1]))
    ####
    ####
    ####Part to calculate loglikelihood
    prev_logLik = curr_logLik
    curr_logLik = logLikelihood(sumTmpQ,doc)
    logLiks.append(curr_logLik)
    iter_time = timer() - iter_start
    times.append(iter_time)
    print(' \n \n \n Iteration {0} finished ! \n Current loglikelihood: {1} \n Iteration time: {2} seconds \n \n \n'.format(nb_iter, curr_logLik, iter_time))
    # we need to collect at this point because otherwise it behaves like a recursive calling which may crash
    #q = q.persist()
    #q = sc.parallelize(q.collect())
#q.collect()

Puz_bis = Puz.map(lambda x : (x[0].split(',')[1], \
                 (x[0].split(',')[0],x[1])))
Nsz_normalized_bis =  Nsz_normalized.map(lambda x : (x[0].split(',')[1], \
                 (x[0].split(',')[0],x[1])))
final = Puz_bis.join(Nsz_normalized_bis)
final = final.map(lambda  x: (x[1][0][0] +','+x[1][1][0], \
                           (x[1][0][1]*x[1][1][1])))
final = final.reduceByKey(lambda x,y : x+y)
result = final

# ...
end = timer()
elapsed_time = end - start

#Save some metadata directly on the client

with open('output_metadata.txt', 'w+') as the_file:
    the_file.write('total runtime: {0}'.format(elapsed_time))
    the_file.write('total iterations: {0}'.format(nb_iter))
    the_file.write('Loglikelihoods: {0}'.format(logLiks))
    the_file.write('iteration times: {0}'.format(times))

#Save the Psu result on hdfs
result.saveAsTextFile("hdfs:///user/hadoop/plsi/output/P_s_u")