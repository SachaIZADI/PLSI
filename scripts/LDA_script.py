#import pyspark
from pyspark.ml.clustering import LDA
from pyspark.ml.linalg import Vectors, SparseVector


#sc = pyspark.SparkContext()

def splitTrainTest(document, alpha=0.1, seed=42):
    train_test = document.randomSplit([1-alpha, alpha], seed=seed)
    train, test = train_test[0], train_test[1]
    
    # We need to check that film 's' in the test set is also in the train set 
    # This is mandatory for evaluating our model
    s_train = train.map(lambda x : x[1])
    s_test = test.map(lambda x : x[1])
    
    # Here we can't just consider the rdd s_test.intersection(s_train) because then we need to filter 
    # the test set and a common reference needs to be shared between all the computers in the cluster.
    s_common = sc.broadcast(s_test.intersection(s_train).collect())
    
    
    # Same for a user 's', we need to check that if he is in the test set, then he is in the train set.
    u_train = train.map(lambda x : x[0])
    u_test = test.map(lambda x : x[0])
    u_common = sc.broadcast(u_test.intersection(u_train).collect())
    
    
    # filter the test set : If an element from the test set is not in the train set, we delete it
    test = test.map(lambda x : (x[0], x[1]))\
               .filter(lambda x : x[1] in s_common.value)\
               .filter(lambda x : x[0] in u_common.value)\
               .map(lambda x : (x[0], x[1]))
            
    return([train,test])

#Load document: change the path if needed. Use the commented line below to load from HDFS
document = sc.textFile("/Users/clementponsonnet/Desktop/Code/Github/PLSI/Data/ml-latest-small/ratings.csv")

#Removing the header
header = document.first()
document = document.filter(lambda x: x != header)

# Each line is converted to a tuple: user, movie
parts = document.map(lambda l: l.split(","))
ratings = parts.map(lambda p: (p[0], p[1].strip()))

#Split into train and test
train = splitTrainTest(ratings, alpha=0.1)[0]
test = splitTrainTest(ratings, alpha=0.1)[1]


#aggregate on users
ag_rating = train.reduceByKey(lambda x, y: "{0},{1}".format(x, y))
#Put it in a list
ag_rating = ag_rating.map(lambda p: (p[0], p[1].strip().split(',')))
#convert strings to int for key, float for values
for_LDA = ag_rating.map(lambda line: [int(line[0]), Vectors.dense([float(x) for x in line[1]])])

#Use sparseVectors to construct adjacency matrix
max_movie_id = train.map(lambda x: int(x[1])).max()
adjmat = for_LDA.map(lambda line: [int(line[0]), Vectors.sparse(max_movie_id+1, line[1], [1] * len(line[1]))])

#Prepare for ml package
#Put it all in a df
adjmatDf = spark.createDataFrame(adjmat)
adjmatDf = adjmatDf.selectExpr("_1 as label", "_2 as features")

#Specify number of clusters
nb_k = 10
lda = LDA(k=nb_k).setOptimizer("em")
model = lda.fit(adjmatDf)

##p(s|z):
topicsRdd = model.describeTopics(max_movie_id).rdd

Pzs = topicsRdd.flatMap(lambda x: [ ([x[0], x[1][i]], x[2][i]) for i in range(len(x[1])) ])

##p(z|u):
transformed = model.transform(adjmatDf)
transformedRdd = transformed.rdd
Puz = transformedRdd.flatMap(lambda x: [ ([x[0], i], x[2][i]) for i in range(nb_k) ])

#p(s|u):
Puz_bis = Puz.map(lambda x : (x[0][1], [x[0][0],x[1]]))
Pzs_bis =  Pzs.map(lambda x : (x[0][0], [x[0][1],x[1]]))

##########Here we want to join Puz_bis and Pzs_bis
#Need to do a broadcast join. Possible because Puz_bis is small and Pzs_bis is large
smallDict=dict( (x[0], x[1]) for x in Puz_bis.collect() )
bc=sc.broadcast(smallDict)
mapJoined = Pzs_bis.map(lambda x : (x[0], (bc.value[x[0]], x[1])))

#Calculate Psu
Psu = mapJoined.map(lambda  x: ((x[1][0][0] , x[1][1][0]), (x[1][0][1]*x[1][1][1])))
Psu = Psu.reduceByKey(lambda x,y : x+y)

#Define the threshold
threshold = .00001

#Make predictions
pred = Psu.map(lambda x: (x[0], x[1] > threshold))
train_for_roc = train.map(lambda x: (int(x[0]), int(x[1]))) 
test_for_roc = test.map(lambda x: (int(x[0]), int(x[1])))

def precisionRecallTrain(pred, train) :
	# pred is of the form [(u,s), boolean)]
	# train is of the form : (u,s)
	# positive predictions (we predict that user u has seen film s)
	pos = pred.filter(lambda x : x[1]).map(lambda x : x[0])
	nb_pos = pos.count()
	nb_TP = train.intersection(pos).count() # Our positive predictions are also in the train set
	nb_FP = nb_pos - nb_TP 
	#negatives predictions (we predict that user u has never seen film s)
	neg = pred.filter(lambda x : not(x[1])).map(lambda x : x[0])
	nb_neg = neg.count()
	nb_FN = train.intersection(neg).count() # Our negative predictions are actually in the train set
	nb_TN = nb_neg - nb_FN
	# compute precision and recall
	precision = nb_TP / (nb_FP + nb_TP)
	recall = nb_TP / (nb_FN + nb_TP)
	return((precision,recall))

def precisionRecallTest(pred, train_for_roc, test_for_roc) :
    # prediction is of the form [('u,s', boolean)]
    # train is of the form : ['u,s']
    # test is of the form : ['u,s']
    # Remove from pred the films already seen by users. To do so we use substractByKey() so we need 
    # to append something to the train rdd
    to_remove = train_for_roc.map(lambda x : (x,1))
    predict = pred.subtractByKey(to_remove)
    
    
    # positive predictions (we predict that user u has seen film s)
    pos = predict.filter(lambda x : x[1]).map(lambda x : x[0])
    nb_pos = pos.count()
    nb_TP = test_for_roc.intersection(pos).count() # Our positive predictions are also in the train set
    nb_FP = nb_pos - nb_TP 
    
    
    #negatives predictions (we predict that user u has never seen film s)
    neg = predict.filter(lambda x : not(x[1]))\
                 .map(lambda x : x[0])
    nb_neg = neg.count()
    nb_FN = test_for_roc.intersection(neg).count() # Our negative predictions are actually in the train set
    nb_TN = nb_neg - nb_FN
    
    
    # compute precision and recall
    precision = nb_TP / (nb_FP + nb_TP)
    recall = nb_TP / (nb_FN + nb_TP)
    
    return((precision,recall))

#Calculate precision and recall on test set
precisionRecallTest(pred, train_for_roc, test_for_roc)
