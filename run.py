import pandas as pd
import timeit
from cft import CondensedFilterTreeClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.metrics import label_ranking_loss
from sklearn import tree
from sklearn import svm

'''
general variables used for CFT running
'''
DATA_FOLDER = 'data/'#Link to data folder
dataset_dict = {'emotions':72,'flags':19,'scene':294,'yeast':103,'birds':260} #Dictionary of datasets - key: name of dataset value:number of attributes

'''
Parameters for CFT running
'''
LOSS_FUNC = 'hamming'
M = 2 #Number of iterations
node_classifier = tree.DecisionTreeClassifier()

'''
Function for computing f1 and hamming scores.
Parameters:
-real_y: real class values (vector of 0/1)
-pred_y: predicted class values (vector of 0/1)
'''
def compute_scores(real_y,pred_y):
    hamm_score = 0.0
    f1score = 0.0
    numOfObservations = len(real_y)
    for index in range(len(real_y)):
        hamm_score = hamm_score + hamming_loss(real_y[index],pred_y[index])
        #rank_score = rank_score + label_ranking_loss(real_y[index],pred_y[index])
        f1score = f1score + f1_score(real_y[index],pred_y[index],average='binary')

    hamm_score = hamm_score /numOfObservations
    f1score = f1score/numOfObservations
    return hamm_score,f1score

'''
Main script code.
This code is used for reading the datasets and applaying the CFT algorithm.
It computes the f1 and hamming scores for each dataset
'''
for key,value in dataset_dict.items():
    train_set = pd.read_csv(DATA_FOLDER+key+'-train.csv',',',header=None)
    test_set = pd.read_csv(DATA_FOLDER+key+'-test.csv',',',header=None)
    num_columns = train_set.shape[1]

    train_x = train_set.ix[:,0:value-1]
    train_y = train_set.ix[:,value:num_columns]

    test_x = test_set.ix[:,0:value-1]
    test_y = test_set.ix[:,value:num_columns]

    model = CondensedFilterTreeClassifier()
    model.set_params(M, LOSS_FUNC,node_classifier)

    start = timeit.default_timer()
    model.fit(train_x,train_y)
    stop = timeit.default_timer()
    run_time_minutes = (stop-start)/60
    predict_y = model.predict(test_x)

    hamming,f1 = compute_scores(test_y.values.tolist(),predict_y)

    m_print, loss_print, classifier_print = model.get_params()

    print("Results for dataset "+key+":")
    print("Number of iterations: " + str(m_print))
    print("Loss function: " + loss_print)
    print("Node classifier: " + str(classifier_print))
    print("Train time: "+str(run_time_minutes)+" min.")
    print("Hamming Loss: "+str(hamming))
    print("F1 score: " + str(f1)+"\n")
