import numpy as np
import math as mt
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
import urllib
import nD_GDA_2c as nD
from sklearn.naive_bayes import MultinomialNB # for checking purpose.......
import gmpy as gm
import ssl
import trainning_cal as tc
import time
from collections import Counter
import PHNB_v2_irirs as pv2


def predict_prior_prob(y):

    y_prob = {}
    for cl in np.unique(y):
        y_prob.update({int(cl):round(float(float(((y==cl).sum())+(1/len(np.unique(y))))/((y.shape[0])+1.0)),5)})

    #print y_prob
    #exit(0)
    return y_prob

def find_prob_ai_c(X,y,Ai,c,ap1,prob_c):
    total_occurence = 0
    for i in range(X.shape[0]):
        if((X[i][ap1]==Ai and y[i]==c)):
            total_occurence += 1

    prob_ai_c = float(total_occurence)/X.shape[0]
    prob_ai_given_c = float(total_occurence)/X.shape[0]/prob_c

    return round(prob_ai_c,5),round(prob_ai_given_c,5)


def find_prob_ai_aj_c(X,y,Ai,Aj,c,ap1,ap2):

    #make easier
    total_occurence = 0
    for i in range(X.shape[0]):
        if((X[i][ap1]==Ai and X[i][ap2]==Aj and y[i]==c)):
            total_occurence += 1

    return round(float(total_occurence)/X.shape[0],5)


def all_probability_dict(key_pair,key_aj_c_pair,key_ai_c_pair,X,y,prior_class_prob):

    #print key_pair.count("5.7#4.4#0#0#1")

    frequency_count = Counter(key_pair)
    frequency_count_aj_c = Counter(key_aj_c_pair)
    frequency_count_ai_c = Counter(key_ai_c_pair)


    rows = X.shape[0]
    cols = X.shape[1]
    prob_ai_aj_c = {k: round(float(v)/rows,5) for k,v in frequency_count.items()}
    prob_aj_c = {k: round(float(v)/rows,5) for k,v in frequency_count_aj_c.items()}
    prob_ai_given_c = {k: round(float(v)/(prior_class_prob[int(k.split("#")[1])]*rows),5) for k,v in frequency_count_ai_c.items()}
    prob_aj_given_c = {k: round(float(v)/prior_class_prob[int(k.split("#")[1])],5) for k,v in prob_aj_c.items()}
    #prob_aj_c[k.spilt("#")[1]+"#"+k.split("#")[2]+"#"+k.spilt("#")[4]]
    prob_ai_aj_given_c = {k: round(float(v)/prior_class_prob[int(k.split("#")[2])],5) for k,v in prob_ai_aj_c.items()}
    #prob_ai_given_aj_c = {k: round((float(v)+(1.0/X.shape[1]))/((find_prob_ai_c(X,y,float(k.split("#")[1]),int(k.split("#")[2]),int(k.split("#")[3]),prior_class_prob[int(k.split("#")[2])]))[0]+1.0),5) 
    prob_ai_given_aj_c = {k: round((float(v))/(prob_aj_c[k.split("#")[1]+"#"+k.split("#")[2]+"#"+k.split("#")[4]]),5) for k,v in prob_ai_aj_c.items()}

 
    
    return prob_ai_aj_c,prob_ai_given_c,prob_aj_given_c,prob_ai_aj_given_c,prob_ai_given_aj_c




def condtional_mutual_info(X,y,key_pair,prob_i_j_c,prob_i_j_given_c,prob_i_c,prob_j_c):

    #make faster after

    IP_matrix = np.zeros((X.shape[1],X.shape[1]),dtype=np.float)
    #print IP_matrix
    #exit(0)

    IP_dict = {}

    key_pair_set = set(key_pair)
    key_pair_set_list = list(key_pair_set)
    """
    for key_p in key_pair:

        key_list = key_p.split("#") #[ai,aj,c,ai_index,aj_index]

        key_IP_dict_i = int(key_list[3])
        key_IP_dict_j = int(key_list[4])
        key_ai_c = key_list[0]+"#"+key_list[2]+"#"+key_list[3]
        key_aj_c = key_list[1]+"#"+key_list[2]+"#"+key_list[4]
        val_n = float(prob_i_j_given_c[key_p])
        val_d = float(prob_i_c[key_ai_c] * prob_j_c[key_aj_c])
        value_IP_dict = round(prob_i_j_c[key_p] * np.log(val_n/val_d),5)

        IP_matrix[key_IP_dict_i][key_IP_dict_j]+= value_IP_dict


        if(not(key_IP_dict in IP_dict)):
            IP_dict.update({key_IP_dict:value_IP_dict})
        else:
            IP_dict[key_IP_dict]+= value_IP_dict

    #print IP_matrix
    #exit(0)
    #return IP_matrix
    """


    for ai in range(X.shape[1]-1):
        for aj in range(ai+1,X.shape[1]):
            each_sum = 0
            for example in range(X.shape[0]):
                Ai = X[example][ai]
                Aj = X[example][aj]
                c = y[example]
                key_p = str(Ai)+"#"+str(Aj)+"#"+str(c)+"#"+str(ai)+"#"+str(aj)
                key_ai_c = str(Ai)+"#"+str(c)+"#"+str(ai)
                key_aj_c = str(Aj)+"#"+str(c)+"#"+str(aj)
                val_n = float(prob_i_j_given_c[key_p])
                val_d = float(prob_i_c[key_ai_c] * prob_j_c[key_aj_c])
                value_IP_dict = round(prob_i_j_c[key_p] * np.log(val_n/val_d),5)
                each_sum+=value_IP_dict
                #prob1,prob_ai_given_c = find_prob_ai_c(X,y,Ai,c,ai,prob_c[int(c)])
                #prob2,prob_aj_given_c = find_prob_ai_c(X,y,Aj,c,aj,prob_c[int(c)])
                #each_sum += round((prob_i_j_c[key] * np.log(float(prob_i_j_given_c[key])/(prob_ai_given_c*prob_aj_given_c))),5)
            IP_matrix[ai,aj] = each_sum
            IP_matrix[aj,ai] = each_sum
    #print "Matrix"
    #print IP_matrix
    #exit(0)
    return IP_matrix


def weight_matrix(ip_matrix):

    #print ip_matrix
    wij_matrix = ip_matrix

    for r in range(wij_matrix.shape[0]):
        wij_matrix[r]/= np.sum(wij_matrix[r])

    #print wij_matrix
    #exit(0)

    return wij_matrix


def prob_ai_hiddenai_c(X_test,W_matrix,prob_ai_given_aj_c,cl):


    product_prob_hidden = np.zeros(X_test.shape[0],dtype=np.float)

    #print X_test
    #exit(0)

    for example in range(X_test.shape[0]):
        prob_each_attribute = np.zeros(X_test.shape[1],dtype=np.float)
        for ai in range(X_test.shape[1]):
            hidden_prob_sum=0
            for aj in range(X_test.shape[1]):
                if(aj!=ai):
                    key = str(X_test[example][ai])+"#"+str(X_test[example][aj])+"#"+str(int(cl))+"#"+str(ai)+"#"+str(aj)
                    if(key in prob_ai_given_aj_c):
                        hidden_prob_sum+= W_matrix[ai][aj] * prob_ai_given_aj_c[key]
                    else:
                        hidden_prob_sum+=0
            prob_each_attribute[ai]=hidden_prob_sum
        #print prob_each_attribute
        #exit(0)
        product_prob_hidden[example] = np.prod(prob_each_attribute)

    #print product_prob_hidden
    #exit(0)
    return product_prob_hidden





def training_calculation(X_train,y_train):
    #print X_train.shape
    #print y_train.shape
    prob_joint_each_assignment = {}
    prob_ai_aj_given_c_each = {}
    prob_ai_given_aj_c_each = {}


    #prob_each_class = predict_prior_prob(y_train)
    key_each_pair = []
    key_aj_c_pair = []
    key_ai_c_pair = []
    #cnt =0
    for example in range(X_train.shape[0]):
        flagj = False
        flagi = False
        for ai in range(X_train.shape[1]):
            for aj in range(X_train.shape[1]):
                #cnt+=1
                if(aj!=ai):
                    Ai = X_train[example][ai]
                    Aj = X_train[example][aj]
                    c = y_train[example]


                    key = str(Ai)+"#"+str(Aj)+"#"+str(c)+ "#"+ str(ai)+"#"+ str(aj)

                    key_aj_c = str(Aj)+"#"+str(c)+ "#"+ str(aj)
                    key_ai_c = str(Ai)+"#"+str(c)+ "#"+ str(ai)
                    key_each_pair.append(key)
                    if((key_aj_c not in key_aj_c_pair) or flagj==False):
                        key_aj_c_pair.append(key_aj_c)
                        flagj=True
                    if((key_ai_c not in key_ai_c_pair) or  flagi==False):
                        key_ai_c_pair.append(key_ai_c)
                        flagi = True

   
    return key_each_pair,key_aj_c_pair,key_ai_c_pair

    

def predicted_y(membership_fn,y):
    member_list=[]
    for d in membership_fn:
        member_list.append(membership_fn[d])
    max_gjx = np.maximum.reduce(member_list)

    #print max_gjx

    y_predicted = np.zeros(len(max_gjx))
    for y_in in range(len(max_gjx)):
        for key in membership_fn:
            value_dic = membership_fn[key]
            if(max_gjx[y_in]== value_dic[y_in]):
                y_predicted[y_in] = key
        #print y_predicted
    return y_predicted


def classify_X(X,y,fold):

    start_time = time.time()
    kf = KFold(X.shape[0],n_folds=fold,shuffle=True)
    current_fold = 1
    all_fold_parameter = {}
    all_para_inbuilt = {}
    p_r_curve = {}
    for train_index, test_index in kf:

       #if(current_fold==11):
       #.    break
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

       prior_class_probability = predict_prior_prob(y_train)
       key_pair,key_aj_c_pair,key_ai_c_pair = training_calculation(X_train,y_train)
       #print len(key_pair)
       #exit(0)
       prob_joint_each_assignment,prob_ai_given_c,prob_aj_given_c,prob_ai_aj_given_c_each,prob_ai_given_aj_c_each=all_probability_dict(key_pair,key_aj_c_pair,key_ai_c_pair,X_train,y_train,prior_class_probability)

       #prob_joint_each_assignment,prob_ai_aj_given_c_each,prob_ai_given_aj_c_each = training_calculation(X_train,y_train)


       #prior_class_probability = predict_prior_prob(y_train)

       IP_matrix = condtional_mutual_info(X_train,y_train,key_pair,prob_joint_each_assignment,prob_ai_aj_given_c_each,prob_ai_given_c,prob_aj_given_c)

       Wij_matrix = weight_matrix(IP_matrix)


       max_hidden_probs = {}

       for cl in np.unique(y):
           #prior_prob = predict_prior_prob(y_train,cl)
           #sigma = calculate_sigma(X_train,y_train,cl)
           member_fun = prob_ai_hiddenai_c(X_test,Wij_matrix,prob_ai_given_aj_c_each,cl)
           max_hidden_probs.update({int(cl):member_fun})
       #print max_gjx

       y_pridct = predicted_y(max_hidden_probs,y)
       """print
       print
       print "Accuracy : ",accuracy_score(y_test,y_pridct),"   or  ", accuracy_score(y_test,y_pridct)*100," %"
       print
       print
       print "-" * 100
       exit(0)"""

       """print "MY - TEST "
       print
       print y_test
       print
       print "PREDICT"
       print
       print y_pridct
       exit(0)"""

       clf = MultinomialNB()
       clf.fit(X_train,y_train)


       conf_matrix =nD.make_canfusion_matrix(y_test,y_pridct,y)
       #print (y_pridct==y_test).sum()
       c_m = confusion_matrix(np.array(y_test),np.array(y_pridct))  # in-built function of confusion matrix.
       all_para = nD.parameter_calculation(c_m,y,y_test,y_pridct)

       y_pridct_inb = clf.predict(X_test)
       all_parameter = {}
       all_parameter.update({'mse':round((np.average((y_pridct_inb-y_test)**2)),4)})
       all_parameter.update({'accuracy':round(accuracy_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'precision':round(precision_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'recall':round(recall_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'f-measure':round(f1_score(y_test,y_pridct_inb),4)})




       all_fold_parameter.update({current_fold:all_para})
       all_para_inbuilt.update({current_fold:all_parameter})

       current_fold+=1
    print  all_fold_parameter
    #exit(0)
    max_accuracy_fold = max(all_fold_parameter,key = lambda k:float(all_fold_parameter[k]['accuracy']))
    max_accuracy_fold_inbuilt = max(all_para_inbuilt,key = lambda k:float(all_para_inbuilt[k]['accuracy']))

    #max_accuracy_fold_inbuilt = max(all_para_inbuilt,key = lambda k:float(all_para_inbuilt[k]['accuracy']))
    print
    print
    print "-" * 100
    print " " * 15,
    print "n-D k-class Hidden Naive Bayes "
    print "-" * 100

    print
    print "-"*75
    print " "*15,"Parameter Evaluation"
    print "-"*75
    print
    nD.print_parameter(all_fold_parameter[max_accuracy_fold])
    print
    print
    print "-"*75
    print " "*15,"In Built Function Parameter"
    print "-"*75
    print
    print
    nD.print_parameter(all_para_inbuilt[max_accuracy_fold_inbuilt])

    print "-"*75
    print
    print
    print "Total Time  :  ",(time.time()-start_time),"  seconds."
    print "-"*75
    print
    print







if __name__ == '__main__':

    print
    print
    print "-" * 100
    print " " * 15,
    print "n-D k-class Hidden Naive Bayes "
    print "-" * 100


    #url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    #url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
    #url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"

    gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)

    raw_data = urllib.urlopen(url_file,context=gcontext)
    #data_path = "./"
    train_file = np.loadtxt(raw_data,dtype = str, delimiter=' ')#,skiprows=0)
    single_var = np.array(train_file)
    #print single_var
    #X_val = np.array(single_var[:,1:],dtype='float64')
    X_val = np.array(single_var[:,:-1],dtype='float64')
    #given_Y = single_var[:,-1]
    #exit(0)

    y_val = single_var[:,-1]
    uniq_y = np.unique(y_val)

    for i in range(len(y_val)):
        for k in range(len(uniq_y)):
            if(y_val[i]==uniq_y[k]):
                y_val[i]= k

    y_val=np.array(y_val,dtype='int')



    #prob_class = predict_prior_prob(y_val)
    #prob_join,prob2,prob3 = training_calculation(X_val,y_val)
    #print prob3["7.7#3.0#2"]
    #print prob3["3.0#7.7#2"]
    #exit(0)
    #ip_mat = condtional_mutual_info(X_val,y_val,prob_join,prob2,prob_class)
    #w_mat = weight_matrix(ip_mat)
    #print prob_ai_hiddenai_c(X_val[135:,:],w_mat,prob3,0)
    #print prob_ai_hiddenai_c(X_val[135:,:],w_mat,prob3,1)
    #print prob_ai_hiddenai_c(X_val[135:,:],w_mat,prob3,2)

    classify_X(X_val,y_val,10)

    #print
    #print '-' * 100


"""
    #url_file = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-zer"

    #url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-pix"

    gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)

    raw_data = urllib.urlopen(url_file,context=gcontext)

    train_file = np.loadtxt(raw_data,dtype = str)

    single_var = np.array(train_file)
    #print single_var

    document_length = 100

    X_val = np.array(single_var[:,:],dtype='float64')
    #X_val = X_val*document_length

    #X_val = np.array(X_val,dtype=np.int)

    #total_X_val = np.sum(X_val,axis=1)

    #y_val = np.array(single_var[:, -1],dtype=np.int)

    #print type(X_val)
    #print y_val



    y_val = np.zeros(len(X_val),dtype=np.int)

    cnt =0
    for i in range(1,11):
        y_val[cnt:cnt+200]=i
        cnt+=200
    #print type(y_val)
    #exit(0)
    #start_time = time.time();

    #prob_join,prob2,prob3 = tc.training_calculation(X_val,y_val)
    #print len(prob_join)
    #print("--- Cython function: %s seconds ---" % (time.time() - start_time));

    #classify_X(X_val[:570,:],y_val[:570],10)
"""
    #predict_prior_prob(y_val)
