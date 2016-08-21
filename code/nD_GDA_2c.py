import numpy as np
import math as mt
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,f1_score,recall_score,precision_recall_curve,average_precision_score

import urllib

from sklearn.lda import LDA

def precision_recall_plot(all_para_fold,no_class):
    #for i in range(no_class):
    recall = all_para_fold['recall']

    precision = all_para_fold['precision']
    average_precision = all_para_fold['average_precision']

    plt.plot(recall, precision,
             label='Precision-recall curve (area = {1:0.2f})'
                   ''.format(0, average_precision))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()



def calculate_mean(X,y):

    mean_vector= {}

    for cl in np.unique(y):
        mean_vector.update({int(cl):np.mean(X[y==cl],axis=0)})
    #print mean_vector
    #exit(0)
    return mean_vector

def calculate_sigma(X,y,class_value):

    class_mean = calculate_mean(X,y)
    sigma_vector = np.zeros((X.shape[1],X.shape[1]),dtype='float64')
    X = X[y==class_value]
    #print (X[:,0:1] - class_mean[0][0]).shape
    #print class_mean[0][0]
    #exit(0)

    #print sigma_vector

    for r in range(X.shape[1]):
        for c in range(X.shape[1]):
            sigma_vector[r,c] = np.mean((X[:,r:r+1]-class_mean[class_value][r]) * ((X[:,c:c+1] - class_mean[class_value][c])))
    #print sigma_vector

    return sigma_vector

def predict_prior_prob(y,class_value):
    """
    y_prob = []
    for cl in np.unique(y):
        y_prob.append((y==cl).sum()/(y.shape[0]))

    return np.array(y_prob)"""

    prob_y =float((y==class_value).sum())/y.shape[0]

    return prob_y


def membership_function(X_test,y_test,class_value,mean,sigma,prior_probability):

    determinant = (-0.5)*mt.log(abs(np.linalg.det(sigma)))
    X_test = np.matrix(X_test)
    y_test = np.matrix(y_test)

    gjx = np.zeros(X_test.shape[0])

    for i in range(X_test.shape[0]):
        gjx[i] = determinant - ((0.5)*((X_test[i] - mean) * sigma**-1 * (X_test[i] - mean).T)) + mt.log(prior_probability)

    return gjx

def discriminant_function(membership,y):

    #membership = classify_X(X,y,fold)
    #if(len(np.unique(y))==2):
    d_x = membership[np.unique(y)[1]] - membership[np.unique(y)[0]]


    return d_x

def predicted_y(discri_function,membership_fn,y):

    #discri_function = discriminant_function(X,y,fold)


    if(len(np.unique(y))==2):
        y_predicted = np.zeros(len(discri_function))
        for i in range(len(y_predicted)):
            if(discri_function[i]>0):
                y_predicted[i] = np.unique(y)[1]
            else:
                y_predicted[i] = np.unique(y)[0]
        #print y_predicted
        return y_predicted

    else:

        #membership_fn = classify_X(X,y,fold)
        #print membership_fn

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


def make_canfusion_matrix(y_test,y_pridct,y):

    mat_size = len(np.unique(y))
    conf_array = np.zeros((mat_size,mat_size),dtype=np.int)
    confusion_matrix =np.matrix(conf_array)
    """if(len(np.unique(y))==3):
        for i in range(len(y_pridct)):
             if(y_pridct[i]==np.unique(y)[0] and y_test[i]==np.unique(y)[0]):
                confusion_matrix[0,0]+=1
             elif(y_pridct[i]==np.unique(y)[1] and y_test[i]==np.unique(y)[0]):
                 confusion_matrix[0,1]+=1
             if(y_pridct[i]==np.unique(y)[1] and y_test[i]==np.unique(y)[1]):
                confusion_matrix[1,1]+=1
             elif(y_pridct[i]==np.unique(y)[0] and y_test[i]==np.unique(y)[1]):
                 confusion_matrix[1,0]+=1"""
    unique_y = np.unique(y)
    for i in range(len(y_pridct)):
        for k in range(len(unique_y)):
            for j in range(len(unique_y)):
                if(y_pridct[i]==unique_y[k] and y_test[i]==unique_y[j]):
                    confusion_matrix[j,k]+=1

    #print confusion_matrix
    #exit(0)
    return confusion_matrix

def parameter_calculation(confusion_mat,train_file,y_test,y_pridct):

    """if(confusion_mat.shape[0]==3 and confusion_mat[1]==2):
        TP = confusion_mat[0,0]
        FN = confusion_mat[0,1]
        FP = confusion_mat[1,0]
        TN = confusion_mat[1,1]

        Accuracy = round(float((TP+TN))/(TN+FP+FN+TP),4)
        Precision = round(float(TP)/(TP+FP),4)
        #if((TP+FN)!=0):
        Recall = round(float(TP)/(TP+FN),4)
        F_measure = round(float(2*(Precision*Recall))/(Precision+Recall),4)
        #else:
        #    Recall = 0.0
        #    F_measure = 0.0

        all_parameter = {}
        all_parameter.update({'accuracy':Accuracy})
        all_parameter.update({'precision':Precision})
        all_parameter.update({'recall':Recall})
        all_parameter.update({'f-measure':F_measure})

        return all_parameter"""


    uniq_y = np.unique(train_file[:,-1])
    MSE = round(np.average((y_pridct-y_test)**2),4)
    Accuracy = round(float(np.matrix.trace(confusion_mat))/np.sum(confusion_mat),4)
    Precision = {}#np.zeros(confusion_mat.shape[0])
    Recall = {}#np.zeros(confusion_mat.shape[0])
    F_measure = {} #np.zeros(confusion_mat.shape[0])


    for i in range(confusion_mat.shape[0]):

        """Precision[i] = round(float(confusion_mat[i,i])/np.sum(confusion_mat[:,i]),4)
        Recall[i] = round(float(confusion_mat[i,i])/np.sum(confusion_mat[i,:]),4)
        F_measure[i] = round(2 * (Precision[i] * Recall[i])/(Precision[i] + Recall[i]),4)"""

        if(np.sum(confusion_mat[:,i])!=0):
            Precision.update({uniq_y[i]:round(float(confusion_mat[i,i])/np.sum(confusion_mat[:,i]),4)})
        else:
            Precision.update({uniq_y[i]:0.0})
        if(np.sum(confusion_mat[i,:])!=0):
            Recall.update({uniq_y[i]:round(float(confusion_mat[i,i])/np.sum(confusion_mat[i,:]),4)})
        else:
            Recall.update({uniq_y[i]:0.0})
        if((Precision[uniq_y[i]] + Recall[uniq_y[i]])!=0):
            F_measure.update({uniq_y[i]:round(2 * (Precision[uniq_y[i]] * Recall[uniq_y[i]])/(Precision[uniq_y[i]] + Recall[uniq_y[i]]),4)})
        else:
            F_measure.update({uniq_y[i]:0.0})


    all_parameter = {}
    all_parameter.update({'mse':MSE})
    all_parameter.update({'accuracy':Accuracy})
    all_parameter.update({'precision':Precision})
    all_parameter.update({'recall':Recall})
    all_parameter.update({'f-measure':F_measure})

    return all_parameter


def print_parameter(all_parameter):


    print "MSE :: ",all_parameter['mse'],"\t\tOR\t\t",100*all_parameter['mse']," %"
    print
    print "Accuracy :: ",all_parameter['accuracy'],"\tOR\t\t",100*all_parameter['accuracy']," %"
    print
    print "Precision :: ",all_parameter['precision']
    print
    print "Recall :: ",all_parameter['recall']
    print
    print "F-measure :: ",all_parameter['f-measure']

def classify_X(X,y,fold):

    kf = KFold(X.shape[0],n_folds=fold,shuffle=True)
    current_fold = 1
    all_fold_parameter = {}
    all_para_inbuilt = {}
    p_r_curve = {}
    for train_index, test_index in kf:

       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

       mean_all_class = calculate_mean(X_train,y_train)
       max_gjx = {}

       for cl in np.unique(y):
           prior_prob = predict_prior_prob(y_train,cl)
           sigma = calculate_sigma(X_train,y_train,cl)
           member_fun = membership_function(X_test,y_test,cl,mean_all_class[cl],sigma,prior_prob)
           max_gjx.update({int(cl):member_fun})
       #print max_gjx
       discri_function = discriminant_function(max_gjx,y)
       y_pridct = predicted_y(discri_function,max_gjx,y)

       conf_matrix =make_canfusion_matrix(y_test,y_pridct,y)
       #print (y_pridct==y_test).sum()
       c_m = confusion_matrix(np.array(y_test),np.array(y_pridct))  # in-built function of confusion matrix.
       all_para = parameter_calculation(conf_matrix,train_file,y_test,y_pridct)

       if(len(np.unique(y))==2):
           precision,recall,thresold = precision_recall_curve(y_test,y_pridct)
           average_precision = average_precision_score(y_test,y_pridct)
           p_r_curve_fold ={}
           p_r_curve_fold.update({'precision':precision})
           p_r_curve_fold.update({'recall':recall})
           p_r_curve_fold.update({'average_precision':average_precision})


           p_r_curve.update({int(current_fold):p_r_curve_fold})

       """
       lda = LDA(solver="svd", store_covariance=True)
       lda.fit(X_train, y_train)
       y_pridct_inb = lda.predict(X_test)
       all_parameter = {}
       all_parameter.update({'mse':round((np.average((y_pridct_inb-y_test)**2)),4)})
       all_parameter.update({'accuracy':round(accuracy_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'precision':round(precision_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'recall':round(recall_score(y_test,y_pridct_inb),4)})
       all_parameter.update({'f-measure':round(f1_score(y_test,y_pridct_inb),4)})
        """



       all_fold_parameter.update({current_fold:all_para})
       #all_para_inbuilt.update({current_fold:all_parameter})

       current_fold+=1
    
    max_accuracy_fold = max(all_fold_parameter,key = lambda k:float(all_fold_parameter[k]['accuracy']))

    #max_accuracy_fold_inbuilt = max(all_para_inbuilt,key = lambda k:float(all_para_inbuilt[k]['accuracy']))
    print
    print "-"*75
    print " "*15,"Parameter Evaluation"
    print "-"*75
    print
    print_parameter(all_fold_parameter[max_accuracy_fold])

    if(len(np.unique(y))==2):
        return p_r_curve[max_accuracy_fold]





if __name__ == '__main__':

    print "-" * 100
    print " " * 15,
    print "1-D 2-class Gaussian Discriminant Analysis "
    print "-" * 100

    url_file = "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    #url_file = "http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"
    #url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    raw_data = urllib.urlopen(url_file)
    #data_path = "./"
    train_file = np.loadtxt(raw_data,dtype = str, delimiter=',',skiprows=1)

   # print train_file
    #exit(0)

    single_var = np.array(train_file)
    #print single_var
    X_val = np.array(single_var[:,0:-1],dtype='float64')
    #given_Y = single_var[:,-1]
    #exit(0)
    y_val = np.array(single_var[:, -1],dtype=np.int)
    #print X_val
    #print y_val"""


    #calculate_mean(X_val,y_val)
    #calculate_sigma(X_val,y_val,class_value=0)
    #classify_X(X_val,y_val,10)
    #discriminant_function(X_val,y_val,fold=10)
    pr_curve = classify_X(X_val,y_val,10)
    #precision_recall_plot(pr_curve,len(np.unique(y_val)))

    #predicted_y(X_val,y_val,fold=10)
    #print calculate_sigma(X_val,y_val,class_value=0)
    print
    print
    print "-" * 100
    print " " * 15,
    print "n-D 2-class Gaussian Discriminant Analysis "
    print "-" * 100

    url_file = "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    raw_data = urllib.urlopen(url_file)
    #data_path = "./"
    train_file = np.loadtxt(raw_data,dtype = str, delimiter=',',skiprows=1)
    single_var = np.array(train_file)
    #print single_var
    X_val = np.array(single_var[:,0:-1],dtype='float64')
    #given_Y = single_var[:,-1]
    #exit(0)
    y_val = np.array(single_var[:, -1],dtype=np.int)

    pr_curve = classify_X(X_val,y_val,10)
    precision_recall_plot(pr_curve,len(np.unique(y_val)))
    print
    print
    print "-" * 100
    print " " * 15,
    print "n-D k-class Gaussian Discriminant Analysis "
    print "-" * 100

    url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    raw_data = urllib.urlopen(url_file)
    #data_path = "./"
    train_file = np.loadtxt(raw_data,dtype = str, delimiter=',',skiprows=1)
    single_var = np.array(train_file)
    #print single_var
    X_val = np.array(single_var[:,0:-1],dtype='float64')
    #given_Y = single_var[:,-1]
    #exit(0)

    y_val = single_var[:,-1]
    uniq_y = np.unique(y_val)

    for i in range(len(y_val)):
        for k in range(len(uniq_y)):
            if(y_val[i]==uniq_y[k]):
                y_val[i]= k

    y_val=np.array(y_val,dtype='int')



    classify_X(X_val,y_val,10)

    print
    print '-' * 100
