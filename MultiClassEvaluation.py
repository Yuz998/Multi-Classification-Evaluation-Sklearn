import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from scipy import interp

def eval(y_true, y_pred, path, n_class, print_flag=True):
    '''
    y_true:    The true values of multiple classifications
    y_pred:    Predictive values of multiple classifications
    y_score:   One hot vector of multiple classifications
    y_one_hot: One hot vector of multiple classifications
    return:
            1. AUC of micro and macro
            2. ROC curves for each classification
            3. Average ROC curve
            4. Average PRC curve
            5. Confusion matrix
            6. Classification report
            7. F1 score
    '''
    np.random.seed(0)

    y_score = np.zeros((len(y_pred), n_class))
    y_one_hot = np.zeros((len(y_true), n_class))
    for i,j in enumerate(y_pred):
        y_score[i][j] = 1
    for i,j in enumerate(y_true):
        y_one_hot[i][j] = 1

    # The function is called to caculate the AUC of type micro and macro
    auc_micro = metrics.roc_auc_score(y_one_hot, y_score, average='micro')
    auc_macro = metrics.roc_auc_score(y_one_hot, y_score, average='macro')
    print('auc micro：', auc_micro)
    print("auc macro: ", auc_macro)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(y_one_hot.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    ## Draw ROC curves for each classification
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    for i in range(n_class):
        fpr, tpr, thresholds = metrics.roc_curve(y_one_hot[:,i], y_score[:,i])
        auc = metrics.auc(fpr, tpr)
        ## Set font and character display
        mpl.rcParams['font.sans-serif'] = u'SimHei'
        mpl.rcParams['axes.unicode_minus'] = False
        # FPR is the X axis,TPR is the Y axis
        plt.plot(fpr, tpr, lw = 2, alpha = 0.7, label = u'ROC curve of class %d (AUC=%.3f)' %(i,auc))
        plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (False Positive Rate)', fontsize=13)
    plt.ylabel('TPR (True Positive Rate)', fontsize=13)
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(path+"multi-class.png")

    # Average ROC curve
    plt.figure()
    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
    auc1 = metrics.auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, lw=2, alpha=0.7, label='Area Under the Curve (area = %0.4f)'%auc1)
    plt.xlabel('FPR (False Positive Rate)', fontsize=13)
    plt.ylabel('TPR (True Positive Rate)', fontsize=13)
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(path+'ROC.png')

    ## Average PRC curve
    plt.figure()
    precision, recall, thresholds = precision_recall_curve(y_one_hot.ravel(), y_score.ravel())
    precision = np.fliplr([precision])[0] 
    recall = np.fliplr([recall])[0]
    AUC_prec_rec = np.trapz(precision,recall)
    print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(path+"PRC.png")

    ## Confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)

    ## Calculate F1 score
    F1_score_micro = f1_score(y_true, y_pred, labels=None, average='micro', sample_weight=None)
    F1_score_macro = f1_score(y_true, y_pred, labels=None, average='micro', sample_weight=None)
    print ("\nF1 score (F-measure) micro: " +str(F1_score_micro))
    print ("\nF1 score (F-measure) macro: " +str(F1_score_macro))

    ## Calculate classification report
    print("\nclassification report: ")
    classification_report = metrics.classification_report(y_true, y_pred, labels=[i for i in range(n_class)])
    print(classification_report)

    ## Save the result
    if print_flag:
        file_perf = open(path+'performances.txt', 'w')
        file_perf.write("micro-average ROC curve: "+str(auc_micro)
                            +"\nmacro-average ROC curve: "+str(auc_macro)
                            +"\nF1 score (F-measure) micro: " +str(F1_score_micro)
                            +"\nF1 score (F-measure) macro: " +str(F1_score_macro)
                            +"\n\nConfusion matrix:\n"
                            +str(confusion)
                            +"\n\nclassification_report:\n"
                            +str(classification_report)
                            )
        file_perf.close()