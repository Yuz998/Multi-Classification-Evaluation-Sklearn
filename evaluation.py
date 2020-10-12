import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn import metrics
from scipy.ndimage.morphology import binary_erosion


def eval(y_true, y_scores, output_folder, cutoff, print_flag=True):
    '''
    y_true:    The true values of multiple classifications
    y_scores:  Multiple classifications prediction scores  example：y_scores = model.predict_proba(X_test)
    return:
            1. AUROC
            2. AUPRC
            3. Confusion matrix
            4. Classification report
            5. Jaccard similarity score
            6. accuracy
            7. sensitivity
            8. specificity
            9. precision
            10. F1 score
    '''
    y_scores_ = y_scores[:, 1]
    y_scores = np.where(y_scores_ > cutoff, 1, 0)

    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    print ("\nArea under the ROC curve: " +str(AUC_ROC))
    fig =plt.figure()
    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(output_folder+"ROC.png")

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
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
    plt.savefig(output_folder+"Precision_recall.png")

    ## Calculate classification report
    print("\nclassification report: ")
    classification_report = metrics.classification_report(y_true, y_scores, labels=[0,1])
    print(classification_report)

    #Confusion matrix
    threshold_confusion = cutoff
    print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i]>=threshold_confusion:
            y_pred[i]=1
        else:
            y_pred[i]=0
    confusion = confusion_matrix(y_true, y_pred)
    print (confusion)
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print ("Global Accuracy: " +str(accuracy))
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print ("Specificity: " +str(specificity))
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print ("Sensitivity: " +str(sensitivity))
    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
    print ("Precision: " +str(precision))

    #Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print ("\nJaccard similarity score: " +str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print ("\nF1 score (F-measure): " +str(F1_score))

    # Save the results
    if print_flag:
        file_perf = open(output_folder+'performances.txt', 'w')
        file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                        + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                        + "\nJaccard similarity score: " +str(jaccard_index)
                        + "\nF1 score (F-measure): " +str(F1_score)
                        +"\n\nConfusion matrix:"
                        +str(confusion)
                        +"\nClassification report:"
                        +str(classification_report)
                        +"\nACCURACY: " +str(accuracy)
                        +"\nSENSITIVITY: " +str(sensitivity)
                        +"\nSPECIFICITY: " +str(specificity)
                        +"\nPRECISION: " +str(precision)
                        )
        file_perf.close()

