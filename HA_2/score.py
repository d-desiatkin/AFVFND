#======================================================
#===================score.py==============
#======================================================

#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith
import numpy as np


LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = np.array([[0, 0, 0, 0],
          			[0, 0, 0, 0],
          			[0, 0, 0, 0],
         			[0, 0, 0, 0]])

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance), LABELS.index(t_stance)] += 1
        
    if np.sum(cm[0,:])==0:
        agree_recall=0
    else:
        agree_recall=cm[0, 0]/np.sum(cm[0,:])
    
    if np.sum(cm[1,:])==0:
        disagree_recall=0
    else:
        disagree_recall=cm[1, 1]/np.sum(cm[1,:])
    
    if np.sum(cm[2,:])==0:
        discuss_recall=0
    else:
        discuss_recall=cm[2, 2]/np.sum(cm[2,:])
        
    if np.sum(cm[3,:])==0:
        unrelated_recall=0
    else:    
        unrelated_recall=cm[3, 3]/np.sum(cm[3,:])
    #===========================================================================================    
    if np.sum(cm[:,0])==0:
        agree_precision=0
    else:
        agree_precision=cm[0, 0]/np.sum(cm[:,0])
    
    if np.sum(cm[:,1])==0:
        disagree_precision=0
    else:
        disagree_precision=cm[1, 1]/np.sum(cm[:,1])
    
    if np.sum(cm[:,2], 0)==0:
        discuss_precision=0
    else:
        discuss_precision=cm[2, 2]/np.sum(cm[:,2])
        
    if np.sum(cm[:,3], 0)==0:
        unrelated_precision=0
    else:    
        unrelated_precision=cm[3, 3]/np.sum(cm[:,3])
        
    if np.sum(np.sum(cm, 1), 0)==0:
        all_recall=0
    else:
        all_recall=(cm[0, 0]+cm[1, 1]+cm[2, 2]+cm[3, 3])/np.sum(np.sum(cm, 1), 0)
    
    f1_agree = 2 * agree_recall * agree_precision / (agree_recall + agree_precision)
    f1_disagree = 2 * disagree_recall * disagree_precision / (disagree_recall + disagree_precision)
    f1_discuss = 2 * discuss_recall * discuss_precision / (discuss_recall + discuss_precision)
    f1_unrelated = 2 * unrelated_recall * unrelated_precision / (unrelated_recall + unrelated_precision)
    F1m = (f1_agree + f1_disagree + f1_discuss + f1_unrelated)/4
    
    return score,cm,agree_recall,disagree_recall,discuss_recall,unrelated_recall, agree_precision, disagree_precision, discuss_precision, unrelated_precision, all_recall, f1_agree, f1_disagree, f1_discuss, f1_unrelated, F1m


def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))


def report_score(actual,predicted):
    score,cm,agree_recall,disagree_recall,discuss_recall,unrelated_recall, agree_precision, disagree_precision, discuss_precision, unrelated_precision,all_recall, f1_agree, f1_disagree, f1_discuss, f1_unrelated, F1m = score_submission(actual,predicted)
    best_score, _, _,_,_,_, _,_,_,_, _, _,_,_,_,_= score_submission(actual,actual)

    #print_confusion_matrix(cm)
    #print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    competition_grade=score*100/best_score
    
    return competition_grade,agree_recall,disagree_recall,discuss_recall,unrelated_recall, agree_precision, disagree_precision, discuss_precision, unrelated_precision, all_recall, f1_agree, f1_disagree, f1_discuss, f1_unrelated, F1m


if __name__ == "__main__":
    actual = [0,0,0,0,1,1,0,3,3]
    predicted = [0,0,0,0,1,1,2,3,3]

    report_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])
