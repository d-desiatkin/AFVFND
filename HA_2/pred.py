#======================================================
#===================pred.py============================
#======================================================

#Adapted from https://github.com/uclmr/fakenewschallenge/blob/master/pred.py
#Original credit - @jaminriedel 

# Import relevant packages and modules
from util import FNCData,bow_train,pipeline_train,pipeline_test,save_predictions
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from score import report_score
from sklearn.utils.class_weight import compute_class_weight

# Prompt for mode
mode = input('mode (load / train)? ')

# Initialise hyperparameters
r = random.Random()
r.seed(123)
lim_unigram = 5000
target_size = 4
hidden_size = 100

train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90
print('train_keep_prob=',train_keep_prob,'clip_ratio=',clip_ratio,'batch_size_train=',batch_size_train)
# Set file names
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "test_stances_unlabeled.csv"
file_test_bodies = "test_bodies.csv"
file_predictions = 'predictions_test.csv'
base_dir='./'
#base_dir='split-data' #change to this directory when doing data-splitting tasks or K-fold cross validation
# Load data sets

raw_train = FNCData(base_dir,file_train_instances, file_train_bodies)
raw_test = FNCData(base_dir,file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)#the total number of entry instances

bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = bow_train(raw_train, raw_test, lim_unigram=lim_unigram)

#===============================
# Define model
# Adapted from https://github.com/uclmr/fakenewschallenge/blob/master/pred.py
# Original credit - @jaminriedel 
#==============================
def model(dataset_number):
    
    # Process data sets
    train_set, train_stances = pipeline_train(dataset_number,raw_train,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    test_set = pipeline_test(dataset_number,raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    feature_size = len(train_set[0])
    
    #to clear the existed nodes for new model training
    tf.reset_default_graph() 

    # Create placeholders
    features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
    stances_pl = tf.placeholder(tf.int64, [None], 'stances')
    weights_pl = tf.placeholder(tf.float32, [None], 'weights')
    keep_prob_pl = tf.placeholder(tf.float32)

    # Infer batch size
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.contrib.layers.fully_connected(features_pl, hidden_size,weights_initializer=tf.contrib.layers.xavier_initializer(seed=100)), seed=101, keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size,weights_initializer=tf.contrib.layers.xavier_initializer(seed=102)),seed=103,keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, target_size])#reshape to be (batch_size*4)
    

    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha
    
    # Define cross entropy loss with weights
    cr_ent_loss = tf.math.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=stances_pl, logits=logits), weights_pl)

    # Define overall loss
    loss = tf.reduce_sum(cr_ent_loss + l2_loss)

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    #predict = tf.argmax(softmaxed_logits, 1)
    predict=softmaxed_logits
    # Define optimiser
    opt_func = tf.train.GradientDescentOptimizer(learn_rate)
    #opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))
    
    # Perform training
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        for epoch in range(epochs):
            total_loss = 0
            #average_acc = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [train_set[i] for i in batch_indices]
                batch_stances = [train_stances[i] for i in batch_indices]
                classes = np.unique(batch_stances)
                #tmp = compute_class_weight('balanced', classes, batch_stances)
                batch_weights = np.array(batch_stances)
                batch_weights=(batch_weights==1)*1000+np.ones((batch_weights.shape))
                #print(classes.shape[0])
                #for o in range(classes.shape[0]):
                  #batch_weights += (batch_stances == classes[o])*tmp[o]
                #print(batch_stances)
                #print(batch_weights)

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, weights_pl: batch_weights, keep_prob_pl: train_keep_prob}
                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss
                #average_acc += current_accuracy
            #average_acc = average_acc / (n_train // batch_size_train)
            
            print('epoch',epoch,'loss',total_loss)
        
        #=====Save the Checkpoints===============================
        saver.save(sess, './model/model%d/mymodel'%dataset_number)
        #=========================================================
        
        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)       
        return test_pred
#======================
# Restore model 
#======================
def restore_model(model_num):
    
    # Define graph
    tf.reset_default_graph()
    test_set = pipeline_test(model_num,raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    feature_size = len(test_set[0])

    # Create placeholders
    features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
    stances_pl = tf.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.placeholder(tf.float32)

    # Infer batch size
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, target_size])

    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    # Define overall loss
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    predict = softmaxed_logits
    
    with tf.Session() as sess:
                
        saver = tf.train.Saver()
        saver.restore(sess,base_dir+'/model/model%d/mymodel'%model_num)
        
        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)

    return test_pred
#=====================   
# Load model
#=====================
if mode == 'load':
    Grade=[]
    Agree_precision=[]
    Disagree_precision=[]
    Discuss_precision=[]
    Unrelated_precision=[]
    Agree_recall=[]
    Disagree_recall=[]
    Discuss_recall=[]
    Unrelated_recall=[]
    Recall=[]
    
    F1_Agree=[]
    F1_Disagree=[]
    F1_Discuss=[]
    F1_Unrelated=[]
    F1_m = []	
    #Define the weight for different class if needed
    #weight_pred_1=np.diag(np.ones(4))
    #weight_pred_1[0][0]=2
    #weight_pred_2=np.diag(np.ones(4))
    #weight_pred_2[2][2]=2
    #weight_pred_3=np.diag(np.ones(4))
    #weight_pred_3[3][3]=2
    weight_pred_1=np.diag(np.array([1, 1.154, 1.042, 1.003]))
    weight_pred_2=np.diag(np.array([1.023, 1.415, 1.041, 1]))
    weight_pred_3=np.diag(np.array([1.253, 1, 1, 1.010]))
    
    
    test_prediction1=restore_model(1)
    test_prediction2=restore_model(2) 
    test_prediction3=restore_model(3) 
    
    #=====ensemble for two========
    #final_pred=np.concatenate((np.matmul(test_prediction1,weight_pred_1),np.matmul(test_prediction2,weight_pred_2)),axis=1)
    #final_pred=np.matmul(test_prediction1,weight_pred_1)+np.matmul(test_prediction2,weight_pred_2)
    #======ensemble for three======
    #Concatenation approach
    #final_pred=np.concatenate((np.matmul(test_prediction1,weight_pred_1),np.matmul(test_prediction2,weight_pred_2),np.matmul(test_prediction3,weight_pred_3)),axis=1)
    #Summation approach
    final_pred=np.matmul(test_prediction1,weight_pred_1)+np.matmul(test_prediction2,weight_pred_2)+np.matmul(test_prediction3,weight_pred_3)
    #=======no ensemble============
    #final_pred=test_prediction1 
    #==============================
    
    final_pred_index = np.argmax(final_pred,1) 
    save_predictions(base_dir,final_pred_index, file_predictions)
   
    # Calculate score
    golden_stance = pd.read_csv(base_dir+"/"+"test_stances_labeled.csv")
    prediction_stance=pd.read_csv(base_dir+"/"+"predictions_test.csv")
    
    competition_grade,agree_recall,disagree_recall,discuss_recall,unrelated_recall, agree_precision, disagree_precision, discuss_precision, unrelated_precision,all_recall, f1_agree, f1_disagree, f1_discuss, f1_unrelated, F1m=report_score(golden_stance['Stance'],prediction_stance['Prediction'])
    
    Grade.append(competition_grade)
    
    
    Agree_recall.append(agree_recall)
    Disagree_recall.append(disagree_recall)
    Discuss_recall.append(discuss_recall)
    Unrelated_recall.append(unrelated_recall)
    
    Agree_precision.append(agree_precision)
    Disagree_precision.append(disagree_precision)
    Discuss_precision.append(discuss_precision)
    Unrelated_precision.append(unrelated_precision)
    
    Recall.append(all_recall)
    
    F1_Agree.append(f1_agree)
    F1_Disagree.append(f1_disagree)
    F1_Discuss.append(f1_discuss)
    F1_Unrelated.append(f1_unrelated)
    F1_m.append(F1m)
    
    # Save the performance to csv
#    df = pd.DataFrame({"Grade" : np.array(Grade), "Agree" : np.array(Agree),"Disagree" : np.array(Disagree),"Discuss" : np.array(Discuss),"Unrelated" : np.array(Unrelated),"Recall" : np.array(Recall)})
#    df.to_csv(base_dir+'/'+"Performance.csv", index=False)
    
    print('Grade',Grade)
    print('Agree recall',Agree_recall)
    print('Agree precision',Agree_precision)
    print('Agree f1',F1_Agree)
    print('Disagree recall',Disagree_recall)
    print('Disagree precision',Disagree_precision)
    print('Disagree f1',F1_Disagree)
    print('Discuss recall',Discuss_recall)
    print('Discuss precision',Discuss_precision)
    print('Discuss f1',F1_Discuss)
    print('Unrelated recall',Unrelated_recall)
    print('Unrelated precision',Unrelated_precision)
    print('Unrelated f1',F1_Unrelated)
    print('Accuracy',Recall)
    print('F1m', F1_m)
    
#===========================	
# Train model
#===========================
if mode == 'train':

    Grade=[]
    Agree_precision=[]
    Disagree_precision=[]
    Discuss_precision=[]
    Unrelated_precision=[]
    Agree_recall=[]
    Disagree_recall=[]
    Discuss_recall=[]
    Unrelated_recall=[]
    Recall=[]
    
    F1_Agree=[]
    F1_Disagree=[]
    F1_Discuss=[]
    F1_Unrelated=[]
    F1_m = []
    
    # dataset=1: baseline feature [head_tf, body_tf, tfidf_cos]
    # dataset=2: baseline+refuting words feature
    # dataset=3: baseline+mutual information words feature
    # dataset=4: baseline+word2vec distance feature
    # dataset=5: baseline+wmd distance feature
    # dataset=6: baseline+combining feature from 2 to 5
    
    test_prediction1=model(dataset_number=1) 
    test_prediction2=model(dataset_number=2)
    test_prediction3=model(dataset_number=3)
    
    #=======Define the weights for different categories if needed======
    weight_pred_1=np.diag(np.ones(4))
    #weight_pred_1[0][0]=2
    weight_pred_2=np.diag(np.ones(4))
    #weight_pred_2[2][2]=2
    weight_pred_3=np.diag(np.ones(4))
    #weight_pred_3[3][3]=2 
    #===================================================================
    
    #=====ensemble for two========
    #final_pred=np.concatenate((np.matmul(test_prediction1,weight_pred_1),np.matmul(test_prediction2,weight_pred_2)),axis=1)
    #final_pred=np.matmul(test_prediction1,weight_pred_1)+np.matmul(test_prediction2,weight_pred_2)
    #======ensemble for three======
    #final_pred=np.concatenate((np.matmul(test_prediction1,weight_pred_1),np.matmul(test_prediction2,weight_pred_2),np.matmul(test_prediction3,weight_pred_3)),axis=1)
    final_pred=np.matmul(test_prediction1,weight_pred_1)+np.matmul(test_prediction2,weight_pred_2)+np.matmul(test_prediction3,weight_pred_3)
    #=======no ensemble============
    #final_pred=test_prediction1 
    #==============================================
    
    final_pred_index = np.argmax(final_pred,1)

    # Save predictions
    save_predictions(base_dir,final_pred_index, file_predictions)
    
    # Calculate score
    golden_stance = pd.read_csv(base_dir+"/"+"test_stances_labeled.csv")
    prediction_stance=pd.read_csv(base_dir+"/"+"predictions_test.csv")
    
    competition_grade,agree_recall,disagree_recall,discuss_recall,unrelated_recall, agree_precision, disagree_precision, discuss_precision, unrelated_precision,all_recall, f1_agree, f1_disagree, f1_discuss, f1_unrelated, F1m=report_score(golden_stance['Stance'],prediction_stance['Prediction'])
    
    Grade.append(competition_grade)
    
    
    Agree_recall.append(agree_recall)
    Disagree_recall.append(disagree_recall)
    Discuss_recall.append(discuss_recall)
    Unrelated_recall.append(unrelated_recall)
    
    Agree_precision.append(agree_precision)
    Disagree_precision.append(disagree_precision)
    Discuss_precision.append(discuss_precision)
    Unrelated_precision.append(unrelated_precision)
    
    Recall.append(all_recall)
    
    F1_Agree.append(f1_agree)
    F1_Disagree.append(f1_disagree)
    F1_Discuss.append(f1_discuss)
    F1_Unrelated.append(f1_unrelated)
    F1_m.append(F1m)
    
    # Save the performance to csv
#    df = pd.DataFrame({"Grade" : np.array(Grade), "Agree" : np.array(Agree),"Disagree" : np.array(Disagree),"Discuss" : np.array(Discuss),"Unrelated" : np.array(Unrelated),"Recall" : np.array(Recall)})
#    df.to_csv(base_dir+'/'+"Performance.csv", index=False)
    
    print('Grade',Grade)
    print('Agree recall',Agree_recall)
    print('Agree precision',Agree_precision)
    print('Agree f1',F1_Agree)
    print('Disagree recall',Disagree_recall)
    print('Disagree precision',Disagree_precision)
    print('Disagree f1',F1_Disagree)
    print('Discuss recall',Discuss_recall)
    print('Discuss precision',Discuss_precision)
    print('Discuss f1',F1_Discuss)
    print('Unrelated recall',Unrelated_recall)
    print('Unrelated precision',Unrelated_precision)
    print('Unrelated f1',F1_Unrelated)
    print('Accuracy',Recall)
    print('F1m', F1_m)
    
    
    
