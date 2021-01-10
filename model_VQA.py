import ipdb
import time
import math
import cv2
import os
import h5py
import sys
import argparse
import codecs
import json
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.models.rnn import rnn_cell
from sklearn.metrics import average_precision_score

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, max_words_q, vocabulary_size, drop_out_rate):
        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.max_words_q = max_words_q
        self.vocabulary_size = vocabulary_size    
        self.drop_out_rate = drop_out_rate

        # 问题embedding
        self.embed_ques_W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_ques_W')

        # RNN编码器
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - self.drop_out_rate)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

        # 状态embedding
        self.embed_state_W = tf.Variable(tf.random_uniform([2*rnn_size*rnn_layer, self.dim_hidden], -0.08,0.08),name='embed_state_W')
        self.embed_state_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_state_b')
        # 图像embedding
        self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_image_b')
        # 打分embedding
        self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random_uniform([num_output], -0.08, 0.08), name='embed_scor_b')

    def build_model(self):
        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        label = tf.placeholder(tf.int64, [self.batch_size,]) 
        
        state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        loss = 0.0
        for i in range(max_words_q):
            if i==0:
            ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
            ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])

            ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
            ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)

        # 融合图像+问题Embedding
        state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
        state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
        state_emb = tf.tanh(state_linear)

        image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
        image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_linear)

        scores = tf.mul(state_emb, image_emb)
        scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
        scores_emb = tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b) 

        # 交叉熵
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(scores_emb, label)

        loss = tf.reduce_mean(cross_entropy)
        
        return loss, image, question, label
    
    def build_generator(self):
        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])

        state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        loss = 0.0
        for i in range(max_words_q):
                if i==0:
                    ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
                else:
            tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])
            ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
            ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)
        
        # 融合图像+问题
        state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
        state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
        state_emb = tf.tanh(state_linear)

        image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
        image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_linear)

        scores = tf.mul(state_emb, image_emb)
        scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
        scores_emb = tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b) 

        # 最终答案
        generated_ANS = tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b)

        return generated_ANS, image, question
    
#####################################################
#                      全局参数                      #  
#####################################################
print '读取参数中...'

# 输入数据路径
input_img_h5 = './data_img.h5'
input_ques_h5 = './data_prepro.h5'
input_json = './data_prepro.json'

# 训练参数
learning_rate = 0.0003
learning_rate_decay_start = -1
batch_size = 500
input_embedding_size = 200
rnn_size = 512
rnn_layer = 2
dim_image = 4096
dim_hidden = 1024
num_output = 1000
img_norm = 1
decay_factor = 0.99997592083

# 模型保存路径
checkpoint_path = 'model_save/'

# misc
gpu_id = 0
max_itr = 150000
n_epochs = 300
max_words_q = 26
num_answer = 1000
#####################################################

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

def get_data():

    dataset = {}
    train_data = {}
    # 读取json文件
    print('读取json文件中...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # 读取图像特征
    print('读取图像特征中...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    # 读取h5文件
    print('读取h5文件中...')
    with h5py.File(input_ques_h5,'r') as hf:
        # 训练数据总数为 215375
        # 问题 (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # 最长为 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # 图像总数为 82460
        tem = hf.get('img_pos_train')
        train_data['img_list'] = np.array(tem)-1
        # 答案为 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('问题对齐')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    print('正则化图像特征')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

    return dataset, img_feature, train_data

def get_data_test():
    dataset = {}
    test_data = {}

    print('读取json文件中...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    print('读取图像特征中...')
    with h5py.File(input_img_h5,'r') as hf:
        tem = hf.get('images_test')
        img_feature = np.array(tem)
    print('读取h5文件中...')
    with h5py.File(input_ques_h5,'r') as hf:
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        tem = hf.get('img_pos_test')
        test_data['img_list'] = np.array(tem)-1
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
    tem = hf.get('MC_ans_test')
    test_data['MC_ans_test'] = np.array(tem)

    print('问题对齐')
    test_data['question'] = right_align(test_data['question'], test_data['length_q'])

    print('正则化图像特征')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

    return dataset, img_feature, test_data

def train():
    print('读取数据集...')
    dataset, img_feature, train_data = get_data()
    num_train = train_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print('字典长度: ' + str(vocabulary_size))

    print('构建模型中...')
    model = Answer_Generator(
        rnn_size = rnn_size,
        rnn_layer = rnn_layer,
        batch_size = batch_size,
        input_embedding_size = input_embedding_size,
        dim_image = dim_image,
        dim_hidden = dim_hidden,
        max_words_q = max_words_q,    
        vocabulary_size = vocabulary_size,
        drop_out_rate = 0.5)

    tf_loss, tf_image, tf_question, tf_label = model.build_model()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=100)

    tvars = tf.trainable_variables()
    lr = tf.Variable(learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate=lr)

    gvs = opt.compute_gradients(tf_loss,tvars)
    clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]
    train_op = opt.apply_gradients(clipped_gvs)

    tf.initialize_all_variables().run()

    print('开始训练...')
    for itr in range(max_itr):
        tStart = time.time()
        # 打乱训练数据顺序
        index = np.random.random_integers(0, num_train-1, batch_size)

        current_question = train_data['question'][index,:]
        current_length_q = train_data['length_q'][index]
        current_answers = train_data['answers'][index]
        current_img_list = train_data['img_list'][index]
        current_img = img_feature[current_img_list,:]

        _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_image: current_img,
                        tf_question: current_question,
                        tf_label: current_answers
                        })

        current_learning_rate = lr*decay_factor
        lr.assign(current_learning_rate).eval()

        tStop = time.time()
        if np.mod(itr, 100) == 0:
            print("训练轮回: ", itr, " Loss: ", loss, " Learning Rate: ", lr.eval())
            print("训练时间:", round(tStop - tStart,2), "s")
        if np.mod(itr, 15000) == 0:
            print("训练轮回 ", itr, " 已完成. 保存模型中...")
            saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)

    print("保存最终模型...")
    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print("总时间:", round(tStop_total - tStart_total,2), "s")


def test(model_path='model_save/model-150000'):
    print '读取数据集...'
    dataset, img_feature, test_data = get_data_test()
    num_test = test_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print '字典长度: ' + str(vocabulary_size)

    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = batch_size,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,
            vocabulary_size = vocabulary_size,
            drop_out_rate = 0)

    tf_answer, tf_image, tf_question, = model.build_generator()

    #sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    tStart_total = time.time()
    result = []
    for current_batch_start_idx in xrange(0,num_test-1,batch_size):
    #for current_batch_start_idx in xrange(0,3,batch_size):
        tStart = time.time()
        # set data into current*
        if current_batch_start_idx + batch_size < num_test:
            current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
        else:
            current_batch_file_idx = range(current_batch_start_idx,num_test)

        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]
        current_img_list = test_data['img_list'][current_batch_file_idx]
        current_ques_id  = test_data['ques_id'][current_batch_file_idx]
        current_img = img_feature[current_img_list,:] # (batch_size, dim_image)

        # deal with the last batch
        if(len(current_img)<500):
                pad_img = np.zeros((500-len(current_img),dim_image),dtype=np.int)
                pad_q = np.zeros((500-len(current_img),max_words_q),dtype=np.int)
                pad_q_len = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_q_id = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_ques_id = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_img_list = np.zeros(500-len(current_length_q),dtype=np.int)
                current_img = np.concatenate((current_img, pad_img))
                current_question = np.concatenate((current_question, pad_q))
                current_length_q = np.concatenate((current_length_q, pad_q_len))
                current_ques_id = np.concatenate((current_ques_id, pad_q_id))
                current_img_list = np.concatenate((current_img_list, pad_img_list))


        generated_ans = sess.run(
                tf_answer,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question
                    })

        top_ans = np.argmax(generated_ans, axis=1)


        # initialize json list
        for i in xrange(0,500):
            ans = dataset['ix_to_ans'][str(top_ans[i]+1)]
            if(current_ques_id[i] == 0):
                continue
            result.append({u'answer': ans, u'question_id': str(current_ques_id[i])})

        tStop = time.time()
        print ("Testing batch: ", current_batch_file_idx[0])
        print ("花费时间:", round(tStop - tStart,2), "s")
    print ("预测完成")
    tStop_total = time.time()
    print ("总时间:", round(tStop_total - tStart_total,2), "s")
    # Save to JSON
    print '保存预测结果中...'
    my_list = list(result)
    dd = json.dump(my_list,open('data.json','w'))

if __name__ == '__main__':
    with tf.device('/gpu:'+str(0)):
        train()
    with tf.device('/gpu:'+str(1)):
        test()