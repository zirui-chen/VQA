import sys
import os.path
import re
import glob
import copy
import json
import scipy.io
import pdb
import string
import h5py
import numpy as np
from random import shuffle, seed
from scipy.misc import imread, imresize
from nltk.tokenize import word_tokenize


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(imgs):
    print('分词结果的例子:')
    for i,img in enumerate(imgs):
        s = img['question']
        txt = tokenize(str(s).lower())
        img['processed_tokens'] = txt
        if i < 10: print(txt)
        if i % 1000 == 0:
            sys.stdout.write("处理中... %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()
    return imgs

def build_vocab_question(imgs):
    count_thr = 0

    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print('出现次数最多的字及其出现次数:')
    print('\n'.join(map(str,cw[:20])))

    total_words = sum(counts.itervalues())
    print('总字数为:', total_words)
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('词典中字的数量应为 %d' % (len(vocab), ))
    print('无法识别的字符数量为: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    print('给字典加入特殊字符[UNK]')
    vocab.append('UNK')
  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs, vocab

def apply_vocab_question(imgs, wtoi):
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs

def get_top_answers(imgs):
    # {'答案':出现的次数}
    counts = {}
    for img in imgs:
        ans = img['ans'] 
        counts[ans] = counts.get(ans, 0) + 1

    # 将上述字典根据出现的次数从高到低排序
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print('出现次数最多的前20个答案及其对应的次数:')
    print('\n'.join(map(str,cw[:20])))
    
    # 根据次数的高低按顺序插入的答案
    vocab = []
    for i in range(1000):
        vocab.append(cw[i][1])

    return vocab[:1000]

def encode_question(imgs, wtoi):
    max_length = 26
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['ques_id']
        label_length[question_counter] = min(max_length, len(img['final_question']))
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
    
    return label_arrays, label_length, question_id


def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi[img['ans']]

    return ans_arrays

def encode_mc_answer(imgs, atoi):
    N = len(imgs)
    mc_ans_arrays = np.zeros((N, 18), dtype='uint32')

    for i, img in enumerate(imgs):
        for j, ans in enumerate(img['MC_ans']):
            mc_ans_arrays[i,j] = atoi.get(ans, 0)
    return mc_ans_arrays

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if atoi.get(img['ans'],len(atoi)+1) != len(atoi)+1:
            new_imgs.append(img)

    print('训练集数据从 %d 减少至 %d '%(len(imgs), len(new_imgs)))
    return new_imgs

def get_unqiue_img(imgs):
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    for img in imgs:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1

    unique_img = [w for w,n in count_img.iteritems()]
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.


    for i, img in enumerate(imgs):
        img_pos[i] = imgtoi.get(img['img_path'])

    return unique_img, img_pos

def pre_process():
    # 读取处理后的训练、测试集原数据
    imgs_train = json.load(open('data/vqa_raw_train.json', 'r'))
    imgs_test = json.load(open('data/vqa_raw_test.json', 'r'))

    # 获得最高的前1000个问题，并根据顺序给定唯一标识id，并制作对应查询字典及反向查询字典
    top_ans = get_top_answers(imgs_train)
    atoi = {w:i+1 for i,w in enumerate(top_ans)}
    itoa = {i+1:w for i,w in enumerate(top_ans)}

    # 筛选问题，将答案非出现次数最高前1000的问题从训练集中剔除
    imgs_train = filter_question(imgs_train, atoi)

    # 打乱数据
    seed(123)
    shuffle(imgs_train)

    # 对训练、测试集中的问题进行分词，并将结果加入作为单条的一个属性
    imgs_train = prepro_question(imgs_train)
    imgs_test = prepro_question(imgs_test)

    # 根据训练集的创建字符id词典，及对应的查询字典和反向查询字典，接着上一个分词结果获得最终的问题表示
    imgs_train, vocab = build_vocab_question(imgs_train)
    itow = {i+1:w for i,w in enumerate(vocab)}
    wtoi = {w:i+1 for i,w in enumerate(vocab)}

    # 使用字典将训练、测试集问题变为id表示形式，获得每个问题的长度及其id
    ques_train, ques_length_train, question_id_train = encode_question(imgs_train, wtoi)
    imgs_test = apply_vocab_question(imgs_test, wtoi)
    ques_test, ques_length_test, question_id_test = encode_question(imgs_test, wtoi)

    # 获得训练、测试集对应的照片名称信息
    unique_img_train, img_pos_train = get_unqiue_img(imgs_train)
    unique_img_test, img_pos_test = get_unqiue_img(imgs_test)

    # get the answer encoding.
    A = encode_answer(imgs_train, atoi)
    MC_ans_test = encode_mc_answer(imgs_test, atoi)

    # create output h5 file for training set.
    N = len(imgs_train)
    f = h5py.File('data_prepro.h5', "w")
    f.create_dataset("ques_train", dtype='uint32', data=ques_train)
    f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train)
    f.create_dataset("answers", dtype='uint32', data=A)
    f.create_dataset("question_id_train", dtype='uint32', data=question_id_train)
    f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
    
    f.create_dataset("ques_test", dtype='uint32', data=ques_test)
    f.create_dataset("ques_length_test", dtype='uint32', data=ques_length_test)
    f.create_dataset("question_id_test", dtype='uint32', data=question_id_test)
    f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)
    f.create_dataset("MC_ans_test", dtype='uint32', data=MC_ans_test)

    f.close()
    print('保存 ', 'data_prepro.h5')

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['ix_to_ans'] = itoa
    out['unique_img_train'] = unique_img_train
    out['unique_img_test'] = unique_img_test
    json.dump(out, open('data_prepro.json', 'w'))
    print('保存 ', 'data_prepro.json')


if __name__ == "__main__":
    pre_process()