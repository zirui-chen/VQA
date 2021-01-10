import json
import os


# 获得数据
def download_vqa():
    # wget(Ubuntu)
    # 下载
    os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip -P zip/')

    # 解压
    os.system('unzip zip/Questions_Train_mscoco.zip -d annotations/')
    os.system('unzip zip/Questions_Val_mscoco.zip -d annotations/')
    os.system('unzip zip/Questions_Test_mscoco.zip -d annotations/')
    os.system('unzip zip/Annotations_Train_mscoco.zip -d annotations/')
    os.system('unzip zip/Annotations_Val_mscoco.zip -d annotations/')


# 处理数据
def process_vqa():
    # 将上述数据转变为 [[Question_id, Image_id, Question, multipleChoice_answer, Answer] ... ] 形式

    train = []
    test = []
    imdir='%s/COCO_%s_%012d.jpg'

    print('读取数据集...')
    train_anno = json.load(open('annotations/mscoco_train2014_annotations.json', 'r'))
    val_anno = json.load(open('annotations/mscoco_val2014_annotations.json', 'r'))

    train_ques = json.load(open('annotations/MultipleChoice_mscoco_train2014_questions.json', 'r'))
    val_ques = json.load(open('annotations/MultipleChoice_mscoco_val2014_questions.json', 'r'))
    test_ques = json.load(open('annotations/MultipleChoice_mscoco_test2015_questions.json', 'r'))
    
    subtype = 'train2014'
    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
        image_path = imdir%(subtype, subtype, train_anno['annotations'][i]['image_id'])

        question = train_ques['questions'][i]['question']
        mc_ans = train_ques['questions'][i]['multiple_choices']

        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

    subtype = 'val2014'
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        image_path = imdir%(subtype, subtype, val_anno['annotations'][i]['image_id'])

        question = val_ques['questions'][i]['question']
        mc_ans = val_ques['questions'][i]['multiple_choices']

        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
    subtype = 'test2015'
    for i in range(len(test_ques['questions'])):
        question_id = test_ques['questions'][i]['question_id']
        image_path = imdir%(subtype, subtype, test_ques['questions'][i]['image_id'])

        question = test_ques['questions'][i]['question']
        mc_ans = test_ques['questions'][i]['multiple_choices']

        test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans})

    print('训练集样本数：%d, 测试集样本数：%d...' %(len(train), len(test)))

    json.dump(train, open('vqa_raw_train.json', 'w'))
    json.dump(test, open('vqa_raw_test.json', 'w'))


if __name__ == "__main__":
    download_vqa()
    process_vqa()
