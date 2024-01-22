
# 라벨링데이터.txt -> json 데이터로 형식 바꾸고, train, test, validation 나누는 코드
import json
import shutil
import os
from datetime import datetime

labelingtxtPath = 'D:\\datasets\\donut_data\\labeling_text\\labeling0_150.txt'



file = open(labelingtxtPath, 'r', encoding="UTF-8")
lines = file.readlines()
label_dict = {}
keyname = ''
for line in lines:
  if line.endswith('.jpg\n'):
    keyname = line.replace('.jpg\n', '')
    label_dict[keyname] = []
  else:
    lst = line.split('\t')
    if len(lst) == 2:
      label_dict[keyname].append([lst[0], lst[1].rstrip()])

result = []
result_dict = {}
donut_dict2 = {}  # ground_truth 뒤에 붙는 딕셔너리


for i in label_dict:
    file_name = str(i) + '.jpg'
    result_dict['file_name'] = file_name
    donut_dict = {}  # dt_parse 뒤에 붙는 딕셔너리

    for category, val in label_dict[i]:
      if category in donut_dict:
        if type(donut_dict[category]) is str:
          donut_dict[category] = [donut_dict[category]]
        donut_dict[category].append(val)
      else:
        donut_dict[category] = val
    donut_dict2['gt_parse'] = donut_dict.copy()
    result_dict['ground_truth'] = json.dumps(donut_dict2.copy(), ensure_ascii=False)
    result.append(result_dict.copy())


traindata_num = int(len(result)*0.9)
valdata_num = int(len(result)*0.07)
testdata_num = int(len(result)*0.03)

if (traindata_num + valdata_num + testdata_num) < len(result):
    traindata_num += (len(result) - (traindata_num + valdata_num + testdata_num))



path_date =  datetime.today().strftime("%Y_%m_%d") + "2"


train_path = "D:\datasets\\donut_data\\" + path_date + "\\train\\images\\"
if not os.path.exists(train_path):
    os.makedirs(train_path)
with open(train_path + "metadata.jsonl", "w", encoding="UTF-8") as f:
    for i in range(traindata_num):
        f.write(json.dumps(result[i], ensure_ascii=False) + '\n')
        copypath = train_path + result[i]['file_name']
        imagepath = 'D:\\datasets\\yolo_train_data_in88\\88images\\' + result[i]['file_name']
        shutil.copyfile(imagepath, copypath)


test_path = "D:\datasets\\donut_data\\" + path_date + "\\test\\images\\"
if not os.path.exists(test_path):
    os.makedirs(test_path)
with open(test_path + "metadata.jsonl", "w", encoding="UTF-8") as f:
    for i in range(traindata_num, traindata_num+testdata_num):
        f.write(json.dumps(result[i], ensure_ascii=False) + '\n')
        copypath = test_path + result[i]['file_name']
        imagepath = 'D:\\datasets\\donut_data\\1000bscards\\' + result[i]['file_name']
        shutil.copyfile(imagepath, copypath)


val_path = "D:\datasets\\donut_data\\" + path_date + "\\validation\\images\\"
if not os.path.exists(val_path):
    os.makedirs(val_path)
with open(val_path + "metadata.jsonl", "w", encoding="UTF-8") as f:
    for i in range(traindata_num+testdata_num, traindata_num+testdata_num+valdata_num):
        f.write(json.dumps(result[i], ensure_ascii=False) + '\n')
        copypath = val_path + result[i]['file_name']
        imagepath = 'D:\\datasets\\donut_data\\1000bscards\\' + result[i]['file_name']
        shutil.copyfile(imagepath, copypath)
