
# 라벨링데이터 -> json 데이터로 형식 바꿔주는 코드
import json
import os
import shutil

imagePath = 'D:\\datasets\\yolo_train_data_in88\\88images\\'
copyPath = 'D:\\datasets\\donut_data\\1000bscards\\'



if not os.path.exists(copyPath):
    os.makedirs(copyPath)

file = open('D:\\Code\\bscard_donut\\labeling_text\\labeling0_1000.txt', 'r', encoding="UTF-8")
lines = file.readlines()
label_dict = {}
keyname = ''
for line in lines:
  if line.endswith('.jpg\n'):
    keyname = line.replace('\n', '')
    imgpath = imagePath + keyname
    copypath = copyPath + keyname
    shutil.copyfile(imgpath, copypath)
# result = []
# result_dict = {}
# donut_dict2 = {}  # ground_truth 뒤에 붙는 딕셔너리
#
# with open("D:\Code\yolov5\\metadata0704.jsonl", "w", encoding="UTF-8") as file:
#   for i in label_dict:
#     file_name = str(i) + '.jpg'
#     result_dict['file_name'] = file_name
#     donut_dict = {}  # dt_parse 뒤에 붙는 딕셔너리
#
#     for category, val in label_dict[i]:
#       if category in donut_dict:
#         if type(donut_dict[category]) is str:
#           donut_dict[category] = [donut_dict[category]]
#         donut_dict[category].append(val)
#       else:
#         donut_dict[category] = val
#     donut_dict2['gt_parse'] = donut_dict.copy()
#     result_dict['ground_truth'] = json.dumps(donut_dict2.copy(), ensure_ascii=False)
#     file.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
#
#
