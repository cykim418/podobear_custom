import json
import re

file = open('D:\datasets\donut_data\\paddleOCR_output.txt', 'r', encoding="UTF-8")
lines = file.readlines()


donut_dict = {}

for line in lines:
    textlist = line.split('\\')
    category = textlist[6].split('_')[1]
    textvalue = textlist[6].split('\t')[1].replace('\n', '')
    dictkey = int(textlist[5])
    if dictkey in donut_dict:
        donut_dict[dictkey].append([category, textvalue])
    else:
        donut_dict[dictkey] = [[category, textvalue]]


# # 라벨링 체크용 txt 파일 만드는 코드
# with open("D:\datasets\donut_data\\labelcheck.txt", "w", encoding="UTF-8") as file:
#     for i in range(1282, 2543):
#         if donut_dict[i]:
#             file.write(str(i) + '.jpg\n')
#             for category, val in donut_dict[i]:
#                 file.write(category + '\t' + val + '\n')



# metadata.jsonl 파일 만드는 코드

donut_dict2 = {}

with open("D:\Code\yolov5\\metadata.jsonl", "w", encoding="UTF-8") as file:
    for f in donut_dict:
        donut_dict2['file_name'] = f + '.jpg'
        donut_dict3 = {}  # dt_parse 뒤에 붙는 딕셔너리
        donut_dict4 = {}  # ground_truth 뒤에 붙는 딕셔너리

        for category, val in donut_dict[f]:
            if category in donut_dict3:
                if type(donut_dict3[category]) is str:
                    donut_dict3[category] = [donut_dict3[category]]
                donut_dict3[category].append(val)
            else:
                donut_dict3[category] = val
        donut_dict4["gt_parse"] = json.dumps(donut_dict3, ensure_ascii=False)
        file.write(json.dumps(donut_dict4, ensure_ascii=False) + "\n")




