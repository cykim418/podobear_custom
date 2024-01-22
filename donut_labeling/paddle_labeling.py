# import json
# import re
#
# file = open('D:\datasets\podobear_dataset\\all_businesscards_crops.txt', 'r', encoding="UTF-8")
# lines = file.readlines()
#
#
# paddle_dict = {}
#
# for line in lines:
#     file_name_dict = {}
#     file_path = line.split('\t')[0] # 전체 파일 경로
#     textlist = line.split('\\')
#     dictkey = int(textlist[5]) # 사진번호
#     file_name = textlist[6]
#     key_idx = file_name.index('_')
#     key_num = int(file_name[:key_idx]) # yolo 키 번호
#
#     if dictkey in paddle_dict:
#         paddle_dict[dictkey][key_num] = file_name
#     else:
#         file_name_dict[key_num] = file_name
#         paddle_dict[dictkey] = file_name_dict
#
#
# # 라벨링 체크용 txt 파일 만드는 코드
# with open("D:\datasets\podobear_dataset\\all_businesscards_crops_sort.txt", "w", encoding="UTF-8") as file:
#     file_path = 'C:\\Users\\hanhyungu\\Downloads\\yolo_result\\'
#     for i in range(1, 1809):
#         for j in range(20):
#             if i in paddle_dict and j in paddle_dict[i]:
#                 res = file_path + str(i) + '\\' + paddle_dict[i][j]
#                 file.write(res)
#             else:
#                 print( i, j)



# paddle output 에서 확률 0.5이상만 다시 저장하는 코드

file = open("D:\datasets\podobear_dataset\\all_businesscards_crops_sort.txt", "r", encoding="UTF-8")
lines = file.readlines()

with open("D:\datasets\podobear_dataset\\all_businesscards_crops_sort2.txt", "w", encoding="UTF-8") as f:
    for line in lines:
        probs = float(line.split('\\')[6].split('_')[2])
        if probs < 0.5:
            continue

        f.write(line)
