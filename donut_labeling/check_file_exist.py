'''
이미지 파일 중복 체크
'''


# import os
#
# with open('paddleOCR_label_eng_filename_only.txt', 'r', encoding='UTF-8') as f:
#     lines = f.readlines()
#     file_dict2 = {line.split('\t')[0]:1 for line in lines}
#     file_dict = [line.split('\t')[0] for line in lines]
#     imgs = {i:'1' for i in os.listdir('paddleOCR_ExtraData_39server_eng_only')}
#
#
#     print(len(file_dict))
#     print(len(imgs))
#
#     res = 0
#     for i in file_dict:
#         if i in imgs:
#             res += 1
#         else:
#             print(i)
#     print(res)
