'''
라벨 파일에서 한글 제거해서 다시 생성하는 코드
'''
# import re
# with open('paddleOCR_label_1102.txt', 'r', encoding='UTF-8') as f:
#     lines = f.readlines()
#     with open('paddleOCR_label_1102_korean_only.txt', 'w', encoding='UTF-8') as file:
#         for line in lines:
#             file_path = line.split('\t')[0]
#             if file_path.startswith('1627') or file_path.startswith('1656') or file_path.startswith('E:\OCR_bscard\Podo'):
#                 check_eng = re.search("[ㄱ-ㅣ가-힣]", line)
#                 if check_eng is None:
#                     file.write(line)



'''
라벨 파일에서 한글 제거해서 다시 생성하기
이전 경로 제외하고 파일명만 남겨서 다시 생성
(겹치는 파일명 있어서 앞에 email/url 붙임)
'''
# import re
# with open('paddleOCR_label_extraData_1106_kor+eng.txt', 'r', encoding='UTF-8') as f:
#     lines = f.readlines()
#     with open('paddleOCR_label_extraData_1106_kor+eng2.txt', 'w', encoding='UTF-8') as file:
#         for line in lines:
#             file_path = line.split('\t')[0]
#             if file_path.startswith('1627') or file_path.startswith('1656'):
#                 file.write(line)
#             elif file_path.startswith('url/'):
#                 new_line = re.sub('url/', 'url', line)
#                 file.write(new_line)
#             elif file_path.startswith('email/'):
#                 new_line = re.sub('email/', 'email', line)
#                 file.write(new_line)
#             elif file_path.startswith('E:\OCR_bscard\Podo'):
#                 new_line = line.split('\\')[5]
#                 file.write(new_line)


'''
add file path in front of file name
'''
with open('paddleOCR_label_extraData_1106_kor+eng2.txt', 'r', encoding='UTF-8') as f:
    with open('paddleOCR_label_extraData_1106_kor+eng3.txt', 'w', encoding='UTF-8') as file:
        lines = f.readlines()
        for line in lines:
            new_line = '/home/user/podobear/dataset/paddleOCRv3_kor_eng/' + line
            file.write(new_line)



