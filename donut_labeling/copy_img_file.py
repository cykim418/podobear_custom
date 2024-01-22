import os
import shutil


'''
생성한 라벨 파일에 맞는 파일만 새 폴더에 복사
'''

new_img_path = 'paddleOCR_kordata_for_test'
if not os.path.exists(new_img_path):
    os.makedirs(new_img_path)

with open('paddleOCR_label_kor_filename_only.txt', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        file_path = line.split('\t')[0]
        new_img_path = 'paddleOCR_kordata_for_test'

        if file_path.startswith('16'):
            img_path = os.path.join('paddleOCR_ExtraData_39server_eng', file_path)

            if os.path.exists(img_path):
                new_img_path = os.path.join(new_img_path, file_path)
                shutil.copyfile(img_path, new_img_path)

        else:
            new_img_path = 'paddleOCR_kordata_for_test'
            file_name = file_path
            img_path = os.path.join('paddleOCR_ExtraData_39server_eng/crop_data', file_name)

            if os.path.exists(img_path):
                new_img_path = os.path.join(new_img_path, file_name)
                shutil.copyfile(img_path, new_img_path)

            if file_path.find("\\") == -1:
                file_name = file_path.split('/')[1]
                cate_name = file_path.split('/')[0]
                img_path = os.path.join('paddleOCR_ExtraData_39server_eng', file_path)

                if os.path.exists(img_path):
                    new_img_path = os.path.join(new_img_path, cate_name + file_name)
                    shutil.copyfile(img_path, new_img_path)

            else:
                file_name = file_path
                img_path = os.path.join('paddleOCR_ExtraData_39server_eng/crop_data', file_name)

                if os.path.exists(img_path):
                    new_img_path = os.path.join(new_img_path, file_name)
                    shutil.copyfile(img_path, new_img_path)

