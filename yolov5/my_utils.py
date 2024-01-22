import random

import cv2
import numpy as np
from utils.plots import plot_results
import shutil
import re
import json

import logging
import torch, torchvision
import torchvision.transforms.functional as F
from torchvision import ops

import os
from typing import List, Dict

from torch.utils.mobile_optimizer import optimize_for_mobile

def yolov4_datasets_to_yolov5():
    '''
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
    '''
    base_path = r'D:\aip'
    yolov5_datasets_path = os.path.join(base_path, 'datasets/images')
    yolov4_datasets_path = os.path.join(base_path, 'yolov4/data')

    for dataset_type in ['val',  'train']:
        yolov5_labels_path = os.path.join(base_path, 'datasets/labels', f'namecard_ver2_yolo_{dataset_type}')
        if not os.path.exists(yolov5_labels_path):
            os.makedirs(yolov5_labels_path, exist_ok=True)

        with open(os.path.join(yolov4_datasets_path, f'{dataset_type}.txt'), 'r') as txt:
            lines = txt.readlines()
            for line in lines:
                '''
                yolo v4 
                x1, y1, x2, y2
                가로 width 좌표, 세로 height 좌표, 가로 width 좌표, 세로 height 좌표, class 번호(0부터 시작)
                '''
                print(line)
                image_name = line[line.rfind('\\')+1:line.rfind('\\')+13]
                print(image_name)
                new_txt_name = image_name + '.txt'

                img_path = os.path.join(yolov5_datasets_path, f'namecard_ver2_yolo_{dataset_type}', image_name + '.jpg')
                img = cv2.imread(img_path)
                print(img.shape)
                img_width = img.shape[1]
                img_height = img.shape[0]

                curr_pos2 = line.rfind('\n')
                curr_pos1 = line.rfind(' ') + 1

                with open(os.path.join(yolov5_labels_path, new_txt_name), 'w') as newtxt:
                    '''
                    yolo v5 
                    center x, center y, w, h (normalized)
                    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
                    '''
                    while curr_pos1:
                        print(curr_pos1, curr_pos2)
                        curr_object  = line[curr_pos1:curr_pos2]
                        data = curr_object.split(',')
                        print(data)
                        data = [int(i) for i in data]


                        center_x = np.mean([data[0], data[2]])
                        center_y = np.mean([data[1], data[3]])

                        normalized_center_x = str(center_x / img_width)
                        normalized_center_y = str(center_y / img_height)

                        width = data[2] - data[0]
                        height = data[3] - data[1]

                        normalized_width = str(width / img_width)
                        normalized_height = str(height / img_height)

                        #label은 0부터 시작
                        newline = ' '.join([str(data[4]), normalized_center_x, normalized_center_y, normalized_width, normalized_height, '\n'])
                        newtxt.write(newline)

                        curr_pos1 -= 1
                        curr_pos2 = curr_pos1
                        curr_pos1 = line.rfind(' ', 0, curr_pos1 - 1) + 1


def plot_train_results():
    plot_results(os.path.join(os.path.dirname(__file__), 'runs/train/exp4/result.csv'))


def labelme_to_yolov5():
    dir_path = './data/labelme'
    new_dir_path = './data/newLabelme'

    files = os.listdir(dir_path)
    cur_path = os.path.dirname(os.path.abspath('__file__'))

    try:
        os.makedirs(os.path.join(cur_path, new_dir_path))
    except:
        print('already exists;', cur_path)

    label_set = set()
    # label_list = ['cls', 'd1', 'n20', 'c2', 'i4', 'c1', 'i3', 'p4', 'p6', 'n10', 'n3', 'd2', 'p1', 'i2', 'id2', 'd3', 'a2', 'i1', 'p3', 'a1', 'id1', 'p2', 'p5', 'n1', 'n2', 'title']
    label_list = ['engname','email','koraddress','korname','engposition','engaddress','korposition','web','korcompany','engcompany','call']

    for file_name in files:
        full_path = os.path.join(cur_path, dir_path, file_name)


        if 'json' in file_name:
            json_path = full_path
            image_path = os.path.join(cur_path, dir_path, file_name[:-4] + 'jpg')
            txt_path = os.path.join(cur_path, 'datasets/labels/train', file_name[:-4] + 'txt')
            cp_image_path = os.path.join(cur_path, 'datasets/images/train', file_name[:-4] + 'jpg')
        else:
            continue


        new_image_path = os.path.join(cur_path, new_dir_path, file_name[:-4] + 'jpg')
        cvimg = cv2.imread(image_path, cv2.IMREAD_COLOR)

        with open(json_path, 'r') as read_json:
            with open(txt_path, 'w') as txt:

                json_data = json.load(read_json)
                for i in range(len(json_data['shapes'])):

                    x1 = int(json_data['shapes'][i]['points'][0][0])
                    y1 = int(json_data['shapes'][i]['points'][0][1])
                    x2 = int(json_data['shapes'][i]['points'][1][0])
                    y2 = int(json_data['shapes'][i]['points'][1][1])
                    label = json_data['shapes'][i]['label']
                    color = (0, 0, 255)

                    cv2.rectangle(cvimg, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(cvimg, label, (x1 - 20, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

                    if not label in label_set:
                        label_set.add(label)

                    img_width = cvimg.shape[1]
                    img_height = cvimg.shape[0]

                    center_x = np.mean([x1, x2])
                    center_y = np.mean([y1, y2])

                    normalized_center_x = str(center_x / img_width)
                    normalized_center_y = str(center_y / img_height)

                    width = x2 - x1
                    height = y2 - y1

                    normalized_width = str(width / img_width)
                    normalized_height = str(height / img_height)

                    number_label = label_list.index(label)
                    newline = ' '.join([str(number_label), normalized_center_x, normalized_center_y, normalized_width, normalized_height,'\n'])
                    txt.write(newline)


        shutil.copyfile(image_path, cp_image_path)
        cv2.imwrite(new_image_path, cvimg)

    print(label_set, len(label_set))
    # {'cls', 'd1', 'n20', 'c2', 'i4', 'c1', 'i3', 'p4', 'p6', 'n10', 'n3', 'd2', 'p1', 'i2', 'id2', 'd3', 'a2', 'i1', 'p3', 'a1', 'id1', 'p2', 'p5', 'n1', 'n2', 'title'} 26


def labelme_to_yolov5_crops():
    dir_path = './data/final_test_yolov5_labelme'
    new_dir_path = './data/final_test_yolov5_labelme/crpos'

    files = os.listdir(dir_path)
    cur_path = os.path.dirname(os.path.abspath('__file__'))

    try:
        os.makedirs(os.path.join(cur_path, new_dir_path))
    except:
        print('already exists;', cur_path)

    label_set = set()
    # label_list = ['cls', 'd1', 'n20', 'c2', 'i4', 'c1', 'i3', 'p4', 'p6', 'n10', 'n3', 'd2', 'p1', 'i2', 'id2', 'd3', 'a2', 'i1', 'p3', 'a1', 'id1', 'p2', 'p5', 'n1', 'n2', 'title']
    label_list = ['engname','email','koraddress','korname','engposition','engaddress','korposition','web','korcompany','engcompany','call']

    for file_name in files:
        full_path = os.path.join(cur_path, dir_path, file_name)


        if 'json' in file_name:
            json_path = full_path
            image_path = os.path.join(cur_path, dir_path, file_name[:-4] + 'jpg')
        else:
            continue

        cvimg = cv2.imread(image_path, cv2.IMREAD_COLOR)

        with open(json_path, 'r') as read_json:
            json_data = json.load(read_json)

            image_folder_name = os.path.join(cur_path, new_dir_path, file_name[:-5])
            try:
                os.makedirs(image_folder_name)
            except:
                print('already exists;', cur_path)

            for i in range(len(json_data['shapes'])):
                x1 = int(json_data['shapes'][i]['points'][0][0])
                y1 = int(json_data['shapes'][i]['points'][0][1])
                x2 = int(json_data['shapes'][i]['points'][1][0])
                y2 = int(json_data['shapes'][i]['points'][1][1])
                label = json_data['shapes'][i]['label']
                rand_num = random.randint(0,1e6)

                crop_file_name = os.path.join(image_folder_name, f'{i}_{label}_{rand_num}.jpg')
                print(crop_file_name)
                cv2.imwrite(crop_file_name, cvimg[y1:y2, x1:x2])


def labelme_to_yolov4():
    dir_path = './data/labelme'
    new_dir_path = './yolov4_dataset'

    files = os.listdir(dir_path)
    cur_path = os.path.dirname(os.path.abspath('__file__'))
    txt_path = os.path.join(cur_path, 'yolov4_dataset', 'val.txt')
    label_list = ['for id from 1','cls', 'd1', 'n20', 'c2', 'i4', 'c1', 'i3', 'p4', 'p6', 'n10', 'n3', 'd2', 'p1', 'i2', 'id2', 'd3', 'a2', 'i1', 'p3', 'a1', 'id1', 'p2', 'p5', 'n1', 'n2', 'title']

    try:
        os.makedirs(os.path.join(cur_path, new_dir_path))
    except:
        print('already exists;', os.path.join(cur_path, new_dir_path))

    image_number = 20
    with open(txt_path, 'w') as txt:
        for file_name in files:
            full_path = os.path.join(cur_path, dir_path, file_name)

            if 'json' in file_name:
                json_path = full_path
            else:
                continue

            image_name = '{:012d}'.format(image_number)
            image_number += 1

            image_path = os.path.join(cur_path, dir_path, file_name[:-4] + 'jpg')
            cp_image_path = os.path.join(cur_path, new_dir_path, image_name + '.jpg')
            shutil.copyfile(image_path, cp_image_path)

            input_file_name = os.path.join(r'D:\nhk_pytorch-YOLOv4\dataset\document_ocr_val', image_name + '.jpg')
            txt.write(input_file_name)

            with open(json_path, 'r') as read_json:
                    json_data = json.load(read_json)
                    for i in range(len(json_data['shapes'])):

                        x1 = int(json_data['shapes'][i]['points'][0][0])
                        y1 = int(json_data['shapes'][i]['points'][0][1])
                        x2 = int(json_data['shapes'][i]['points'][1][0])
                        y2 = int(json_data['shapes'][i]['points'][1][1])
                        label = json_data['shapes'][i]['label']

                        number_label = label_list.index(label)
                        newline = ' ' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(number_label)
                        txt.write(newline)
            txt.write('\n')


logging.getLogger(__name__)

def post_cd():
    # iouThreshold는 원래 conf threshold 를 의미
    # input 형태 = [box 4개 좌표, box conf, 각 class당 확신 정도->softmax 값]
    class PostCD(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inputs: torch.Tensor, iouThreshold: float):
            # shape e.g., "shape": [1, 25200, 16]

            inputs = inputs[inputs[:, :, 4] > iouThreshold]
            max_class_tensor = torch.transpose(torch.argmax(inputs[:, 5:], dim=1).unsqueeze(0), 0, 1)
            outputs = torch.cat((inputs[:, :5], max_class_tensor), 1).unsqueeze(0)
            return outputs


    class PostCD2(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def iou(self, box1, box2):
            # input [x,y,w,h]
            min_x = torch.min(box1[0], box2[0])
            max_x = torch.max(box1[0] + box1[2], box2[0] + box2[2])
            min_y = torch.min(box1[1], box2[1])
            max_y = torch.max(box1[1] + box1[3], box2[1] + box2[3])

            union_w = max_x - min_x
            union_h = max_y - min_y
            inter_w = box1[2] + box2[2] - union_w
            inter_h = box1[3] + box2[3] - union_h

            if inter_w <=0 or inter_h <=0:
                return .0
            else:
                area1 = box1[2] * box1[3]
                area2 = box2[2] * box2[3]
                inter_area = inter_w * inter_h
                union_area = area1 + area2 - inter_area
                return float(inter_area / union_area)

        # def forward(self, inputs: torch.Tensor, confThreshold: float, iouThreshold: float):
        #     # shape e.g., "shape": [1, 25200, 16]
        #     # [x1,y1,w1,h1,conf1,class softmax11]
        #
        #     inputs = inputs[inputs[:, :, 4] > confThreshold]
        #     max_class_tensor = torch.transpose(torch.argmax(inputs[:, 5:], dim=1).unsqueeze(0), 0, 1) #argmax -> max index
        #     outputs = torch.cat((inputs[:, :5], max_class_tensor), 1).unsqueeze(0) # x,y,w,h,conf,class
        #     names = ['engname','email','koraddress','korname','engposition','engaddress','korposition','web','korcompany','engcompany','call']  # class names
        #
        #     nms_outputs : Dict[str, torch.Tensor] = {}
        #     save_first_idx = 0
        #     outputs_0 = outputs[0]
        #
        #     for i in range(outputs.shape[1]):
        #         e = i+1
        #
        #         if (e == outputs.shape[1]) or (self.iou(outputs_0[i, :4], outputs_0[e, :4]) <= iouThreshold): #위치가 다른 박스가 나오면
        #             max_conf_idx = save_first_idx + torch.argmax(outputs_0[save_first_idx:e, 4]) #이전까지 박스에서 conf 높은 박스의 index 얻어
        #             is_already = False
        #
        #             for key in nms_outputs: #이미 넣은 박스와 검사한다
        #                 if self.iou(nms_outputs[key][:4], outputs_0[max_conf_idx, :4]) > iouThreshold: #이미넣은 박스와 겹치는 경우
        #                     is_already = True
        #
        #                     if nms_outputs[key][4] < outputs_0[max_conf_idx, 4]: #이미 넣은 박스와 비교해서 conf 높은 걸 넣는다
        #                         del nms_outputs[key]
        #
        #                         object_class = names[int(outputs_0[max_conf_idx, 5])] + '+' + str(max_conf_idx)
        #                         nms_outputs[object_class] = outputs_0[max_conf_idx]
        #                     break
        #
        #
        #             if not is_already: #겹치지 않는 경우
        #                 object_class = names[int(outputs_0[max_conf_idx, 5])] + '+' + str(max_conf_idx)
        #                 nms_outputs[object_class] = outputs_0[max_conf_idx]
        #
        #             save_first_idx = e
        #
        #     res : Dict[str, List[int]] = {}
        #     for key in nms_outputs:
        #         result: List[int] = []
        #         for i in range(4): # tolist() for nms_outputs[key][0:4]
        #             result.append(round(nms_outputs[key][i].item()))
        #         res[key] = result # conf 제거
        #
        #     return res

        def forward(self, inputs: torch.Tensor, confThreshold: float, iouThreshold: float):
            # shape e.g., "shape": [1, 25200, 16]
            # [x1,y1,w1,h1,conf1,class softmax11]

            inputs = inputs[inputs[:, :, 4] > confThreshold]
            max_class_tensor = torch.transpose(torch.argmax(inputs[:, 5:], dim=1).unsqueeze(0), 0, 1) #argmax -> max index
            outputs = torch.cat((inputs[:, :5], max_class_tensor), 1).unsqueeze(0) # x,y,w,h,conf,class

            nms_outputs = [] # [[x,y,w,h,conf,class],[...],...]
            save_first_idx = 0
            outputs_0 = outputs[0]

            for i in range(outputs.shape[1]):
                e = i+1

                if (e == outputs.shape[1]) or (self.iou(outputs_0[i, :4], outputs_0[e, :4]) <= iouThreshold): #위치가 다른 박스가 나오면
                    max_conf_idx = save_first_idx + torch.argmax(outputs_0[save_first_idx:e, 4]) #이전까지 박스에서 conf 높은 박스의 index 얻어
                    is_already = False

                    for j in range(len(nms_outputs)): #이미 넣은 박스와 검사한다
                        if self.iou(nms_outputs[j][:4], outputs_0[max_conf_idx, :4]) > iouThreshold: #이미넣은 박스와 겹치는 경우
                            is_already = True

                            if nms_outputs[j][4] < outputs_0[max_conf_idx, 4]: #이미 넣은 박스와 비교해서 conf 높은 걸 넣는다
                                nms_outputs[j] = outputs_0[max_conf_idx]
                            break

                    if not is_already: #겹치지 않는 경우
                        nms_outputs.append(outputs_0[max_conf_idx])

                    save_first_idx = e

            return nms_outputs


    # pcd = PostCD()
    pcd = PostCD2()
    scripted_model = torch.jit.script(pcd)

    data = [[247.37535095214844, 525.0855712890625, 129.0428924560547, 31.98747444152832, 0.8668637275695801,
             0.0027807741425931454, 0.002542165108025074, 0.00628146156668663, 0.0026534167118370533,
             0.002410427201539278, 0.0035515064373612404, 0.00348650268279016, 0.006360366474837065,
             0.002872081473469734, 0.0036974542308598757, 0.9905574321746826],
            [247.37535095214844, 525.0855712890625, 129.0428924560547, 31.98747444152832, 0.8668637275695801,
             0.9027807741425931454, 0.002542165108025074, 0.00628146156668663, 0.0026534167118370533,
             0.002410427201539278, 0.0035515064373612404, 0.00348650268279016, 0.006360366474837065,
             0.002872081473469734, 0.0036974542308598757, 0.005574321746826],
            [247.37535095214844, 525.0855712890625, 129.0428924560547, 31.98747444152832, 0.1, 0.0027807741425931454,
             0.002542165108025074, 0.00628146156668663, 0.0026534167118370533, 0.002410427201539278,
             0.0035515064373612404, 0.00348650268279016, 0.006360366474837065, 0.002872081473469734,
             0.0036974542308598757, 0.9905574321746826],
            [247.37535095214844, 525.0855712890625, 129.0428924560547, 31.98747444152832, 0.1, 0.0027807741425931454,
             0.9, 0.00628146156668663, 0.0026534167118370533, 0.002410427201539278, 0.0035515064373612404,
             0.00348650268279016, 0.006360366474837065, 0.002872081473469734, 0.0036974542308598757, 0.0004000],
            [247.37535095214844, 525.0855712890625, 129.0428924560547, 31.98747444152832, 0.1, 0.0027807741425931454,
             0.9, 0.00628146156668663, 0.0026534167118370533, 0.002410427201539278, 0.0035515064373612404,
             0.00348650268279016, 0.006360366474837065, 0.002872081473469734, 0.0036974542308598757, 0.0004000],
            [247.37535095214844, 525.0855712890625, 129.0428924560547, 31.98747444152832, 0.1, 0.0027807741425931454,
             0.9, 0.00628146156668663, 0.0026534167118370533, 0.002410427201539278, 0.0035515064373612404,
             0.00348650268279016, 0.006360366474837065, 0.002872081473469734, 0.0036974542308598757, 0.0004000],
            [2470.37535095214844, 5250.0855712890625, 129.0428924560547, 31.98747444152832, 0.4, 0.0027807741425931454,
             0.9, 0.00628146156668663, 0.0026534167118370533, 0.002410427201539278, 0.0035515064373612404,
             0.00348650268279016, 0.006360366474837065, 0.002872081473469734, 0.0036974542308598757, 0.0004000],
            ]

    x_data = torch.tensor(data).unsqueeze(0)

    print(x_data)
    print(x_data.shape, end='\n\n')

    outputs = scripted_model(x_data, 0.3, 0.3)

    print(outputs)
    # print(outputs.shape)

    scripted_model.save("post_cd2_111.pt")

    optimized_scripted_module = optimize_for_mobile(scripted_model)
    optimized_scripted_module._save_for_lite_interpreter("post_cd2_111.ptl")


def print_space(img):
    img = cv2.resize(img, (320, 48))

    img_width = int(img.shape[1])
    color_range = 20
    center_row = int(img.shape[0]/2)

    list_represent_color = []
    list_count_one = []
    count_one = 0
    print('height: ', img.shape[0]) # paddle ocr input 으로 resize하기
    for col in range(0, img_width):
        c1 = [img[center_row, col, 0] - color_range, img[center_row, col, 0] + color_range]

        represent_color_num = len(np.unique(np.logical_or(c1[0] > img[:, col, 0], c1[1] < img[:, col, 0])))

        list_represent_color.append(represent_color_num)

        # count 1
        if represent_color_num == 1:
            count_one +=1
        elif represent_color_num != 1 and count_one != 0:
            list_count_one.append(count_one)
            count_one = 0

    # print(list_count_one)
    # print(list_represent_color)
    average_of_space_pixel([list_count_one])
    return list_count_one


def average_of_space_pixel(lists_count_one):
    # 제일 앞뒤 스페이스는 제거

    elements = []
    for l in lists_count_one:
        for i, e in enumerate(l):
            if i!=0 or i!=len(lists_count_one)-1:
                elements.append(e)

    print('sum: ', sum(elements))
    print('max: ', max(elements))
    print('min: ', min(elements))
    print('avg: ', sum(elements)/len(elements))


def img_write(outputdir, img, st, et):
    '''
    st means a start point
    et means a end point

    '''
    newimg = np.zeros((img.shape[0], et - st, 3), np.uint8)
    newimg[:] = img[:, st:et, :]
    cv2.imwrite('./' + outputdir + '/' + str(et) + '.jpg', newimg)


def cutdetector_rough(img, outputdir):
    # 간격이 넓은 부분에서 자음 모음 분리
    if os.path.exists('./' + outputdir):
        shutil.rmtree('./' + outputdir)

    IMAGE_SIZE = 640
    if img.shape[1] < IMAGE_SIZE * 0.6:
        return

    width_range = int(img.shape[1]/10) # robust
    color_range = 20
    center_row = int(img.shape[0]/2)
    represent_color_list = []
    pts_list = []
    count_one = 0

    if width_range > img.shape[1]-width_range:
        return

    for col in range(0+width_range, img.shape[1]-width_range):
        c1 = [img[center_row, col, 0] - color_range, img[center_row, col, 0] + color_range]
        # c2 = [img[center_row, col, 1] - color_range, img[center_row, col, 1] + color_range]
        # c3 = [img[center_row, col, 2] - color_range, img[center_row, col, 2] + color_range]

        represent_color_num = len(np.unique( np.logical_or (c1[0] > img[:, col, 0], c1[1] < img[:,col,0])))
        if represent_color_num == 1:
            count_one += 1
        elif count_one != 0:
            # list one-count is not included for this hack algorithm,
            # but it is not a problem for logic
            represent_color_list.append(count_one)
            pts_list.append([col-count_one,col])
            count_one = 0


    if not represent_color_list:
        return

    min_width_space=min(represent_color_list)
    max_width_space=max(represent_color_list)
    middle_width_space=(min_width_space+max_width_space)/2
    # print(sum(represent_color_list)/len(represent_color_list) < middle_width_space)
    tmp = []

    for i, rc in enumerate(represent_color_list):
        if rc > middle_width_space and i%4==1:
            tmp.append(pts_list[i])


    if not tmp:
        return

    os.mkdir('./' + outputdir)

    for l in range(len(tmp)):
        # img_write(outputdir, img, tmp[l][0], tmp[l][1]) # only space
        if l == 0:
            img_write(outputdir, img, 0, tmp[l][0])
        else:
            img_write(outputdir, img, tmp[l-1][1], tmp[l][0])
        if l == len(tmp)-1:
            img_write(outputdir, img, tmp[l][1], img.shape[1])


def run_cut_detector():
    path = './runs/detect/exp23/crops/'
    image_dir = os.listdir(path)

    avg_list = []

    for folder_name in image_dir:
        folder_path = os.path.join(path, folder_name)
        images_name = os.listdir(folder_path)

        for test_image in images_name:
            if 'address' in test_image:
                test_image_name = os.path.join(folder_path, test_image)
                img = cv2.imread(test_image_name)

                # cutdetector_rough(img, os.path.join(folder_path, 'cut'))
                test_CropWithCutDetector(img, os.path.join(folder_path, 'cut'))

                # new_list = print_space(img)                 # for data analysis
                # avg_list.append(new_list)                # for data analysis

    # average_of_space_pixel(avg_list)                # for data analysis


def wrapper_cutdetector():
    class CropWithCutDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def cutdetector_rough(tensor):
            pass

        def forward(self, inputs: torch.Tensor, points: List[int]):
            IMAGE_SIZE = 640 # This is the yolo input size

            if points[2]-points[0] >= IMAGE_SIZE * 0.6:
                return cutdetector_rough(inputs[:, points[1]:points[3], points[0]:points[2]])
            else:
                return inputs[:, points[1]:points[3], points[0]:points[2]]  # y1,x1,y2,x2


    crop = CropWithCutDetector()
    scripted_model = torch.jit.script(crop)
    scripted_model.save("crop_with_cut_detector.pt")

    optimized_scripted_module = optimize_for_mobile(scripted_model)
    optimized_scripted_module._save_for_lite_interpreter("crop_with_cut_detector.ptl")



def test_CropWithCutDetector(img, outputdir):
    # 간격이 넓은 부분에서 자음 모음 분리
    if os.path.exists('./' + outputdir):
        shutil.rmtree('./' + outputdir)

    IMAGE_SIZE = 640
    if img.shape[1] < IMAGE_SIZE * 0.6:
        return

    img_tensor = torch.as_tensor(np.expand_dims(img.astype("float32").transpose(2, 0, 1), axis=0)) # C, H, W
    # print(img_tensor.shape)

    width_range = int(img_tensor.shape[3]/10) # robust
    color_range = 20
    center_row = int(img_tensor.shape[2]/2)
    represent_color_list = []
    pts_list = []
    count_one = 0

    if width_range > img_tensor.shape[3]-width_range:
        return

    for col in range(0+width_range, img_tensor.shape[3]-width_range):
        c1 = [img_tensor[:, 0, center_row, col] - color_range, img_tensor[:, 0,center_row, col] + color_range]
        # c2 = [img[center_row, col, 1] - color_range, img[center_row, col, 1] + color_range]
        # c3 = [img[center_row, col, 2] - color_range, img[center_row, col, 2] + color_range]

        represent_color_num = len(torch.unique( torch.logical_or (c1[0] > img_tensor[:, :, 0, col], c1[1] < img_tensor[:, :, 0, col])))
        if represent_color_num == 1:
            count_one += 1
        elif count_one != 0:
            # list one-count is not included for this hack algorithm,
            # but it is not a problem for logic
            represent_color_list.append(count_one)
            pts_list.append([col-count_one,col])
            count_one = 0


    if not represent_color_list:
        return

    min_width_space=min(represent_color_list)
    max_width_space=max(represent_color_list)
    middle_width_space=(min_width_space+max_width_space)/2
    # print(sum(represent_color_list)/len(represent_color_list) < middle_width_space)
    tmp = []

    for i, rc in enumerate(represent_color_list):
        if rc > middle_width_space and i%4==1:
            tmp.append(pts_list[i])


    if not tmp:
        return

    os.mkdir('./' + outputdir)

    for l in range(len(tmp)):
        # img_write(outputdir, img, tmp[l][0], tmp[l][1]) # only space
        if l == 0:
            img_write(outputdir, img, 0, tmp[l][0])
        else:
            img_write(outputdir, img, tmp[l-1][1], tmp[l][0])
        if l == len(tmp)-1:
            img_write(outputdir, img, tmp[l][1], img.shape[1])


if __name__ == "__main__":
    # yolov4_datasets_to_yolov5()
    # plot_train_results()
    # labelme_to_yolov5()
    # labelme_to_yolov4()
    # post_cd()
    # labelme_to_yolov5_crops()
    run_cut_detector()