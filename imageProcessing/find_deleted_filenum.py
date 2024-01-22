import os

# 중복되어 삭제된 파일명 찾는 코드
def find_deleted_filenum(filepath):
    images = os.listdir(filepath)
    dictimg = {}
    deletedimgs = {}

    for img in images:
        dictimg[img] = ''

    for i in range(1282):
        imgkey = str(i) + '.jpg'
        if imgkey not in dictimg:
            deletedimgs[i] = ''

    return deletedimgs


# 중복파일 제거하고 labeling check 용 txt 파일 만드는 코드
def create_labeling_text_remove_duplicate(deletedimgs):
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

    # 라벨링 체크용 txt 파일 만드는 코드
    with open("D:\datasets\donut_data\\labelcheck.txt", "w", encoding="UTF-8") as file:
        for i in range(1282, 2543):
            if donut_dict[i] and i not in deletedimgs:
                file.write(str(i) + '.jpg\n')
                for category, val in donut_dict[i]:
                    file.write(category + '\t' + val + '\n')


if __name__ == '__main__':
    filepath = 'D:\datasets\donut_data\\all_businesscards\\'
    deletedimgs = find_deleted_filenum(filepath)
    # create_labeling_text_remove_duplicate(deletedimgs)