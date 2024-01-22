import os


def check_donut_duplicate(filepath):
    lstdir = os.listdir(filepath)
    for root, dir, files in os.walk(filepath):
        if len(files) > 1:
            image1 = files[0]
            image2 = files[1]
            image1 = int(image1.replace('.jpg', ''))
            image2 = int(image2.replace('.jpg', ''))

            # print(image1, image2)

            if image1 == 946 or image2 == 946:
                print(root)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filepath = 'D:\datasets\donut_data\duplicateimages\\'
    check_donut_duplicate(filepath)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
