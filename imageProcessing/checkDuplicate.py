from PIL import Image
import os
import hashlib
import shutil

def find_duplicate_images(rootdir):

    img_path = rootdir + '\\all_businesscards'
    hash_dict = {}
    duplicates = []
    hash_list = []
    for subdir, dirs, files in os.walk(img_path):
        for file in files:
            filepath = os.path.join(subdir, file)

            # Open the image and calculate its hash
            with open(filepath, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            # Check if the hash already exists
            if img_hash in hash_dict:
                duplicates.append(filepath)
                hash_dict[img_hash].append(filepath)
            else:
                hash_dict[img_hash] = [filepath]
                hash_list.append(img_hash)

    save_duplicate_images(hash_dict, hash_list, rootdir)

    return hash_dict, duplicates

def save_notduplicate_images(hash_dict, rootdir):
    image_dir = rootdir + '\\removeDuplicate'
    origin_dir = rootdir + '\\all_businesscards'

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for hash, images in hash_dict.items():
        if len(images) == 2:
            img1, img2 = images
            img_num1 = int(img1.split('\\')[4].replace('.jpg', ''))
            img_num2 = int(img2.split('\\')[4].replace('.jpg', ''))
            origin_img_dir = os.path.join(origin_dir, str(min(img_num1, img_num2)) + '.jpg')
            new_img_dir = os.path.join(image_dir, str(min(img_num1, img_num2)) + '.jpg')

            shutil.copyfile(origin_img_dir, new_img_dir)

        elif len(images) == 1:
            img1 = images[0]
            img_num1 = int(img1.split('\\')[4].replace('.jpg', ''))
            origin_img_dir = os.path.join(origin_dir, str(img_num1) + '.jpg')
            new_img_dir = os.path.join(image_dir, str(img_num1) + '.jpg')

            shutil.copyfile(origin_img_dir, new_img_dir)



def save_duplicate_images(hash_dict, hash_list, rootdir):
    copypath = rootdir + "\duplicateimages"


    if not os.path.exists(copypath):
        os.makedirs(copypath)

    for i in range(len(hash_list)):
        copysubpath = os.path.join(copypath, str(i))

        if not os.path.exists(copysubpath):
            os.makedirs(copysubpath)


        for imgpath in hash_dict[hash_list[i]]:
            imgname = imgpath.split('\\')[4]
            copyimgpath = os.path.join(copysubpath, imgname)
            shutil.copyfile(imgpath, copyimgpath)
        # print(i, hash_dict[hash_list[i]])

if __name__ == '__main__':
    rootdir = 'D:\\datasets\\donut_data' # root directory
    hash_dict, duplicates = find_duplicate_images(rootdir)
    if duplicates:
        print("Duplicate images found:", duplicates)
        save_notduplicate_images(hash_dict, rootdir)
    else:
        print("No duplicate images found.")
    print(len(duplicates))