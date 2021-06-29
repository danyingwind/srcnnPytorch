import os
import shutil
 
"""
一个文件夹中有多个文件，把所有文件分成 num 份，新建文件夹放入
"""
 
#文件存放地址
file_path = '/home/wangdanying/SRCNN/png_R1_hr'
test_file = '/home/wangdanying/SRCNN/png_R1_hr_test'
eval_file = '/home/wangdanying/SRCNN/png_R1_hr_eval'
train_file = '/home/wangdanying/SRCNN/png_R1_hr_train'
os.mkdir(test_file)
os.mkdir(eval_file)
os.mkdir(train_file)
images_list = os.listdir(file_path)
for img in images_list:
    GOFid = img.split("_")[2].split("GOF")[1]
    old_path = os.path.join(file_path, img)
    if(GOFid == '0'):
        new_file = test_file
    elif(GOFid == '1'):
        new_file = eval_file
    else:
        new_file = train_file

        
    new_path = os.path.join(new_file, img)
    shutil.copy(old_path, new_path)

