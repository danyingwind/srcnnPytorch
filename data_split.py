import shutil
import os
 
def remove_file(old_path, new_path):
    # print(old_path)
    # print(new_path)
    filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    # print(filelist)
    old_path_parts = old_path.split("/")
    file2 = old_path_parts[4]
    for file in filelist:
        src = os.path.join(old_path, file)
        if(file[19] == '7' or file[19] == '8'):
            dst = os.path.join(new_path, "for_eval_dataset",file2, file)
            shutil.move(src, dst)
        if(file[19] >= '0' and file[19] <= '6'):
            dst = os.path.join(new_path, "for_train_dataset",file2, file)
            shutil.move(src, dst)
            # print('src:', src)
            # print('dst:', dst)
        


 
if __name__ == '__main__':
    remove_file(r"/home/wangdanying/SRCNN/png_R5_texture", r"/home/wangdanying/SRCNN")
 