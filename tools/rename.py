import os
 
import re
 
filepath = "/home/wangdanying/SRCNN/png_R1_lr_train"

#要删除的名字字符串 
delect = "rec_"
if __name__ == '__main__':
    if not os.path.exists(filepath):
        print("目录不存在!!")
        os._exit(1)
    filenames = os.listdir(filepath)
    print("文件数目为%i" %len(filenames))
    count = 0
    for name in filenames:
        newname = re.sub(delect, '', name)
        os.rename(filepath + '/' + name, filepath + '/' + newname)
        count += 1
        if count % 100 == 0:
            print("第%i个文件已经改名完成" %count)
 

