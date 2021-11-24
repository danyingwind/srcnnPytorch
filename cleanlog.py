import os

mypath = "/home/wangdanying/SRCNN/srcnnPytorch/debug/trainLog/x1"
#mypath = "/home/wangdanying/SRCNN/srcnnPytorch/debug/1"
trainlogs = []
for root, dirs, files in os.walk(mypath):
    for i in range(len(files)):
        #print(files[i])
        file_path = root+'/'+files[i]  
        trainlogs.append(file_path)
        
for trainlog in trainlogs:
    if os.path.exists(trainlog):
        os.remove(trainlog)