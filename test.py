import os 
TrainPath = 'lsvc2017/lsvc_val.txt'
f = open(TrainPath)
fout = open('new_val.txt','w')
for line in f.readlines(): 
    contents = line.strip().split(',')
    datapath = '/workspace/data/trainval/' + contents[0] +'_pool5_senet.binary'
    if not os.path.exists(datapath):
        print(contents[0])
        continue
    for i in range(len(contents)-1):
        fout.write(contents[0] + ',' + contents[i+1] + '\n')
fout.close()
f.close()
