import os 
import numpy as np 
import pdb

def search_index(lmk,x,y):
    # pdb.set_trace()
    index_x = np.where(np.around(lmk[:,0],decimals=4)==np.around(x,decimals=4))[0]
    index_y = np.where(np.around(lmk[index_x,1],decimals=4)==np.around(y,decimals=4))[0]
    return index_x[index_y[0]]

A_path = r"C:\Users\Administrator\Desktop\BIGOWORK\0906\lmk1.npy"
B_path = r"C:\Users\Administrator\Desktop\BIGOWORK\SHADER\4.obj"

lmk1 = np.load(A_path)
lmk1 = np.reshape(lmk1,(-1,3))[:,:2]

# get index mapping
index_dict = {}
trangle_list = []
i = 1
with open(B_path,"r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("v "):
            x,y = list(map(lambda t: float(t),line.split()[1:3]))
            # try:
            index_dict[i] = search_index(lmk1,x,y)
            # except:
            #     continue
            
            i += 1


        # get trangle index
        if line.startswith("f"):
            try:
                v1,v2,v3 = list(map(lambda x: index_dict[int(x.split("//")[0])],line.split()[1:]))
                trangle_list.append([v1,v2,v3])
            except:
                continue

np.save(r"C:\Users\Administrator\Desktop\BIGOWORK\0906\index",trangle_list)
print("!!!!",len(trangle_list))