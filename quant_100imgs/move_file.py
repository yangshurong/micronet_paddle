import os
import shutil
pre=r'C:\Users\Administrator\Desktop\quant_100imgs\quant_100imgs'
loca=os.path.join(pre,'val_list.txt')
with open(loca, "r", encoding="utf-8") as f:
  for x in f.readlines():
    x=x.strip('\n')
    x=x.split(' ')

    new_locate=os.path.join(pre,'new_val',x[1])
    old_locate=os.path.join(pre,x[0])
    if not os.path.exists(new_locate):
      os.makedirs(new_locate)  
    shutil.copy(old_locate,new_locate)
      