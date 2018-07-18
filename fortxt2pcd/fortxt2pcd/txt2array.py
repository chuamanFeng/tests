import numpy as np
import os
import sys
import time


def asc2txt(filefold): 
  for filename in filefold:      
      portion = os.path.splitext(filename)
     
      if portion[1] ==".asc":
         newname = portion[0]+".txt"        
         os.rename(filename,newname)
         

def origdata2array(filepath):    
        
   files = os.listdir(filepath) 
   asc2txt(files)     
   cnt=0  
   files = os.listdir(filepath) 
   for filename in files:   
     if os.path.splitext(filename)[1]==(".txt"):
       
        file = open(filename)  
        list_arr = file.readlines()
        l = len(list_arr)  
        listi=[]

        for i in range(l):                   #lens in one file
            if list_arr[i].find('S')<=-1:
                list_arr[i] = list_arr[i].split()   
            if len(list_arr[i])==6:
                listi.append(list_arr[i])  
              
            #list_arr[i]=np.array(list_arr[i])    
                    
        outi=np.expand_dims(np.array(listi),0)
        cnt+=1        #all effective lines in one file finished
        if cnt==1:
            out=outi
        else:
            out=np.hstack((out,outi))

   print(cnt,'files have been pre-cooked ')
   if cnt!=0:
     out=out.astype(float)     
   else:
       out=None
   return(out)

if __name__=="__main__":
    
    os.chdir('C:/Aaa') 
    BASE_DIR=os.getcwd()
    print(BASE_DIR)
    now=time.time()
    #asc2txt(BASE_DIR)
    #files = os.listdir(BASE_DIR)
    out=origdata2array(BASE_DIR)      #15sec for 11 files
    print(out)
    print(time.time()-now)     