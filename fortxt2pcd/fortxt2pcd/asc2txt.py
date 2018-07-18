import os
import sys
import codecs  

def trans_tail(filefold):
  for filename in filefold:      
      portion = os.path.splitext(filename)
      print(portion)
  
      if portion[1] ==".asc":
         newname = portion[0]+".txt"
         os.chdir("C:\Aaa")
         os.rename(filename,newname)

 
if __name__=="__main__":

    files = os.listdir("C:\Aaa")
    trans_tail(files)
