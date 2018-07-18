from sys import argv
#import time

filename = "U3DRGB.txt"
print ("the input file name is:%r." %filename)

#start = time.time()
print("open the file.....")
file = open(filename,"r+")
count = 0

for line in file:
    count=count+1
print("size is %d" %count)
file.close()

#output = open("out.pcd","w+")
f_prefix = filename.split('.')[0]
output_filename = '{prefix}.pcd'.format(prefix=f_prefix)
output = open(output_filename,"w+")

list = ['# .PCD v.5 - Point Cloud Data file format\n','VERSION .5\n','FIELDS x y z\n','SIZE 4 4 4\n','TYPE F F F\n','COUNT 1 1 1\n']
output.writelines(list)
output.write('WIDTH ')
output.write(str(count))
output.write('\nHEIGHT ')
output.write(str(1))
output.write('\nPOINTS ')
output.write(str(count))
output.write('\nDATA ascii\n')

file1 = open(filename,"r")
all = file1.read()
output.write(all)
output.close()
file1.close()

#end = time.time()
#print("run time is:",end-start)
#------------------------------------------------------------------------------#
def eachFile(filepath):
    pathDir = os.listdir(filepath)      #获取当前路径下的文件名，返回List
    for s in pathDir:
        newDir=os.path.join(filepath,s)     #将文件命加入到当前文件路径后面
        if os.path.isfile(newDir) :         #如果是文件
            if os.path.splitext(newDir)[1]==".txt":  #判断是否是txt
                readFile(newDir)                     #读文件
                pass
            else:
                eachFile(newDir)      