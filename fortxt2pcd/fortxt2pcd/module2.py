import os
import sys



def Modifyprefix(Path,oldcontent,newcontent):
    
    all_file_list = os.listdir(Path)                            #列出指定目录下的所有文件
    for file_name in all_file_list:
        currentdir =os.path.join(Path, file_name)               #连接指定的路径和文件名or文件夹名字
        if os.path.isdir(currentdir):                           #如果当前路径是文件夹，则跳过
            Modifyprefix(currentdir,oldcontent,newcontent)
        fname = os.path.splitext(file_name)[0]                  #分解出当前的文件路径名字
        ftype = os.path.splitext(file_name)[1]                  #分解出当前的文件扩展名
        if oldcontent in fname:
            fdcount[0]+=1
            replname =fname.replace(oldcontent,newcontent)      #将原文件名中的'oldcontent'字符串内容全替换为'newcontent'字符串内容
            newname = os.path.join(Path,replname+ftype)         #文件路径与新的文件名字+原来的扩展名
            os.rename(currentdir,newname)                       #重命名

#批量修改文件扩展名(后缀)
def Modifypostfix(Path,oldftype,newftype):
    all_file_list = os.listdir(Path)          #列出指定目录下的所有文件
    for file_name in all_file_list:
        currentdir =os.path.join(Path,file_name)
        if os.path.isdir(currentdir):                    #迭代
            Modifypostfix(currentdir,oldftype,newftype)
        fname = os.path.splitext(file_name)[0]
        ftype = os.path.splitext(file_name)[1]
        if oldftype in ftype[1:]:                         #找到需要修改的扩展名
            typecount[0]+=1
            ftype=ftype.replace(oldftype,newftype)
            newname = os.path.join(Path,fname+ftype) #文件路径与原来的文件名字+新的扩展名
            os.rename(currentdir,newname)               #重命名

def Useage():
    print ("\n[+] 用法: python Modifer.py  [指定目录] [选项] [参数1] [参数2]","utf-8")
    print ("[+] 选项 [-fd]  :批量修改目录和文件名               [参数1]: 需要替换的字符      [参数2]:替换字符串","utf-8")
    print ("[+] 选项 [-fp]  :批量修改文件后缀名                 [参数1]: 原后缀名            [参数2]:需要替换的后缀名","utf-8")
    print ("[+] 选项 [-all] :批量修改目录、文件名和文件后缀名   [参数1]: 需要替换的字符部分  [参数2]:替换字符串","utf-8")
    print (r"[+] 用法示例:python Modifer.py D:\files -fp txt data","utf-8")

if __name__=="__main__":
    typecount=[0]
    fdcount=[0]
    if len(sys.argv)==2 and "-h" in sys.argv[1]:
        Useage()
        sys.exit()
    elif len(sys.argv) !=5:
        print ("\n[+] 参数错误 !\n","utf-8")
        print ("[+] 用 -h 或--help 参数查看Modifer.py用法","utf-8")
        sys.exit()
    elif os.path.isdir(sys.argv[1]) is False:
        print ("\n[+] 指定目录错误 ! 请检查输入路径是否正确,路径中不能有空格\n","utf-8")
        print ("[+] 用 -h 或--help 参数查看Modifer.py用法","utf-8")
        sys.exit()
    elif sys.argv[2]=="-fd":
        Modifyprefix(sys.argv[1],sys.argv[3],sys.argv[4])
        print ("\n[+] Modifer.py    Build by LandGrey","utf-8")
        print ("[+] 完成 !","utf-8")
        print ("[+] 共修改%s个目录和文件名"%fdcount[0],"utf-8")
    elif sys.argv[2]=="-fp":
        Modifypostfix(sys.argv[1],sys.argv[3],sys.argv[4])
        print ("\n[+] Modifer.py    Build by LandGrey","utf-8")
        print ("[+] 完成 !","utf-8")
        print ("[+] 共修改%s个后缀名"%typecount[0],"utf-8")
    elif sys.argv[2]=="-all":
        Modifypostfix(sys.argv[1],sys.argv[3],sys.argv[4])
        Modifyprefix(sys.argv[1],sys.argv[3],sys.argv[4])
        print ("\n[+] Modifer.py    Build by LandGrey","utf-8")
        print ("[+] 完成 !","utf-8")
        print ("[+] 共修改%s个目录名、文件名和后缀名"%(typecount[0]+fdcount[0]),"utf-8")
    else:
        print ("\n[+] 选项错误 !\n","utf-8")
        print ("[+] 用 -h 或--help 参数查看Modifer.py用法","utf-8")
        sys.exit()