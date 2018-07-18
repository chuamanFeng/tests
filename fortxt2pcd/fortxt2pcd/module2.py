import os
import sys



def Modifyprefix(Path,oldcontent,newcontent):
    
    all_file_list = os.listdir(Path)                            #�г�ָ��Ŀ¼�µ������ļ�
    for file_name in all_file_list:
        currentdir =os.path.join(Path, file_name)               #����ָ����·�����ļ���or�ļ�������
        if os.path.isdir(currentdir):                           #�����ǰ·�����ļ��У�������
            Modifyprefix(currentdir,oldcontent,newcontent)
        fname = os.path.splitext(file_name)[0]                  #�ֽ����ǰ���ļ�·������
        ftype = os.path.splitext(file_name)[1]                  #�ֽ����ǰ���ļ���չ��
        if oldcontent in fname:
            fdcount[0]+=1
            replname =fname.replace(oldcontent,newcontent)      #��ԭ�ļ����е�'oldcontent'�ַ�������ȫ�滻Ϊ'newcontent'�ַ�������
            newname = os.path.join(Path,replname+ftype)         #�ļ�·�����µ��ļ�����+ԭ������չ��
            os.rename(currentdir,newname)                       #������

#�����޸��ļ���չ��(��׺)
def Modifypostfix(Path,oldftype,newftype):
    all_file_list = os.listdir(Path)          #�г�ָ��Ŀ¼�µ������ļ�
    for file_name in all_file_list:
        currentdir =os.path.join(Path,file_name)
        if os.path.isdir(currentdir):                    #����
            Modifypostfix(currentdir,oldftype,newftype)
        fname = os.path.splitext(file_name)[0]
        ftype = os.path.splitext(file_name)[1]
        if oldftype in ftype[1:]:                         #�ҵ���Ҫ�޸ĵ���չ��
            typecount[0]+=1
            ftype=ftype.replace(oldftype,newftype)
            newname = os.path.join(Path,fname+ftype) #�ļ�·����ԭ�����ļ�����+�µ���չ��
            os.rename(currentdir,newname)               #������

def Useage():
    print ("\n[+] �÷�: python Modifer.py  [ָ��Ŀ¼] [ѡ��] [����1] [����2]","utf-8")
    print ("[+] ѡ�� [-fd]  :�����޸�Ŀ¼���ļ���               [����1]: ��Ҫ�滻���ַ�      [����2]:�滻�ַ���","utf-8")
    print ("[+] ѡ�� [-fp]  :�����޸��ļ���׺��                 [����1]: ԭ��׺��            [����2]:��Ҫ�滻�ĺ�׺��","utf-8")
    print ("[+] ѡ�� [-all] :�����޸�Ŀ¼���ļ������ļ���׺��   [����1]: ��Ҫ�滻���ַ�����  [����2]:�滻�ַ���","utf-8")
    print (r"[+] �÷�ʾ��:python Modifer.py D:\files -fp txt data","utf-8")

if __name__=="__main__":
    typecount=[0]
    fdcount=[0]
    if len(sys.argv)==2 and "-h" in sys.argv[1]:
        Useage()
        sys.exit()
    elif len(sys.argv) !=5:
        print ("\n[+] �������� !\n","utf-8")
        print ("[+] �� -h ��--help �����鿴Modifer.py�÷�","utf-8")
        sys.exit()
    elif os.path.isdir(sys.argv[1]) is False:
        print ("\n[+] ָ��Ŀ¼���� ! ��������·���Ƿ���ȷ,·���в����пո�\n","utf-8")
        print ("[+] �� -h ��--help �����鿴Modifer.py�÷�","utf-8")
        sys.exit()
    elif sys.argv[2]=="-fd":
        Modifyprefix(sys.argv[1],sys.argv[3],sys.argv[4])
        print ("\n[+] Modifer.py    Build by LandGrey","utf-8")
        print ("[+] ��� !","utf-8")
        print ("[+] ���޸�%s��Ŀ¼���ļ���"%fdcount[0],"utf-8")
    elif sys.argv[2]=="-fp":
        Modifypostfix(sys.argv[1],sys.argv[3],sys.argv[4])
        print ("\n[+] Modifer.py    Build by LandGrey","utf-8")
        print ("[+] ��� !","utf-8")
        print ("[+] ���޸�%s����׺��"%typecount[0],"utf-8")
    elif sys.argv[2]=="-all":
        Modifypostfix(sys.argv[1],sys.argv[3],sys.argv[4])
        Modifyprefix(sys.argv[1],sys.argv[3],sys.argv[4])
        print ("\n[+] Modifer.py    Build by LandGrey","utf-8")
        print ("[+] ��� !","utf-8")
        print ("[+] ���޸�%s��Ŀ¼�����ļ����ͺ�׺��"%(typecount[0]+fdcount[0]),"utf-8")
    else:
        print ("\n[+] ѡ����� !\n","utf-8")
        print ("[+] �� -h ��--help �����鿴Modifer.py�÷�","utf-8")
        sys.exit()