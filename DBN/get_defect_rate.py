import os
import pandas as pd


def get_defect_rate():
    path = "./PROMISE/promise_data"
    projects =[]
    projects_dict = {}
    for dirpath, dirs, files in os.walk(path):
        print(dirs)
        projects = dirs
        break
    for i in projects:
        project_path = os.path.join(path,i)
        for dirpath,dirs,files in os.walk(project_path):
            source_files = [f for f in files if is_csv(f)]
            for j in source_files:
                csv_path = os.path.join(project_path,j)

                df = pd.read_csv(csv_path)
                projects_dict[csv_path] =len(df[df["bugs"]!=0])/len(df)
                print(csv_path)
                print(len(df))
    # with open('./defect_rate.txt','wb') as f:
    #     for i,j in projects_dict.items():
    #         f.write((i+"\n").encode())
    #         f.write((str(j)+'\n\n').encode())






def is_csv(f):
    return f.endswith(".csv")

def is_code(filename):
    return filename.endswith('.java')


def process_for_each_project(project,version):
    count1 = 0
    project_path = './dataset_source/{}_{}'.format(project,version)
    for path,file_dirs,files in os.walk(project_path):
        source_files =[f for f in files if is_code(f)]
        for file in source_files:
            count1 += 1
    return count1

if __name__ == '__main__':
    projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
                                "ivy":['1.4','2.0'],"xalan":['2.4','2.5']}
    for project in projects_with_version.keys():
        for version in projects_with_version[project]:
            count = process_for_each_project(project,version)
            print("{}_{}: {}".format(project,version,count))
    get_defect_rate()   
