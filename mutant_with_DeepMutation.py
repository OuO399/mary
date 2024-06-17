import os

projects_with_version = {'ant':['1.5','1.6'],"jEdit":['4.0','4.1'],"synapse":['1.0','1.1'],"camel":['1.4'],
                            "ivy":['1.4'],"xalan":['2.4']}

for project in projects_with_version.keys():
    for version in projects_with_version[project]:
        