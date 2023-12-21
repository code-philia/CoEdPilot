# a demo of how to use enre-py to parse python dependency graph
import json
import subprocess
from tqdm import tqdm

with open('./dataset/python_hunk_dataset.json', 'r', encoding="utf8") as f:
    hunkset = json.load(f)
with open('./dataset/python_parent_sha_dict.json', 'r', encoding="utf8") as f:
    parent_sha_dict = json.load(f)

commit_shas=[]
for commit_url in hunkset.keys():
    proj_name = commit_url.split('/')[-3]
    if proj_name == 'ColossalAI':
        commit_shas.append(commit_url.split('/')[-1])

print('generating report for each commit')
for commit_sha in tqdm(commit_shas[200:]):
    git_repo_path = "C:\\Users\\NUS\\Downloads\\workspace\\dependency\\dataset\\repo\\ColossalAI"
    parent_sha = parent_sha_dict[commit_sha]
    print('parent_sha:', parent_sha)
    try:
        checkout_command = f'git -C {git_repo_path} checkout {parent_sha}'
        result = subprocess.run(checkout_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        build_dep_graph_cmd = '.\\enre-py.exe C:\\Users\\NUS\\Downloads\\workspace\\dependency\\dataset\\repo\\ColossalAI'
        result = subprocess.run(build_dep_graph_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        mv_command = f'move ColossalAI-report-enre.json C:\\Users\\NUS\\Downloads\\workspace\\dependency\\dataset\\reports\\report_{commit_sha}.json'
        result = subprocess.run(mv_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except:
        print(f'error in {parent_sha}')
        continue
