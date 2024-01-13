import os
import json
import random
from tqdm import tqdm
from itertools import combinations
from codegraph import CodeGraph

REPO_PATH = '.\\dataset\\repo\\'

# acquire all files in the dataset
def get_files(repo_name):
    global REPO_PATH
    files = []
    for root, dirs, filenames in os.walk(os.path.join(REPO_PATH, repo_name)):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return [file[len(REPO_PATH):] for file in files]

def main(report_path, lang):
    global REPO_PATH
    dataset = []
    codegraph = CodeGraph(report_path)
    files = get_files() # 获取项目下所有文件路径
    for file in tqdm(files):
        # 逐个打开
        try:
            with open(os.path.join(REPO_PATH, file), 'r', encoding="utf8") as f:
                lines = f.readlines()
        except:
            continue
        for i in range(len(lines)//10-1): # 每 10 行 为一个 code window
            line_idx = [i*10+j for j in range(1, 11)]
            # 把 10 行拼起来
            code_snippet = ''.join([lines[idx] for idx in line_idx])
            # 找到这 10 行中的变量
            variables = codegraph.location2variable(file.replace('\\', '/'), line_idx)
            if len(variables) == 0:
                continue
            # else:
                # print('file name:', file)
                # print('file variable',variables)
                
            # 找到这些变量依赖的变量
            dep_variables = []
            for vairable in variables: # 逐个变量寻找依赖
                dep_variables += codegraph.find_dependencies(vairable)
            dep_variables = list(set(dep_variables))
            # print('dep variable:',dep_variables)
            # 从依赖变量中随机选取5%的变量
            for dep_variable in random.sample(dep_variables, int(len(dep_variables)*1)):
                # 找到对应的代码行，提取出一个 code window
                variable_info = codegraph.get_variable(dep_variable)
                at_line = variable_info['location']['startLine']
                if at_line < 1:
                    continue
                
                dep_file_path = os.path.join(REPO_PATH,variable_info['File'])
                
                try:
                    with open(dep_file_path, 'r') as f:
                        dep_lines = f.readlines()
                    line_range = [i for i in range(at_line-4, at_line+5)] # 得到代码行范围
                    dep_code_snippet = ''.join(dep_lines[line_range[0]-1:line_range[-1]])
                    # 得到这些代码行中的变量 id
                    variables_in_dep_window = codegraph.location2variable(variable_info['File'], line_range)
                    # 得到这些变量依赖的变量 id
                    dep_variables_in_dep_window = []
                    for vairable in variables_in_dep_window:
                        dep_variables_in_dep_window += codegraph.find_dependencies(vairable)
                    # 如果依赖变量的依赖变量没有当前变量，则仅有依赖关系，而不是互相依赖
                    if set(dep_variables_in_dep_window).intersection(set(variables_in_dep_window)) == set():
                        if random.random() < 0.5:
                            dataset.append([[code_snippet, dep_code_snippet], [1, 0]])
                        else:
                            dataset.append([[dep_code_snippet, code_snippet], [0, 1]])
                    else:
                        dataset.append([[code_snippet, dep_code_snippet], [1,1]])
                except:
                    continue

            
    # 从当前 dataset 中提取出所有 snippet
    original_length = len(dataset)
    snippets = [data[0][0] for data in dataset] + [data[0][1] for data in dataset]
    for pair in range(int(original_length*0.66)):
        snippet1 = random.sample(snippets, 1)[0]
        snippet2 = random.sample(snippets, 1)[0]
        dataset.append([[snippet1, snippet2], [0,0]])

    # 打乱 dataset
    random.shuffle(dataset)
    print('dataset length:', len(dataset))
    # 保存 dataset
    # train:valid:test = 7:1:2
    if os.path.exists(f'./dataset/{lang}') == False:
        os.mkdir(f'./dataset/{lang}')
    with open(f'./dataset/{lang}/train.json', 'w') as f:
        json.dump(dataset[:int(len(dataset)*0.7)], f)
    with open(f'./dataset/{lang}/valid.json', 'w') as f:
        json.dump(dataset[int(len(dataset)*0.7):int(len(dataset)*0.8)], f)
    with open(f'./dataset/{lang}/test.json', 'w') as f:
        json.dump(dataset[int(len(dataset)*0.8):], f)

if __name__ == '__main__':
    report_path = ''
    lang = 'python'
    main(report_path, lang)