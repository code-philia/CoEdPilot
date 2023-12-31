from genericpath import isfile
import subprocess
import os
import json
import random
from tqdm import tqdm
import re
from codegraph import CodeGraph

CUR_DIR = os.path.dirname(__file__)

# ENRE 可执行文件的路径
ENRE_EXECUTABLES = {
    "java": os.path.join(CUR_DIR, "enre_out", "java", "enre_java_1.2.4.jar"),
    "python": os.path.join(CUR_DIR, "enre_out", "java", "enre_java_1.2.4.jar"),
    "javascript": os.path.join(CUR_DIR, "enre_out", "typescript", "enre-ts-0.0.1-gamma.js"),
    "go": os.path.join(CUR_DIR, "enre_out", "go", "enre_java_1.2.4.jar")
}

# 存放仓库的路径
REPO_FOLDER = os.path.join(CUR_DIR, "repo")

def get_repo_name_from_url(repo_url):
    return re.search(r"/([^/]*?)(.git)?$", repo_url)[1]

def get_repo_dir(repo_name, lang):
    return os.path.join(REPO_FOLDER, lang, repo_name)

def get_enre_out_file_dir(lang):
    return os.path.join(CUR_DIR, "enre_out", f"{lang}")

# 因 ENRE 无法指定路径，不同语言的 ENRE 分析器的固定输出路径

def get_enre_out_file_path(repo_name, lang):
    match lang:
        case "java":
            return os.path.join(get_enre_out_file_dir(lang), f'{repo_name}-enre-out', f'{repo_name}-out.json')
        case "python":
            return os.path.join(get_enre_out_file_dir(lang), f'{repo_name}-report-enre.json')
        case "javascript":
            return os.path.join(get_enre_out_file_dir(lang), f'{repo_name}-report-enre.json')
        case _:
            raise NotImplementedError("get_enre_out_file_path only supporting [java, python]")

# acquire all files in the dataset
def get_files(repo_dir):
    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        rel_to_repo = os.path.relpath(root, repo_dir)
        if (rel_to_repo == '.'):
            rel_to_repo = ''
        for filename in filenames:
            files.append(os.path.join(rel_to_repo, filename))
    return files

def sys_exec_cmd(cmd):
    print(f'Executing command: "{cmd}"')
    exit_code = os.system(cmd)
    # if (exit_code != 0):
    #     raise RuntimeError(f'Problem running "{cmd}"')

# 下载仓库

def clone_repo(repo_url: str, lang):
    lang_dir = os.path.join(REPO_FOLDER, lang)
    os.makedirs(lang_dir, exist_ok=True)
    
    repo_dir = get_repo_dir(get_repo_name_from_url(repo_url), lang)
    if os.path.isdir(repo_dir):
        print(f"Ignoring {repo_url}. Repo already cloned")
        return repo_dir
    
    if not repo_url.endswith(".git"):
        repo_url += ".git"
    clone_command = f'cd "{lang_dir}" && git clone "{repo_url}"'
    sys_exec_cmd(clone_command)
    print(f"Cloned repo: {repo_url} at {repo_dir}")
    return repo_dir

# 运行 ENRE 并生成依赖分析文件

ENRE_COMMAND_PATTERNS = {
    "java": 'java -jar "{exec_file}" java "{repo_dir}" {repo_name}"',
    "python": '{exec_file} "{repo_dir}" --cg',
    # "javascript": 'node {exec_file} -i {repo_dir} -v -o {repo_name}-report-enre.json',
    "javascript": 'node {exec_file} -i {repo_dir} -v',
    "go": ""
}

def general_generate_dep_for_lang(repo_name, lang):
    if lang not in ENRE_COMMAND_PATTERNS:
        raise NotImplementedError(f"generate_dep only supporting {[ENRE_COMMAND_PATTERNS.keys()]}")

    repo_dir = get_repo_dir(repo_name, lang)
    out_file_path = get_enre_out_file_path(repo_name, lang)
    exec_file = ENRE_EXECUTABLES[lang]
    enre_command = f'cd "{get_enre_out_file_dir(lang)}" && ' + ENRE_COMMAND_PATTERNS[lang].format(exec_file=exec_file, repo_dir=repo_dir, repo_name=repo_name)
    sys_exec_cmd(enre_command)
    print(f'ENRE {lang}: generated dep for "{repo_name}" at {out_file_path}')

def generate_dep(repo_name, lang):
    enre_out_path = get_enre_out_file_path(repo_name, lang)
    if os.path.isfile(enre_out_path):
        print(f"ENRE out file {enre_out_path} existed. Skipping...")
        return False
    
    general_generate_dep_for_lang(repo_name, lang)
    
    if not os.path.isfile(enre_out_path):
        print(f"ENRE cannot generated out file {enre_out_path}")
        return False
    return True

# 转化路径格式，ENRE 的不同语言的输出有着不同的路径格式

def convert_file_to_graph_file_path(file, repo_name, lang):
    file = file.replace('\\', '/')
    if lang == "java":
        return file
    if lang == "python":
        return repo_name + '/' + file
    else:
        raise NotImplementedError()

# 主方法

def make_dataset(repo_url, lang):
    repo_dir = clone_repo(repo_url, lang)
    
    repo_name = get_repo_name_from_url(repo_url)
    if not generate_dep(repo_name, lang):     # if the out file is already analyzed or invalid
        return
        
    dataset = []

    try:
        codegraph = CodeGraph(get_enre_out_file_path(repo_name, lang))
    except Exception as e:
        print(e.with_traceback())
        print(f"Cannot build code graph for module {repo_name}. Skipping...")
        return
    print("Finished building code graph")

    files = get_files(repo_dir) # 获取项目下所有文件路径
    for file in tqdm(files):    # File is relative path now
        # 逐个打开
        try:
            with open(os.path.join(repo_dir, file), 'r', encoding="utf-8") as f:
                lines = f.readlines()
        except:
            # print(f"Error reading {file}. Skipping...")
            continue

        file_id = convert_file_to_graph_file_path(file, repo_name, lang)
        if not codegraph.has_file(file_id):
            continue
        
        for i in range(len(lines)//10-1): # 每 10 行 为一个 code window
            line_idx = [i*10+j for j in range(1, 11)]
            # 把 10 行拼起来
            code_snippet = ''.join([lines[idx] for idx in line_idx])
            # 找到这 10 行中的变量
            variables = codegraph.location2variable(file_id, line_idx)
            if len(variables) == 0:
                continue
            # else:
                # print('file name:', file)
                # print('file variable',variables)
                
            # 找到这些变量依赖的变量
            dep_variables = []
            for vairable in variables: # 逐个变量寻找依赖
                dep_variables += codegraph.find_depend(vairable)
            dep_variables = list(set(dep_variables))
            # print('dep variable:',dep_variables)
            # 从依赖变量中随机选取5%的变量
            for dep_variable in random.sample(dep_variables, int(len(dep_variables)*1)):
                # 找到对应的代码行，提取出一个 code window
                variable_info = codegraph.get_variable(dep_variable)
                if 'location' not in variable_info:
                    continue
                at_line = variable_info['location']['startLine']
                if at_line < 1:
                    continue
                
                dep_file_path = os.path.join(repo_dir, file)
                
                try:
                    with open(dep_file_path, 'r') as f:
                        dep_lines = f.readlines()
                    line_range = [i for i in range(at_line-4, at_line+5)] # 得到代码行范围
                    dep_code_snippet = ''.join(dep_lines[line_range[0]-1:line_range[-1]])
                    # 得到这些代码行中的变量 id
                    variables_in_dep_window = codegraph.location2variable(variable_info['File'], line_range)    # Here assumed: those not with 'location' attribute doesn't have 'File' attribute too 
                    # 得到这些变量依赖的变量 id
                    dep_variables_in_dep_window = []
                    for vairable in variables_in_dep_window:
                        dep_variables_in_dep_window += codegraph.find_depend(vairable)
                    # 如果依赖变量的依赖变量没有当前变量，则仅有依赖关系，而不是互相依赖
                    if len(set(dep_variables_in_dep_window).intersection(set(variables_in_dep_window))) == 0:
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
    dataset_dir = f"./dataset/{lang}/{repo_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, 'train.json'), 'w') as f:
        json.dump(dataset[:int(len(dataset)*0.7)], f)
    with open(os.path.join(dataset_dir, 'valid.json'), 'w') as f:
        json.dump(dataset[int(len(dataset)*0.7):int(len(dataset)*0.8)], f)
    with open(os.path.join(dataset_dir, 'test.json'), 'w') as f:
        json.dump(dataset[int(len(dataset)*0.8):], f)

if __name__ == '__main__':
    print("Not doing anything")