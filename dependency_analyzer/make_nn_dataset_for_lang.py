import sys
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
    res = re.search(r"/([^/]*?)(.git)?$", repo_url)
    if res is None:
        raise RuntimeError(f"Cannot get repo name from URL: {repo_url}")
    return res[1]

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
        return True
    
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

def get_dep_type(dep):
    return next((i for i in dep['values'].keys() if re.match(r'^[A-Z]', i)), "Unknown")

def format_dep(dep):
    return {
        'src': dep['src'],
        'dest': dep['dest'],
        'type': get_dep_type(dep)
    }

def filter_defined_variables(var_dict, snippet, use_modifier=True):
    if use_modifier:
        return {
            v:v_info for v, v_info in var_dict.items()
            if next(re.finditer(re.escape(v_info.get('modifiers', '')) + '.*?' + re.escape(v_info['name']), snippet, re.DOTALL), None) is not None
        }
    else:
        return {
            v:v_info for v, v_info in var_dict.items()
            if next(re.finditer(re.escape(v_info['name']), snippet, re.DOTALL), None) is not None
        }

# 主方法

def make_dataset(repo_url, lang):
    repo_dir = clone_repo(repo_url, lang)
    
    repo_name = get_repo_name_from_url(repo_url)
    if not generate_dep(repo_name, lang):     # if the out file is already analyzed or invalid
        return
        
    dataset = []
    pos_n = 0
    neg_n = 0

    codegraph = CodeGraph(get_enre_out_file_path(repo_name, lang))
    # try:
    # except Exception as e:
    #     print(e.with_traceback(sys.exc_info()[2]))
    #     print(f"Cannot build code graph for module {repo_name}. Skipping...")
    #     return
    # print("Finish building code graph")

    files = get_files(repo_dir) # 获取项目下所有文件路径
    _files = []
    for file in files:
        file_path = convert_file_to_graph_file_path(file, repo_name, lang)
        if codegraph.has_file(file_path):
            _files.append((file, file_path))
    files = _files

    for file, file_path in tqdm(files):    # File is relative path now
        try:
            with open(os.path.join(repo_dir, file), 'r', encoding="utf-8") as f:
                lines = f.readlines()
        except:
            # print(f"Error reading {file}. Skipping...")
            continue
            
        i = 1
        while i <= len(lines): # 每 2~15 行 为一个 code window
            _i = i + random.randint(2, 15)  # upper bound of code window
            code_snippet = ''.join(lines[i-1:_i-1])
            line_idx = [j for j in range(i,_i)]
            i = _i

            # 找到这些行中定义的变量
            def_variables = codegraph.location2variable(file_path, line_idx)
            def_variables = filter_defined_variables(def_variables, code_snippet)
            if len(def_variables) <= 0:
                continue

            # 找到这些行依赖的变量
            dep_variables = codegraph.location2dep(file_path, line_idx)
            if len(dep_variables) <= 0:
                continue

            window_pos_n = 0
            var_ids = sorted(dep_variables)
            for dep_var_id in random.sample(var_ids, max(len(var_ids)//3, 1)): # 随机选取一半的依赖变量))):
                dep = dep_variables[dep_var_id]
                # 找到对应的代码行，提取出一个 code window
                dep_var_info = codegraph.get_variable(dep_var_id)
                if dep_var_info is None or 'location' not in dep_var_info \
                    or len(filter_defined_variables({dep_var_id: dep_var_info}, code_snippet, use_modifier=False)) == 0:
                    continue

                at_line = dep_var_info['location'].get('startLine', 0)
                if at_line <= 0:
                    continue
                
                _dep = format_dep(dep)
                dep_file_path = os.path.join(repo_dir, dep_var_info["File"])
                
                try:
                    with open(dep_file_path, 'r') as f:
                        dep_lines = f.readlines()
                except:
                    continue
                look_back = random.randint(0, 5)
                look_forward = random.randint(1, 6)
                line_range = [i for i in range(max([at_line-look_back,1]), at_line+look_forward)] # 得到代码行范围

                dep_code_snippet = ''.join(dep_lines[line_range[0]-1:line_range[-1]])
                
                # check if depended-on variable's code snippet depends on original defined variables
                back_dep = codegraph.include_dep_to(def_variables, dep_var_info['File'], line_range)
                if back_dep:
                    dataset.append([[code_snippet, dep_code_snippet], [1,1], [_dep, format_dep(back_dep)]])
                else:
                    if random.random() < 0.5:
                        dataset.append([[code_snippet, dep_code_snippet], [1, 0], [_dep, None]])
                    else:
                        dataset.append([[dep_code_snippet, code_snippet], [0, 1], [None, _dep]])
                pos_n += 1
                window_pos_n += 1

            # construct negative samples
            neg_cnt = 0
            while neg_cnt < max([1, window_pos_n * 2]):
                _file, _file_path = random.choice(files)
                try:
                    with open(os.path.join(repo_dir, _file), 'r') as f:
                        _lines = f.readlines()
                    _l = len(_lines)
                    _from = random.randint(0, _l-1)
                    _to = random.randint(2, 15)
                    _code_snippet = ''.join(_lines[_from:_to])
                    if _code_snippet.strip() == '':
                        continue
                    _line_idx = [i for i in range(_from, _to)]
                    _v = codegraph.location2variable(_file_path, _lines)
                    _v = filter_defined_variables(_v, code_snippet)
                    if not (codegraph.include_dep_to(_v, file_path, line_idx) or codegraph.include_dep_to(def_variables, _file_path, _line_idx)):
                        dataset.append([[code_snippet, _code_snippet], [0,0], ['None', 'None']])
                        neg_cnt += 1
                        neg_n += 1
                except:
                    continue
            # print("negative +1")

    # # 从当前 dataset 中提取出所有 snippet
    # original_length = len(dataset)
    # snippets = [data[0][0] for data in dataset] + [data[0][1] for data in dataset]
    # for pair in range(int(original_length*0.66)):
    #     snippet1 = random.sample(snippets, 1)[0]
    #     snippet2 = random.sample(snippets, 1)[0]
    #     dataset.append([[snippet1, snippet2], [0,0]])

    print(f"pos_n: {pos_n}, neg_n: {neg_n}")
    print(f"Totally {len(dataset)} samples")
    # 打乱 dataset
    random.shuffle(dataset)
    print('dataset length:', len(dataset))
    # 保存 dataset
    # train:valid:test = 7:1:2
    dataset_dir = f"./dataset/{lang}/{repo_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, 'all.json'), 'w') as f:
        json.dump(dataset, f)
    # with open(os.path.join(dataset_dir, 'train.json'), 'w') as f:
    #     json.dump(dataset[:int(len(dataset)*0.7)], f)
    # with open(os.path.join(dataset_dir, 'valid.json'), 'w') as f:
    #     json.dump(dataset[int(len(dataset)*0.7):int(len(dataset)*0.8)], f)
    # with open(os.path.join(dataset_dir, 'test.json'), 'w') as f:
    #     json.dump(dataset[int(len(dataset)*0.8):], f)

if __name__ == '__main__':
    print("Not doing anything")