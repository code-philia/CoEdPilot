import json
import pandas as pd

class CodeGraph:
    def __init__(self, report_path):
        with open(report_path, 'r') as f:
            self.report = json.load(f)
        self.search_variable_dict = {} # {file_path: {line_idx: [variable_id]}}
        self.graph = pd.DataFrame(self.report['cells'])
        for variable in self.report['variables']:
            try:
                if variable['File'] not in self.search_variable_dict.keys():
                    self.search_variable_dict[variable['File']] = {}
                for line_idx in range(variable['location']['startLine'], variable['location']['endLine']+1):
                    if line_idx not in self.search_variable_dict[variable['File']].keys():
                        self.search_variable_dict[variable['File']][line_idx] = [variable['id']]
                    else:
                        self.search_variable_dict[variable['File']][line_idx].append(variable['id'])
            except:
                continue
        self.src2dest = {} # {src_id: [dest_id]}
        self.dest2src = {} # {dest_id: [src_id]}
        for edge in self.report['cells']:
            if edge['src'] not in self.src2dest.keys():
                self.src2dest[edge['src']] = [edge['dest']]
            else:
                self.src2dest[edge['src']].append(edge['dest'])
            if edge['dest'] not in self.dest2src.keys():
                self.dest2src[edge['dest']] = [edge['src']]
            else:
                self.dest2src[edge['dest']].append(edge['src'])

    def location2variable(self, file_path, locations): # 根据文件路径和行号寻找变量 id
        ids = []
        for location in locations:
            try:
                ids += self.search_variable_dict[file_path][location]
            except:
                continue
        return list(set(ids))
    
    def find_dependant(self, id): # 寻找依赖于输入 id 的变量 id
        try:
            dest_ids = self.src2dest[id]
        except:
            return []
        return list(set(dest_ids))

    def find_depend(self, id): # 寻找输入 id 依赖的变量 id
        try:
            src_ids = self.dest2src[id]
        except:
            return []
        return list(set(src_ids))

    def find_dependency(self, id): # 寻找输入 id 的所有依赖变量 id
        return list(set(self.find_dependant(id) + self.find_depend(id)))
    
    def get_variable(self, id): # 根据 id 获取变量信息
        for variable in self.report['variables']:
            if variable['id'] == id:
                return variable
    
    def location_exist_edge(self, file_path1, locations1, file_path2, locations2):
        ids1 = self.location2variable(file_path1, locations1)
        ids2 = self.location2variable(file_path2, locations2)
        
        all_dependency_of_ids1 = []
        for id1 in ids1:
            all_dependency_of_ids1 += self.find_dependency(id1)
        all_dependency_of_ids1 = list(set(all_dependency_of_ids1))

        for id2 in ids2:
            if id2 in all_dependency_of_ids1:
                return True
        return False

    def exist_depend_on(self, file_path1, locations1, file_path2, locations2): # 判断 file_path1 中 locations1 行是否依赖于 file_path2 中 locations2 行
        ids1 = self.location2variable(file_path1, locations1)
        ids2 = self.location2variable(file_path2, locations2)

        ids1_depends_on = []
        for id1 in ids1:
            ids1_depends_on += self.find_depend(id1)
        ids1_depends_on = list(set(ids1_depends_on))

        for id2 in ids2:
            if id2 in ids1_depends_on:
                return True
        return False

    def exist_depend_by(self, file_path1, locations1, file_path2, locations2): # 判断 file_path1 中 locations1 行是否被 file_path2 中 locations2 行依赖
        ids1 = self.location2variable(file_path1, locations1)
        ids2 = self.location2variable(file_path2, locations2)

        ids1_depends_by = []
        for id1 in ids1:
            ids1_depends_by += self.find_dependant(id1)
        ids1_depends_by = list(set(ids1_depends_by))

        for id2 in ids2:
            if id2 in ids1_depends_by:
                return True
        return False