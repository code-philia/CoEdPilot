import json
import pandas as pd

class CodeGraph:
    def __init__(self, report_path):
        with open(report_path, 'r', encoding='latin-1') as f:
            self.report = json.load(f)
        self.search_node_dict = {} # {file_path: {line_idx: [variable_id]}}
        self.search_dep_dict = {}

        # self.def_pos = {} # position of definition
        # for dep in self.report['cells']:
        #     if "Define" in dep['values']:
        #         self.def_pos[dep['dest']] = dep['values']['startPos']

        self.nodes = {node["id"]:node for node in self.report['variables'] if 'File' in node} # only extract nodes lower than file level
        # self.graph = pd.DataFrame(self.report['cells'])
        for variable in self.report['variables']:
            try:
                if variable['File'] not in self.search_node_dict:
                    self.search_node_dict[variable['File']] = {}
                f = self.search_node_dict[variable['File']]

                # this step is inefficient, consider using ordered set or other range-search-efficient data structure
                for l in range (variable['location']['startLine'], variable['location']['endLine']+1):
                    if l not in f:
                        f[l] = {}
                    f[l][variable['id']] = variable

                # else:
                #     f[l].append(variable['id'])
                # for line_idx in range(variable['location']['startLine'], variable['location']['endLine']+1):
                #     if line_idx not in self.search_node_dict[variable['File']].keys():
                #         self.search_node_dict[variable['File']][line_idx] = [variable['id']]
                #     else:
                #         self.search_node_dict[variable['File']][line_idx].append(variable['id'])
                #     if line_idx not in self.search_node_dict[variable['File']].keys():
                #         self.search_node_dict[variable['File']][line_idx] = [variable['id']]
                #     else:
                #         self.search_node_dict[variable['File']][line_idx].append(variable['id'])
            except:
                continue
        for dep in self.report['cells']:
            if dep["src"] not in self.nodes or dep["dest"] not in self.nodes:    # ignore deps that are file-level
                continue
            in_file = self.nodes[dep["src"]]['File']
            loc = dep["values"]["loc"]
            if in_file not in self.search_dep_dict:
                self.search_dep_dict[in_file] = {}
            # for line_idx in range(loc['startLine'], loc['endLine']):
            #     if line_idx not in self.search_dep_dict[in_file]:
            #         self.search_dep_dict[in_file][line_idx] = {}
            #     self.search_dep_dict[in_file][line_idx][dep['dest']] = dep
            line_idx = loc['startLine']
            if line_idx not in self.search_dep_dict[in_file]:
                self.search_dep_dict[in_file][line_idx] = {}
            self.search_dep_dict[in_file][line_idx][dep['dest']] = dep

        self.src2dest = {} # {src_id: [dest_id]}
        self.dest2src = {} # {dest_id: [src_id]}
        for edge in self.report['cells']:
            if edge['src'] not in self.src2dest.keys():
                self.src2dest[edge['src']] = set([edge['dest']])
            else:
                self.src2dest[edge['src']].add(edge['dest'])
            if edge['dest'] not in self.dest2src.keys():
                self.dest2src[edge['dest']] = set([edge['src']])
            else:
                self.dest2src[edge['dest']].add(edge['src'])

    def has_file(self, file_path):
        return file_path in self.search_node_dict
    
    def files(self):
        return self.search_node_dict.keys()

    def __search_loc_attr(self, file_path, locations, from_dict):
        if file_path not in from_dict:
            return {}
        
        nodes_at_file = from_dict[file_path]
        ids = {i:j for l in locations if l in nodes_at_file for i,j in nodes_at_file[l].items()}
        return ids

    def location2dep(self, file_path, locations): # 根据文件路径和行号寻找变量 id
        return self.__search_loc_attr(file_path, locations, self.search_dep_dict)

    def location2variable(self, file_path, locations): # 根据文件路径和行号寻找变量 id
        return self.__search_loc_attr(file_path, locations, self.search_node_dict)
    
    def find_dependants(self, id): # 寻找依赖于输入 id 的变量 id
        return self.src2dest.get(id, set())

    def find_dependencies(self, id): # 寻找输入 id 依赖的变量 id
        return self.dest2src.get(id, set())

    def find_dependants_and_dependencies(self, id): # 寻找输入 id 的所有依赖变量 id
        return self.find_dependants(id).union(self.find_dependencies(id))
    
    def get_variable(self, id): # 根据 id 获取变量信息
        return self.nodes.get(id, None)
    
    def __snippet_related(self, file_path1, locations1, file_path2, locations2, dep_finder):
        ids1 = self.location2variable(file_path1, locations1)
        ids2 = self.location2variable(file_path2, locations2)
        
        all_dependency_of_ids1 = []
        for id1 in ids1:
            all_dependency_of_ids1 += dep_finder(id1)
        all_dependency_of_ids1 = list(set(all_dependency_of_ids1))

        for id2 in ids2:
            if id2 in all_dependency_of_ids1:
                return True
        return False

    def location_exist_edge(self, file_path1, locations1, file_path2, locations2):
        return self.__snippet_related(file_path1, locations1, file_path2, locations2, self.find_dependants_and_dependencies)

    def exist_depend_on(self, file_path1, locations1, file_path2, locations2): # 判断 file_path1 中 locations1 行是否依赖于 file_path2 中 locations2 行
        return self.__snippet_related(file_path1, locations1, file_path2, locations2, self.find_dependencies)

    def exist_depend_by(self, file_path1, locations1, file_path2, locations2): # 判断 file_path1 中 locations1 行是否被 file_path2 中 locations2 行依赖
        return self.__snippet_related(file_path1, locations1, file_path2, locations2, self.find_dependants)
    
    def include_dep_to(self, dep_variables, dep_file, dep_lines):  
        '''check if a set of variables is depending on another code snippet's variable definitions, and return the variable id if so'''
        variables_in_dep_window = self.location2dep(dep_file, dep_lines)

        for i in dep_variables:
            if i in variables_in_dep_window:
                return variables_in_dep_window[i]
        # return not dep_variables.isdisjoint(variables_in_dep_window)
