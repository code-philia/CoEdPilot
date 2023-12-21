# 工序

## 建立目录树

见 `make_dataset_from_repo_url.ipynb` 第一块。

## 利用 ENRE 依赖分析器制作数据集的步骤（详见 `make_dataset_from_repo_url.ipynb`）

+ 下载仓库至 `dataset/repo/`
+ 在 `enre_out/` 中运行对应语言的 ENRE 程序，得出依赖分析结果
+ 对每个项目，利用依赖分析结果制作 CodeGraph
+ 从 CodeGraph 中随机地提取数据，制成该项目的数据集，存放于 `dataset/`

## 训练模型

略

## 目录结构

```
\dataset        # 不同语言的数据集及其仓库
    \java       # Java 数据集，每个文件夹是一个仓库生成的 3 份数据集
        ...
    \python     # Python 数据集，每个文件夹是一个仓库生成的 3 份数据集
        ...
    \repo       # clone 下来的不同语言的 GitHub 仓库
        \java
            ...
        \python
            ...
\enre_out       # 不同语言的 ENRE 分析器与分析结果，因无法指定输出文件名，文件名各有特定的格式
    \java       # ENRE Java 分析器与结果
        enre_java_1.2.4.jar
        ...
    \python     # ENRE Python 分析器与结果
        enre-zdh-stable.exe
        ...
\repo_url       # 不同语言的仓库 URL 列表
    java_repo_urls.json
    python_repo_urls.json
```
