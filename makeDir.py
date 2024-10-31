import os

# 定义项目的目录结构以及需要创建的 Python 文件
project_structure = {
    "data/raw": [],
    "data/processed": [],
    "models": ["model.py", "utils.py"],
    "scripts": ["train.py", "test.py", "evaluate.py"],
    "notebooks": [],
    "logs": [],
    "configs": [],
}

# 创建目录和文件
def create_project_structure(base_dir):
    for folder, files in project_structure.items():
        # 创建文件夹
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        # 创建文件
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, "w") as f:
                f.write("")  # 创建空文件

# 创建基础文件
def create_base_files(base_dir):
    base_files = [
        "requirements.txt",
        "setup.py",
        "README.md"
    ]
    for file in base_files:
        file_path = os.path.join(base_dir, file)
        with open(file_path, "w") as f:
            f.write("")  # 创建空文件

# 运行创建函数
base_directory = "."
create_project_structure(base_directory)
create_base_files(base_directory)

print("Directory structure and Python files created successfully.")
