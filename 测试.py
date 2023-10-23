import os

def count_lines_of_code(directory):
    total_lines = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            # if file.endswith("segment_anything"):
            #     continue
            if file.endswith(".py"):  # 可以根据需要修改文件扩展名
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                    total_lines += len(lines)
    return total_lines

if __name__ == "__main__":
    code_directory = "/homes/syli/python/LVSI"  # 将 "your_code_directory" 替换为你的代码目录
    total_lines = count_lines_of_code(code_directory)
    print(f"Total lines of code: {total_lines}")
