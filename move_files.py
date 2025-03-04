import os
import shutil

def split_files(source_dir, num_parts=10):
    # 确保源目录存在
    if not os.path.exists(source_dir):
        print(f"错误：源目录 {source_dir} 不存在！")
        return

    # 获取源目录中的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    total_files = len(files)
    print(f"源目录中共有 {total_files} 个文件。")

    # 如果文件数量少于10个，提示用户
    if total_files < num_parts:
        print(f"文件数量少于 {num_parts} 个，无法分割！")
        return

    # 计算每份文件的数量
    files_per_part = total_files // num_parts
    remaining_files = total_files % num_parts

    # 创建目标文件夹并分配文件
    for i in range(num_parts):
        # 创建目标文件夹
        target_dir = os.path.join('/home/khj/agis', str(i))
        os.makedirs(target_dir, exist_ok=True)

        # 计算当前部分的文件范围
        start_index = i * files_per_part + min(i, remaining_files)
        end_index = (i + 1) * files_per_part + min(i + 1, remaining_files)

        # 将文件复制到目标文件夹
        for file in files[start_index:end_index]:
            shutil.copy(os.path.join(source_dir, file), target_dir)

        print(f"已将文件复制到 {target_dir} 文件夹。")

    print("文件分割完成！")

# 指定源目录
source_directory = "/home/khj/Feb21_2025_7971gene_info_omics"
split_files(source_directory)
