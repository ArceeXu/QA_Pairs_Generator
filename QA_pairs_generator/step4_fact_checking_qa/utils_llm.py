import json
import os
import pathlib

def mkdir(fp):
    pathlib.Path(fp).mkdir(exist_ok=True)

def read_json(src):
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(obj, dst, indent=None):
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)

def write_jsonl(obj, dst, mode="w"):
    """
    将对象列表写入 JSONL 文件
    mode: 'w' 覆盖写入, 'a' 追加写入
    """
    with open(dst, mode, encoding="utf-8") as jsonl_file:
        jsonl_file.writelines(
            [json.dumps(o, ensure_ascii=False) + '\n' for o in obj]
        )

def read_jsonl(src):
    with open(src, "r", encoding="utf-8") as f:
        data = [json.loads(o.strip()) for o in f.readlines() if o.strip()]
    return data