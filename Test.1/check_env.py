import sys

# 让系统标准输出同时写入屏幕与文件
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

sys.stdout = Logger("check_env_result.txt")

# 原代码
print('Python 版本：', sys.version)

mods = ['numpy', 'pandas', 'matplotlib', 'sklearn', 'jieba', 'nltk']
for m in mods:
    try:
        mod = __import__(m)
        print(f'{m}: {getattr(mod, "__version__", "OK")}')
    except Exception as e:
        print(f'{m}: 失败 -> {e}')

import jieba
text = '自然语言处理是人工智能皇冠上的明珠'
print('jieba分词：', jieba.lcut(text))