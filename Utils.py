import os.path

def build_path(path):
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)
    



