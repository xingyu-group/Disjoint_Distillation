import os


join = os.path.join
def count_num_files(dir_path):
    files = os.listdir(dir_path)
    return len(files)
    
    
    
    
    
