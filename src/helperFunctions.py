import os

def createDir(path):
    '''
    Tries to create a directory (relative to root)    
    '''
    try:
        os.mkdir(path)
    except FileExistsError:
        print(path, 'directory already exists')