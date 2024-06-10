import os

def clear_console():
    if os.name == 'nt':  
        os.system('cls')
    else:  
        os.system('clear')