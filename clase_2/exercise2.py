import numpy as np
import re
import pandas as pd
arr1=np.array([int(x) for x in re.split(r"\s",input("Dame 5 numeros separados por espacio: "))])
arr2=np.array([int(x) for x in re.split(r"\s",input("Dame 5 numeros separados por espacio: "))])
print(arr1*arr2)
#-----------------Sin numpy-----------------------------------
arr1=[int(x) for x in input("Dame 5 numeros separados por espacio: ").split()]
arr2=[int(x) for x in input("Dame 5 numeros separados por espacio: ").split()]
resultado2=list()
for x,y in zip(arr1,arr2):
    resultado2.append(x*y)
print(resultado2)
#------------------sin numpy ------------------------------------
print([x*y for x,y in zip([int(x) for x in input("Dame 5 numeros separados por espacios:").split()],[int(x) for x in input("Dame 5 numeros separados por espacio: ").split()])])