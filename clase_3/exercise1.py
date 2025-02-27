import re
elementos=input("Dame 5 numeros separados con espacios: ")
lista=re.split(r"\s",elementos)
lista=[int(x) for x in lista]
lista.sort()
print(lista)

#-----------------Acortdado
print(*sorted([int(x) for x in re.split(r"\s",input("Dame 5 numeros separados por espacios: "))]))
#---------------------------Acortado 2
print(*sorted(map(int,re.split(r"\s",input("Dame 5 numeros separados por espacios: ")))))
