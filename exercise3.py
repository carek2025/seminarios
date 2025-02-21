def es_primo(n):
    if n<=1:
        return False
    for i in range(2,n//2):
        if n%i==0:
            return False
    return True

def lista_de_primos(inicio,final):
    primos=list()
    for i in range(inicio,final):
        if es_primo(i):
            primos.append(i)
    print(primos)

lista_de_primos(1,20)


#algoritmos turing y algoritmo burbuja y redes neurales y arboles binarios y mergesort























def es_primo(num):
    if num <=1:
        return False
    for i in range(2,num//2):
        if
