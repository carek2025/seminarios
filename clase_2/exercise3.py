def es_primo(n):
    if n<=1:
        return False
    for i in range(2,n//2):
        if n%i==0:
            return False
    return True

def lista_de_primos(inicio,final=1000):
    primos=list()
    for i in range(inicio,final):
        if es_primo(i):
            primos.append(i)
            if len(primos)==20:
                break
    print(primos)

lista_de_primos(1)


#algoritmos turing y algoritmo burbuja y redes neurales y arboles binarios y mergesort

