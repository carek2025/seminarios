import math as m
def operacion_suma_y_multiplicacion():    
    isnum=True
    while isnum:
        try:
            num1=m.fabs(float(input("dame el primer numero: ")))
            num2=m.fabs(float(input("dame el segundo numero: ")))
            print(f"La suma es {round(num1+num2)}")
            print(f"La multiplicacion es {round(num1*num2)}")
            isnum=False
        except ValueError:
            print("Ninguno de los numeros puede ser un string o texto por favor ingrese de nuevo los numeros")
