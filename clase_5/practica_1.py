def quick_sort(arr,bajo,alto):
    if bajo<alto:
        pi=particion(arr,bajo,alto)

        quick_sort(arr,bajo,pi-1)
        quick_sort(arr,pi+1,alto)

def particion(arr,bajo,alto):
    pivot=arr[alto]
    i=bajo-1

    for j in range (bajo,alto):
        if arr[j]<=pivot:
            i+=1
            arr[i],arr[j]=arr[j],arr[i]
    arr[i+1],arr[alto]=arr[alto],arr[i+1]
    return i+1

array=[1,4,2,7,3,5,8,0,4]
quick_sort(array,0,(len(array)-1))
print (array)