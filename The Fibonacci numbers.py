x= int(input("Enter the number : "))
n1= 0
n2 = 1
count = 0
while count < x:
    print(n1)
    nth = n1+n2
    n1= n2
    n2= nth
    count += 1
