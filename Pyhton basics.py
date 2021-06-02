#!/usr/bin/env python
# coding: utf-8

# In[10]:


def even(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    elif n%2==0:
        print("even")
    else:
        print("noo")
even(6)


# In[11]:


def fact(num):
    if num==1:
        return num
    else:
        return num*fact(num-1) 
num= int(input("Enter a num: "))
if num<0:
    print("enter positive num: ")
elif num==0:
    print("factorial is 1")
else:
    print(fact(num))

        


# In[13]:


a='RADAR'

b=reversed(str1)
if list(a)==list(b):
    print('palindrome')
else:
    print("noo")


# In[45]:


a=[]
n=int(input("Enter a number: "))
for i in range(1,n+1):
    b=int(input("Enter a num: "))
    a.append(b)
a.sort()
print(a[n-1])


# In[43]:


a.sort()


# In[44]:


a


# In[49]:


a=3
b=4
a=a+b
b=a-b
a=a-b


# In[50]:


a


# In[51]:


b


# In[ ]:





# In[14]:


def gen_seq(length):
    if(length <= 1):
        return length
    else:
        return (gen_seq(length-1) + gen_seq(length-2))

length = int(input("Enter number of terms:"))
if length<=0:
    print("enter positive one: ")
else:
    print("Fibonacci sequence using Recursion :")
for iter in range(length):
    print(gen_seq(iter))


# In[1]:


def fib(n):
    if n<=1:
        return n
    else:
        return(fib(n-1)+fib(n-2))
nterm=int(input("Enter number of nterms: "))
if nterm<0:
    print("Enter posituve one: ")
else:
    print("fibonacci series: ")
    for i in range(nterm):
        print(fib(i))


# In[3]:


a="jamin"
a[4]


# In[16]:


n=int(input("Enter a number: "))
temp=n
rev=0
while n>0:
    dig=n%10
    rev=rev+dig*dig*dig
    n=n//10
if rev==temp:
    print("yaa palindrome")
else:
    print("oohh noo")


# In[19]:


a=2
b=5

a%b


# In[22]:


15%10


# In[21]:


153//10


# In[23]:


125+27


# In[24]:


15//10


# In[25]:


1%10


# In[26]:


1//10


# In[28]:


n=int(input("Enter a number: "))
temp=n
sum=0
while n>0:
    reg=n%10
    sum=sum*10+reg
    n=n//10
if(temp==sum):
    print("palindrome")
else:
    print("ohh noo")


# In[1]:


i=1
sum=0
n=int(input(('Enter a number: ')))
while i<n:
    if n%i==0:
        sum=sum+i
    i+=1
if sum ==n:
    print("perfect")
else:
    print("ohh noo")


# In[2]:


x=lambda a,b:a+b
x(10,20)


# In[6]:


x=lambda a,b:(a<b and 'first' or 'second')
print(x(10,29))


# In[8]:


def double(n):
    return lambda a:a*n
cal = double(2)
print(cal(4))


# In[10]:


def square(n):
    return n*n
result=map(square,[1,2,3,4])
print('square')
print(list(result))





# In[12]:


n=[1,2,9,4,6,7,2,5]
result=filter(lambda x:x<5,n)
print(list(result))


# In[16]:


from functools import reduce
n=[1,2,3,5]
prod=reduce(lambda x,y:x*y,n)
print(prod)


# In[31]:


def nums(num):
    return num**2
nums=(1,2,3,5,6,7,8,9)


# num

# In[32]:


print("\U0001F600")


# In[33]:


nums


# In[5]:


print("\U0001F618")


# In[6]:


print("\U0001F917")


# In[7]:


print("\U0001F62A")


# In[31]:


print('(❁´◡`❁)(❁´◡`❁)(❁´◡`❁)')


# In[10]:


print("\U0001F637")


# # Send emails using python
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # import winsound to get beep sound

# In[3]:


import winsound


# In[66]:


frequency =3000
duration=6000
winsound.Beep(frequency,duration)


# In[ ]:





# In[ ]:





# In[15]:


def robber(a):
    c=0
    x=""
    for i in a:
        #hi
        c+=1
        for i in range(0,c):
            if a[i] == 'a' or a[i]=='e' or a[i]=='i' or a[i]=='o' or a[i]=='u':
                b=a[i]
                x+=b
            elif a[i] ==' ':
                b=a[i] + 'o' + a[i]
                x+=b
            elif a[i] ==' ':
                x+=a[i]
        print(x)
string = "hii"
robber(string)


# In[16]:


a='hello'
for i in a:
    print(i)


# In[29]:


def robber(a):
    c=0
    for i in a:
        #hel
        c+=1
        for i in range(0,c):
            print(i)
robber('hel')


# In[36]:


dict={'hello':'hamin',
      'sye':'vaio',
      'what': 'the hell and heaven'
}


# In[37]:


dict


# In[38]:


set=('hello','hemin'',sye','vau=ilo')


# In[39]:


set


# In[40]:


class Myclass:
    x=10
obj=Myclass()
obj.x


# In[48]:


class Myclass:
    def func(self):
        print('hello')
obj=Myclass()
obj.func()


# In[49]:


class Myclass:
    def __init__(self):
        print("function constructor")
    def func(self):
        print("function function")
obj=Myclass()
obj.func()


# In[52]:


class VotingAge:
    def __init__(self,eligibleAge):
        self.eligibleAge = eligibleAge
    def isEligible(self,user_age):
        if user_age >=self.eligibleAge:
            print("eligible for voting")
        else:
            print("not eligible fot voting")
v1=VotingAge(18)
v1.isEligible(25)

v2=VotingAge(16)
v2.isEligible(14)


# In[62]:


class Student:
    def __init__(self,name,course,ha):
        self.name = name
        self.course = course
        self.ha=ha
s=Student("prsv","cse","ohh")
        
print('Student details: ')
print("Name:",s.name)
print("Course:",s.course)
print("ha:",s.ha)
del s.ha

print("\After Modifying Details")
print('Student details: ')


print("Name:",s.name)
print("Course:",s.course)


# In[52]:


import socket
hostn = socket.gethostname()
ipad = socket.gethostbyname(hostn)
print("Ip Address="+ipad)


# In[50]:


import cv2


# In[53]:


s='roose'
v={i for i in s}
print(len(v))


# In[ ]:


import pywhatkit as kit
kit.sendwhatmsg("+918187846479","bondamm..",9,50)


# In[9]:


for i in range(1,21):
    for j in range(1,11):
         print('{0} * {1} =  {2}'.format(i,j,i*j))
    print(" ")
print()


# In[10]:


a =  int(input("enter a number: "))
for i in range(1,11):
    print(a,"x",i,'=',a*i)


# In[7]:


num=[1,2,3,4,5,6,7,8,9]
mylist=[]
for i in num:
    if i%2==0:
        mylist.append(i**2)
    else:
        mylist.append(i**3)
print(mylist)


# In[5]:


mylist


# In[6]:


x,y='12'
y,z='34'
print(x+y+z)


# In[3]:


x


# In[4]:


y


# In[ ]:





# In[12]:





# In[21]:


def nums(num):
    
    return x**3
x= lambda if num:num%2==0 
nums=[1,2,3,4,5,6,7,8,9]


# In[22]:


num


# In[23]:


x


# In[24]:


nums


# In[34]:


help('lambda')


# In[ ]:





# In[18]:


def c(penguin=None):
       if penguin is None:
            penguin = []
            penguin.append("property of the zoo")
            return penguin
c()


# # dogecoin_price

# In[21]:


import requests
from bs4 import BeautifulSoup as bs
url="https://www.coindesk.com/price/dogecoin"
r=requests.get(url)
soup = bs(r.content,'html.parser')
price=soup.find('div',{'class':'price-large'})
print(price.text)


# In[3]:


def length(a,b):
    return a(b)
length(len,'...............')


# # creating_table using python

# In[3]:


from prettytable import PrettyTable
x= PrettyTable()
x.field_names=["city_names","Area","location","num"]
x.add_row(['adelaode',90950,49546,999])
x.add_row(['adela',90950,49546,99])
x.add_row(['laode',950,496,94499])
x.add_row(['elao',90950,546,949])
x.add_row(['ode',90950,49546,999])
x.add_row(['aede',950,446,949])


# In[4]:


print(x)


# In[20]:


class digit:
    def __init__(self,var1):
        self.var1=var1
        print("constructor output",self.var1)
        
        
    def infun(self,var2):
        self.var2=var2
        print("infuc output",self.var2)
obj =digit(5)
obj.infun(15)


# In[21]:


one={'a':1,"b":2}
two={'c':3,'d':4}
merged={**one,**two}
print(merged)


# In[5]:


import wikipedia
result=wikipedia.summary('binary operators')
print(result)


# In[8]:


from pynotifier import Notification


# In[9]:


Notification(title='learn new things',
             description='you should learn new things day by day',
             duration=2
            ).send()


# In[23]:


import pywhatkit
pywhatkit.text_to_handwriting("The sorted() function returns a sorted list of the spwcified iterable.",rgb=('red'))


# #  keywords in python

# In[25]:


import keyword
print(keyword.kwlist)


# In[57]:


len(keyword.kwlist)


# In[1]:



liss=list(set(lis))


# In[2]:


liss


# In[9]:


lis=[1,2,5,6,4,6,5,6,5,7,7,9,9,3,4]
li=[]
for i in lis:
    if i not in li:
        li.append(i)
list(set(li))

