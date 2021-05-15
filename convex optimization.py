import numpy as np
from inspect import signature
epsilon = 0.001

def gradient(f,x):
    num_param = len(signature(f).parameters)
    if num_param != x.size:
        raise Exception
    grad = np.zeros(num_param,dtype='float64')

    for i in range(num_param):
        x_c1 = x.copy()
        x_c2 = x.copy()
        x_c1[i] += epsilon
        x_c2[i] -= epsilon
        grad[i] = (f(*x_c1)-f(*x_c2))/(2*epsilon)
    return grad


def hessian(f,x):
    num_param = len(signature(f).parameters)
    if num_param != x.size:
        raise Exception
    hess = np.zeros((num_param,num_param))
    for i in range(num_param):
        eps_i = np.zeros(num_param)
        eps_i[i] = epsilon
        for j in range(num_param):
            eps_j = np.zeros(num_param)
            eps_j[j] = epsilon
            hess[i][j] = (f(*(x+eps_i+eps_j))-f(*(x+eps_i-eps_j))-f(*(x-eps_i+eps_j))+f(*(x-eps_i-eps_j)))/(4*(epsilon**2))
    return hess

def is_convex(f):
    num_param = len(signature(f).parameters)
    for i in range(1000):
        theta = np.random.uniform()
        x1 = np.random.uniform(low=-1,high=1,size=num_param)
        x2 = np.random.uniform(low=-1,high=1,size=num_param)
        if f(*((theta*x1)+((1-theta)*x2)))>((theta*f(*x1))+((1-theta)*f(*x2))):
            return False
    return True

def is_convex_1nd_order(f):
    num_param = len(signature(f).parameters)
    for i in range(1000):
        x = np.random.uniform(low=-1,high=1,size=num_param)
        y = np.random.uniform(low=-1,high=1,size=num_param)
        if f(*x)-f(*y) < gradient(f,y).T@(x-y):
            return False
    return True

def is_convex_2nd_order(f):
    num_param = len(signature(f).parameters)
    for i in range(1000):
        x = np.random.uniform(low=-1, high=1, size=num_param)
        v = np.random.uniform(low=-1, high=1, size=num_param)
        if v.T@hessian(f,x)@v<0:
            return False
    return True


def f(x1,x2,x3,x4):
    return (f2(x1)*f2(x1)*x2)-(x3/f2(1+x2))+(100*x1*np.exp(x3))+x4*f2(x4)


def f1(x1,x2):
    return float(x1)*float(x2)

def f2(x1):
    return float(x1)*float(x1)

def f3(x1):
    return float(x1)*float(x1)*float(x1)

def f4(x1,x2):
    return f1(x1,x2)* f1(x1,x2)


print(gradient(f1,np.full(2,2,dtype='float64')))
print(gradient(f1,np.array([3,5],dtype='float64')))
print(gradient(f2,np.full(1,3,dtype='float64')))
print(is_convex(f3))
print(is_convex(f2))
print(is_convex_1nd_order(f3))
print(is_convex_1nd_order(f2))
print(is_convex_2nd_order(f3))
print(is_convex_2nd_order(f2))
print(hessian(f1,np.full(2,2,dtype='float64')))
print(hessian(f4,np.full(2,2,dtype='float64')))

print(is_convex_2nd_order(f))