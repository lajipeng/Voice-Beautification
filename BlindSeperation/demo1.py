import numpy as np
a = np.array([-5.6j,  0.2j,  11.  ,  1+1j])
print(  '我们的数组是：')
print( a)
print('\n')
print ('调用 real() 函数：')
print (np.real(a))
print(  '\n')
print(  '调用 imag() 函数：')
print( np.imag(a))
print(  '\n')
print(  '调用 conj() 函数：')
print( np.conj(a))
print(  '\n')
print(  '调用 angle() 函数：')
print( np.angle(a))
print(  '\n')
print(  '再次调用 angle() 函数(以角度制返回)：')
print( np.angle(a, deg =  True))
print('\n')
print(np.abs(a))