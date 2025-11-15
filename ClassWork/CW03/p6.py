import numpy as np
array1 = np.arange(1, 10).reshape(3, 3)
array2 = np.arange(9, 0, -1).reshape(3, 3)
print(f"The first array, A=\n {array1}")
print(f"The second array, B=\n {array2}")

sumArryas=np.add(array1,array2)
print(f"The sum of first and second arrays, A+B=\n {sumArryas}")

scaler=2
productScalerArray1=np.dot(array1,scaler)
productScalerArray2=np.dot(array2,scaler)
print(f"The scaler * first array, {scaler}*A=\n {productScalerArray1}")
print(f"The scaler * second array, {scaler}*B=\n {productScalerArray2}")

productArray1Array2=np.dot(array1,array2)
# productArray1Array2 = array1 @ array2     
print(f"The inner product of first and second arrays, A.B=\n {productArray1Array2}")

multiplyArray1Array2=np.multiply(array1,array2)
# multiplyArray1Array2 = array1 * array2     
print(f"The multiply of first and second arrays, A*B=\n {multiplyArray1Array2}")

array1Column2=np.take(array1,1,axis=1)
print(f"The second column of array1 is: {array1Column2}")
