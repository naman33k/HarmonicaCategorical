import all_harmonica_code as harmonica
import numpy as np
import itertools
"""# Synthetic Test Code"""

num_vars = 4
category_list = [1,2,3,4]

#randomdatagen = lambda l: [np.random.randint(i) for i in l]
datax = []
datay = []
value_list = []
for i in range(len(category_list)):
  value_list.append(np.arange(category_list[i]))
return_vals = None
for vals in itertools.product(*value_list):
  datax += [list(vals)]
  datay.append(np.random.random(1)[0])
print(datay)
print(datax)
data = [np.array(datax), np.array(datay)]

#print(harmonica_one_stage(category_list, data, 2, 4, 100, convertToFourier))
#print(harmonica_multistage(data, convertToFourier, category_list, 2, 4, 2, 0.5))
train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(data, data, category_list, harmonica.convertToFourier, 0)
print("degree 0")
print(train_error)
print(test_error)
train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(data, data, category_list, harmonica.convertToFourier, 1)
print("degree 1")
print(train_error)
print(test_error)
train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(data, data, category_list, harmonica.convertToFourier, 2)
print("degree 2")
print(train_error)
print(test_error)
train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(data, data, category_list, harmonica.convertToFourier, 3)
print("degree 3")
print(train_error)
print(test_error)
train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(data, data, category_list, harmonica.convertToFourier, 4)
print("degree 4")
print(train_error)
print(test_error)