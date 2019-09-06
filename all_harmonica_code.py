import numpy as np
import cmath
import math
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import itertools

def convertToOneHot(dataset, categories):
  sumval = sum(categories)
  output = np.zeros((len(dataset), sumval))
  for i in range(len(dataset)):
    total_j = 0
    for j in range(len(categories)):
      output[i][total_j + dataset[i][j]] = 1
      total_j += categories[j]
  return output

def convertToFourier(dataset, categories):
  sumval = sum(categories)
  output = np.zeros((len(dataset), sumval),dtype=np.complex_)
  for i in range(len(dataset)):
    total_j = 0
    for j in range(len(categories)):
      for k in range(categories[j]):
        output[i][total_j] = np.exp(2*cmath.pi*complex(0., 1.)*k*dataset[i][j]/categories[j])
        total_j += 1
  return output

def convertToPolyFeaturesCategorical(data, degree, categories):
  return [polyFeaturesCategoriesSingleDatapoint(d, degree, categories) for d in data], polyFeaturesCategoriesIdentity(degree, categories)

def filterRelevantVariables(exp_relevant):
  answer_set = set([])
  for m in exp_relevant:
    for r in m:
      answer_set = answer_set | set([r[0]])
  return list(answer_set) 

def evaluatePol(relevantVars, flattenedVals, polynomialDesc, coeffs, categories):
  assert(len(relevantVars)==len(categories))
  ## convert to a 2d array
  twodvals = []
  sumi = 0
  for c in categories:
    twodvals.append(flattenedVals[sumi:sumi+c])
    sumi += c
  #maybe this needs to be a complex definition
  sumval = 0.0
  relevantVarsInverse = {}
  for i in range(len(relevantVars)):
    relevantVarsInverse[relevantVars[i]] = i
  for i in range(len(polynomialDesc)):
    prodval = 1.0
    for entry in polynomialDesc[i]:
      prodval *= twodvals[relevantVarsInverse[entry[0]]][entry[1]]
    prodval*= coeffs[i]
    sumval += prodval
  return sumval


######### Internal Functions ##############################

def polyFeaturesCategoriesSingleDatapoint(data, degree, categories):
  assert(len(data)==sum(categories))
  # data is finished
  if categories == []:
    return [1]
  # no more degree left
  if degree==0:
    return [1]
  else:
    #Initialize with the recursion. Represents not taking this variable.
    answer = polyFeaturesCategoriesSingleDatapoint(data[categories[0]:], degree, categories[1:])
    # Recurision representing taking this variable. 
    keep_this = polyFeaturesCategoriesSingleDatapoint(data[categories[0]:], degree-1, categories[1:])
    for i in range(categories[0]):
      answer+= [a*data[i] for a in keep_this]
    return answer

def polyFeaturesCategoriesIdentity(degree, categories, varname=0):
  if categories == []:
    return [[]]
  if degree==0:
    return [[]]
  else:
    #Initialize with the recursion. Represents not taking this variable.
    answer = polyFeaturesCategoriesIdentity(degree, categories[1:],varname+1)
    # Recurision representing taking this variable. 
    keep_this = polyFeaturesCategoriesIdentity(degree-1, categories[1:], varname+1)
    for i in range(categories[0]):
      answer+= [a + [[varname, i]] for a in keep_this]
    return answer

def random_sample_data(list_of_datasets, num_samples):
  random_data = []
  rand_indices = np.random.choice(len(list_of_datasets[0]), num_samples, replace=False)
  for x in list_of_datasets:
    random_x = x[rand_indices]
    random_data.append(random_x)
  assert len(list_of_datasets) == len(random_data)
  return random_data

def minimize_polynomial(relevant_variables, relevant_monomials, coeffs, category_list):
  # minimize sum of monomials
  # forget complex part
  fixed_var_dict = {}
  best_val = 0.0
  value_list = []
  for i in range(len(category_list)):
    value_list.append(np.arange(category_list[i]))
  return_vals = None
  for vals in itertools.product(*value_list):
    flattened_vals = convertToFourier([vals], category_list)
    curr_val = evaluatePol(relevant_variables, flattened_vals[0], relevant_monomials, coeffs, category_list)
    if isinstance(curr_val, np.complex_):
      curr_val_real = curr_val.real
    else:
      curr_val_real = curr_val
    if curr_val_real > best_val:
      best_val = curr_val_real
      return_vals = vals
  assert(return_vals != None)
  return return_vals

def update_dataset(data, relevant_variables, min_values, leftover_variables):
  x,y = data
  index_set = []
  for i in range(len(x)):
    flag = 0
    for j in range(len(relevant_variables)):
      if not(x[i][relevant_variables[j]] == min_values[j]):
        flag = 1
        break
    if flag == 0:
      index_set.append(i)
  new_x_ = x[index_set]
  new_y = y[index_set]
  new_x = new_x_[:, leftover_variables]
  return (new_x, new_y)

def update_index_dict(relevant_variables, index_dict):
  updated_index_dict = {}
  orig_indices = list(index_dict.keys())
  i = 0
  index = 0
  while i < len(orig_indices):
    if i not in relevant_variables:
      updated_index_dict[index] = index_dict[i]
      index += 1
    i += 1
  return updated_index_dict

def lasso_solver(x, y):
  lasso_solver_internal = LassoCV(fit_intercept=False, cv=5)
  
  # fit only the real part
  if isinstance(x[0][0], np.complex_):
     lasso_solver_internal.fit(x.real, y)
  else:
    lasso_solver_internal.fit(x, y)
  preds = lasso_solver_internal.predict(x.real)
  train_error = sum(map(lambda x,y:np.absolute(x-y)**2,preds,y))
  return lasso_solver_internal.coef_

def harmonica_multistage(data, basis_change, category_list, d, s, r, sampling_schedule):
  fixed_variables = {}
  index_dict = {}
  for i in range(len(category_list)):
    index_dict[i] = i
  print("in multistage")
  accuracy_vals = []
  for c in range(r):
    #print("Average accuracy of the dataset ", sum(data[1])/len(data[1]))
    #accuracy_vals.append(sum(data[1])/len(data[1]))
    print("Max accuracy of the dataset ", max(data[1]))
    accuracy_vals.append(max(data[1]))
    print("Starting Stage ",c)
    if len(data[1]) >= sampling_schedule[c]:
      relevant_variables, min_values = harmonica_one_stage(category_list, data, d, s, sampling_schedule[c], basis_change)
    else:
      relevant_variables, min_values = harmonica_one_stage(category_list, data, d, s, len(data[1]), basis_change)
      sampling_schedule[c] = len(data[1])
    print("Postprocessing for Stage ", c)
    for i in range(len(relevant_variables)):
      fixed_variables[index_dict[relevant_variables[i]]] = min_values[i]
    leftover_variables = []
    for i in range(len(category_list)):
      if not(i in relevant_variables):
        leftover_variables.append(i)
    new_category_list = [category_list[i] for i in leftover_variables]
    updated_dataset = update_dataset(data, relevant_variables, min_values, leftover_variables)
    print("Length of updated dataset", len(updated_dataset[0]))
    if len(updated_dataset[0])<5: 
      print("Dataset has become too small")
      break
    updated_index_dict = update_index_dict(relevant_variables, index_dict)
    category_list = new_category_list
    data = updated_dataset
    index_dict = updated_index_dict    
  if(len(data[1]) > 0):
    print("Final Average accuracy of the dataset ", sum(data[1])/len(data[1]))
    accuracy_vals.append(sum(data[1])/len(data[1]))
  return fixed_variables, accuracy_vals, sampling_schedule

def harmonica_one_stage_inner(category_list, flat_data, d, s=None):
  x, y = flat_data
  # Expanding to polynomial features
  x_poly_exp_list, exp_identity_list = convertToPolyFeaturesCategorical(x, d, category_list)
  print("Number of features ", len(exp_identity_list))
  #TODO: Fix the fact that this returns lists
  x_poly_exp = np.array(x_poly_exp_list)
  exp_identity = np.array(exp_identity_list)
  coef = lasso_solver(x_poly_exp, y)
  if s==None:
    non_zero_coeff = coef
    exp_relevant = exp_identity
  else:
    index = np.argsort(-np.abs(coef))
    value = np.abs(coef[index[s - 1]])
    non_zero_coeff_index = np.abs(coef) >= value
    exp_relevant = exp_identity[non_zero_coeff_index]
    non_zero_coeff = coef[non_zero_coeff_index]
  relevant_variables = filterRelevantVariables(exp_relevant)
  relevant_category_list = []
  for i in range(len(category_list)):
    if i in relevant_variables:
      relevant_category_list.append(category_list[i])
  return relevant_variables, relevant_category_list, exp_relevant, non_zero_coeff

def harmonica_one_stage(category_list, data, d, s, num_samples, basis_changer):
  x,y = data
  if num_samples < len(data[0]):
    x_sample, y_sample = random_sample_data(data, num_samples)
  else:
    x_sample = x
    y_sample = y
  x_flat = basis_changer(x_sample,category_list)
  #print("running harmonica inner loop")
  relevant_variables, relevant_category_list, exp_relevant, coef = harmonica_one_stage_inner(category_list, [x_flat, y_sample], d, s)
  #print("minimize_polynomial")
  min_values = minimize_polynomial(relevant_variables, exp_relevant, coef, relevant_category_list)
  return relevant_variables, min_values

def prediction(datapoint, relevant_variables, exp_relevant, coef, category_list, basis_changer):
  flattened_data = basis_changer([datapoint], category_list)
  relevant_category_list = [category_list[i] for i in relevant_variables]
  curr_val = evaluatePol(relevant_variables, flattened_data[0], exp_relevant, coef, relevant_category_list)
  if isinstance(curr_val, np.complex_):
    curr_val_real = curr_val.real
  else:
    curr_val_real = curr_val
  return curr_val_real

#Naman needs to test this
def predict_loss_harmonica(train_data, test_data, category_list, basis_changer, d, s=None):
  x_train,y_train = train_data
  print("Number of Data points ", len(y_train))
  x_test, y_test = test_data
  x_train_flat = basis_changer(x_train,category_list)
  print("running harmonica inner loop")
  relevant_variables, relevant_category_list, exp_relevant, coef = harmonica_one_stage_inner(category_list, [x_train_flat, y_train], d, s)
  train_preds = [prediction(datapoint, relevant_variables, exp_relevant, coef, category_list, basis_changer) for datapoint in x_train]
  test_preds = [prediction(datapoint, relevant_variables, exp_relevant, coef, category_list, basis_changer) for datapoint in x_test]
  train_error = sum(map(lambda x,y:np.absolute(x-y)**2,train_preds,y_train))
  test_error = sum(map(lambda x,y:np.absolute(x-y)**2,test_preds,y_test))
  if len(x_train)==0:
    train_mean = 0
  else:
    train_mean = train_error/len(x_train)
  if len(x_test)==0:
    test_mean = 0
  else:
    test_mean = test_error/len(x_test)
  
  return train_mean, test_mean, train_preds, test_preds