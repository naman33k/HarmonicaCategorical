"""#Experiments"""

# Download the raw data (only 108 epoch data points, for full dataset,
# uncomment the second line for nasbench_full.tfrecord).

# !curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
# # !curl -O https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

# # Clone and install the code and dependencies.

# !git clone https://github.com/google-research/nasbench
# !pip install ./nasbench

# # Initialize the NASBench object which parses the raw data into memory (this
# # should only be run once as it takes up to a few minutes).
from nasbench import api

# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench = api.NASBench('nasbench_only108.tfrecord')

import numpy as np 
import all_harmonica_code as harmonica
import itertools
import pandas as pd

block_dict = {
"conv3x3-bn-relu": 0,
"conv1x1-bn-relu": 1, 
"maxpool3x3": 2,
"input": 3,
"output": 4,
}

# range of number of edges: 2 to 7
# No. of datapoints: 1, 6, 84, 2441, 62010, 359082
# So there are 359082 models with 7 edges and vertices

def convert_categorical_data_frame_to_standard_basis(data, categories):
  converted_data = []
  for d in data:
    flattened_data = []
    for i in range(len(d)):
      l = [0]*categories[i]
      l[d[i]] = 1
      flattened_data += l
    converted_data.append(flattened_data)
  return converted_data

# iterate through all architectures, reformat
def get_all_architectures():
  """Returns all architectures."""
  index_upper = np.triu_indices(7, 1)
  all_module_vars = []
  ops_vars = [0] * 5
  yvals = []
  for unique_hash in nasbench.hash_iterator():
    # module_adjacency, module_operations in fixed_metrics
    # halfway_training_time, halfway_train_accuracy, halfway_validation_accuracy, 
    # halfway_test_accuracy, final_training_time, final_train_accuracy
    # final_validation_accuracy, final_test_accuracy in computed_metrics
    fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(
          unique_hash)
    if len(fixed_metrics['module_adjacency']) != 7: 
      continue
    module_edges = fixed_metrics['module_adjacency']
    module_ops = fixed_metrics['module_operations']
    module_edge_vars = module_edges[index_upper]
    for i in range(1,6):
      ops_vars[i-1] = block_dict[module_ops[i]]
    module_vars = np.concatenate((module_edge_vars, ops_vars))
    assert len(module_vars) == 26
    all_module_vars.append(module_vars)
    yvals.append(computed_metrics[108][0]['final_test_accuracy'])
  
  cols = []
  cats = []
  for i in range(21):
    cols.append('edge_' + str(i))
    cats.append(2)
  for i in range(5):
    cols.append('op_' + str(i))
    cats.append(3)

  print(cols)
  print(cats)
  all_module_vars_flat = convert_categorical_data_frame_to_standard_basis(all_module_vars, cats)
  df = pd.DataFrame(all_module_vars, columns=cols)
  flat_df = pd.DataFrame(all_module_vars_flat)
  
  return df, flat_df,yvals


df, flat_df, yvals = get_all_architectures()

d = 1
s = 5
r = 1
category_list = [2] * 21 + [3] * 5
# best_arch = harmonica_multistage(category_list, (convertToFourier(flat_df.values[:1000], category_list), np.array(yvals[:1000])), d, s, r, 100)

train_datax = df.values[:100000]
train_datay = yvals[:100000]
train_data = [np.array(train_datax), np.array(train_datay)]

test_datax = df.values[-1000:]
test_datay = yvals[-1000:]
test_data = [np.array(test_datax), np.array(test_datay)]

# train_error, test_error, train_preds, test_preds = predict_loss_harmonica(train_data, train_data, category_list, convertToFourier, 0)
# print(train_error)
# print(test_error)
train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(train_data, test_data, category_list, harmonica.convertToFourier, 0)
print("degree 0")
print(train_error)
print(test_error)
train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(train_data, test_data, category_list, harmonica.convertToFourier, 1)
print("degree 1")
print(train_error)
print(test_error)
train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(train_data, test_data, category_list, harmonica.convertToFourier, 2)
print("degree 2")
print(train_error)
print(test_error)
# train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(train_data, test_data, category_list, harmonica.convertToFourier, 3)
# print("degree 3")
# print(train_error)
# print(test_error)

# train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(train_data, test_data, category_list, harmonica.convertToOneHot, 0)
# print("degree 0")
# print(train_error)
# print(test_error)
# train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(train_data, test_data, category_list, harmonica.convertToOneHot, 1)
# print("degree 1")
# print(train_error)
# print(test_error)
# train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(train_data, test_data, category_list, harmonica.convertToOneHot, 2)
# print("degree 2")
# print(train_error)
# print(test_error)
# train_error, test_error, train_preds, test_preds = harmonica.predict_loss_harmonica(train_data, test_data, category_list, harmonica.convertToOneHot, 3)
# print("degree 3")
# print(train_error)
# print(test_error)



# 10k examples or more, up to 100k, degree 0, 1, 2, 3, see prediction, in fourier and 1-hot
# plot prediction error