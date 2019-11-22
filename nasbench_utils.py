import numpy as np 
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