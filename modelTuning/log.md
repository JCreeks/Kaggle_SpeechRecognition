
Traceback (most recent call last):
  File "lightWeight_Tuning.py", line 73, in <module>
    random_search.fit(x_train, y_train)
  File "/home/ubuntu/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py", line 625, in fit
    base_estimator = clone(self.estimator)
  File "/home/ubuntu/anaconda3/lib/python3.6/site-packages/sklearn/base.py", line 62, in clone
    new_object_params[name] = clone(param, safe=False)
  File "/home/ubuntu/anaconda3/lib/python3.6/site-packages/sklearn/base.py", line 53, in clone
    return copy.deepcopy(estimator)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 180, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 280, in _reconstruct
    state = deepcopy(state, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 150, in deepcopy
    y = copier(x, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 240, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 150, in deepcopy
    y = copier(x, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 215, in _deepcopy_list
    append(deepcopy(a, memo))
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 180, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 280, in _reconstruct
    state = deepcopy(state, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 150, in deepcopy
    y = copier(x, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 240, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 180, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 280, in _reconstruct
    state = deepcopy(state, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 150, in deepcopy
    y = copier(x, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 240, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 180, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 280, in _reconstruct
    state = deepcopy(state, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 150, in deepcopy
    y = copier(x, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 240, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
  File "/home/ubuntu/anaconda3/lib/python3.6/copy.py", line 169, in deepcopy
    rv = reductor(4)
TypeError: can't pickle _thread.lock objects
