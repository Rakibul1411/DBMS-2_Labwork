import numpy as np

def load_transactions(path_to_data, order):
  Transactions = []
  with open(path_to_data, 'r') as fid:
    for lines in fid:
      str_line = list(lines.strip.split(','))
      _t = list(np.unique(str_line))
      _t.sort(key=lambda x: order.index(x)) # order = ['I1', 'I2', 'I3', 'I4']
      Transactions.append(_t)
  return Transactions    


def count_occurences(itemset, Transactions):
  count = 0
  for i in range(len(Transactions)):
    if set(itemset).issubset(set(Transactions[i])):
      count += 1
  return count    


def join_two_itemsets(it1, it2, order):
  it1.sort(key=lambda x: order.index(x))
  it2.sort(key=lambda x: order.index(x))
  
  for i in range(len(it1)-1):
    if it1[i] != it2[i]:
      return []
  
  if order.index(it1[-1]) < order.index(it2[-1]):
    return it1 + [it2[-2]]  
  return []


def join_set_itemsets(set_of_its, order):
  C = []
  for i in range(len(set_of_its)):
    for j in range(i+1, len(set_of_its)):
      it_out = join_set_itemsets(set_of_its[i], set_of_its[j], order)
      if len(it_out) > 0:
        C.append(it_out)
  return C     


def get_frequent(itemsets, Transactions, min_support, prev_discarded):
  L = []
  supp_count = []
  new_discarded = []
  
  k = len(prev_discarded.keys())
  
  for s in range(len(itemsets)):
    discarded_before = False
    if k > 0:
      for it in prev_discarded[k]:
        if set(it).issubset(set(itemsets[s])):
          discarded_before = True
          break
    
    if not discarded_before:
      count = count_occurences(itemsets[s], Transactions)
      if count/len(Transactions) >= min_support:
        L.append(itemsets[s])
        supp_count.append(count)
      else:
        new_discarded.append(itemsets[s])
              
  return L, supp_count, new_discarded           