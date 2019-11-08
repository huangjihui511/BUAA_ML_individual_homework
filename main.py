import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df.head()
X = [[int(df['age'][i]), df['workclass'][i], int(df['fnlwgt'][i]), df['education'][i]\
         , int(df['education-num'][i]), df['marital-status'][i], df['occupation'][i]\
         , df['relationship'][i], df['race'][i], df['sex'][i]\
         , int(df['capital-gain'][i]), int(df['capital-loss'][i]), int(df['hours-per-week'][i])\
         ,df['native-country'][i]] for i in range(len(df))]
y = []
for x in df['income']:
    if x == '<=50K':
        y.append(0)
    else:
        y.append(1)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.99, random_state=0)
print(len(X_train))
X,y = X_train,y_train


used_features = []
time_counter = {'init':0, 'find':0, 'seperate':0, 'pre_cut':0\
                , 'train':0, 'G_D_a':0, 'H_D':0, 'G_D_a_t_num':0, 'G_D_a_t_type':0}

def H_D(y):
    time_begin = time.time()
    r = 0
    l_y = len(y)
    log_2 = np.log(2)
    count = np.bincount(y)
    for i in set(y):
        r -= count[i]/l_y * np.log(count[i]/l_y) / log_2
    time_end = time.time()
    time_counter['H_D'] += time_end - time_begin
    return r

def T_a(y_a):
    l = [item for item in y_a]
    l.sort()
    l = [(l[i] + l[i + 1]) / 2 for i in range(len(l) - 1)]
    return l
    
def G_D_a_t_num(y,a,t,h_D):
    time_begin = time.time()
    pos_list, neg_list = [], []
    for i in range(len(a)):
        if a[i] > t:
            pos_list.append(y[i])
        else:
            neg_list.append(y[i])
    t = len(pos_list) / len(y) * H_D(pos_list)
    t += len(neg_list) / len(y) * H_D(neg_list)
    time_end = time.time()
    time_counter['G_D_a_t_num'] += time_end - time_begin
    return h_D - t

def G_D_a_t_type(y,a,h_D):
    time_begin = time.time()
    data_sep = {}
    t = 0
    for i in range(len(a)):
        data_sep.setdefault(a[i],[])
        data_sep[a[i]].append(y[i])
    for type_name in data_sep:
        t += len(data_sep[type_name]) / len(y) * H_D(data_sep[type_name])
    time_end = time.time()
    time_counter['G_D_a_t_type'] += time_end - time_begin
    return h_D - t
    
def G_D_a(y,a):
    time_begin = time.time()
    max_value = 0
    max_t = 0
    h_D = H_D(y)
    if type(a[0]) in [type(1),type(0.1)]:
        t_a = T_a(a)
        for t in t_a:
            v = G_D_a_t_num(y,a,t,h_D)
            if v > max_value:
                max_value = v
                max_t = t
    elif type(a[0]) == type('a'):
        max_value = G_D_a_t_type(y,a,h_D)
    time_end = time.time()
    time_counter['G_D_a'] += time_end - time_begin
    return max_value, max_t
import time

class Decision_tree():
    
    def __init__(self):
        pass
    
    def train(self, X, y):
        time_begin = time.time()
        rest_indexs = list(range(len(X[0])))
        self.dtn = Decision_tree_node(rest_indexs, X, y)
        self.dtn.train()
        time_end = time.time()
        print('Training takes',time_end - time_begin,'s')
        
    def predict(self, X):
        y = [self.dtn.predict(x) for x in X]
        return np.array(y)
        
        
class Decision_tree_node():
    
    def __init__(self, rest, X, y):
        time_begin = time.time()
        self.using_index, self.t = self.find_best_index(rest, X, y)
        self.rest = rest
        self.X, self.y = X, y
        self.pos = None
        #print('\tadd\t',used_features,self.using_index)
        print(used_features,self.using_index,len(y))
        used_features.append(self.using_index)
        time_end = time.time()
        time_counter['init'] += time_end - time_begin
        self.sub_trees = {}
    
    def find_best_index(self, rest, X, y):
        
        time_begin = time.time()
        max_index = -1
        max_value = -1
        max_t = 0
        #print(rest,used_features)
        for index in rest:
            if index in used_features:
                continue
            a = [item[index] for item in X]
            g, t = G_D_a(y,a) #
            #print('g',g)
            if g > max_value:
                max_value = g
                max_index = index
                max_t = t
        
        time_end = time.time()
        time_counter['find'] += time_end - time_begin
        return max_index, max_t
    
    def seperate_data_num(self, X, y):
        time_begin = time.time()
        X_pos, X_neg, y_pos, y_neg = [], [], [], []
        index, t = self.using_index, self.t
        for i in range(len(X)):
            if X[i][index] > t:
                X_pos.append(X[i])
                y_pos.append(y[i])
            else:
                X_neg.append(X[i])
                y_neg.append(y[i])
        time_end = time.time()
        time_counter['seperate'] += time_end - time_begin
        return X_pos, X_neg, y_pos, y_neg
    
    def seperate_data_type(self, X, y):
        time_begin = time.time()
        data_sepe_x = {}
        data_sepe_y = {}
        for i in range(len(X)):
            data_sepe_x.setdefault(X[i][self.using_index],[])
            data_sepe_y.setdefault(X[i][self.using_index],[])
            data_sepe_x[X[i][self.using_index]].append(X[i])
            data_sepe_y[X[i][self.using_index]].append(y[i])
        time_end = time.time()
        time_counter['seperate'] += time_end - time_begin
        return data_sepe_x, data_sepe_y
    
    def pre_cut(self, y_pos, y_neg):
        time_begin = time.time()
        count = {}
        for item in self.y:
            count.setdefault(item,0)
            count[item] += 1
        index_max = max(count, key=count.get)
        gener_per_cut = count[index_max] / len(self.y)
        count = {}
        for item in y_pos:
            count.setdefault(item,0)
            count[item] += 1
        index_pos_max = max(count, key=count.get)
        gener_per_not_cut = count[index_pos_max] / len(y_pos)
        for item in y_neg:
            count.setdefault(item,0)
            count[item] += 1
        index_neg_max = max(count, key=count.get)
        gener_per_not_cut += count[index_neg_max] / len(y_neg)
        self.index_max = index_max
        time_end = time.time()
        time_counter['pre_cut'] += time_end - time_begin
        return gener_per_cut < gener_per_not_cut
        # return false if need cut
    
    def train(self):
        time_begin = time.time()
        if H_D(self.y) == 0 or self.using_index == -1:
            self.pos = self.y[0]
        else:
            if True:
                #print(1)
                if type(X[0][self.using_index]) in [type(1),type(0.1)]:
                    X_pos, X_neg, y_pos, y_neg = self.seperate_data_num(self.X, self.y)
                    if len(X_pos) > 0:
                        self.sub_trees['pos'] = Decision_tree_node(self.rest, X_pos, y_pos)
                        self.sub_trees['pos'].train()
                    #print(2)
                    if len(X_neg) > 0:
                        self.sub_trees['neg'] = Decision_tree_node(self.rest, X_neg, y_neg)
                        self.sub_trees['neg'].train()
                elif type(X[0][self.using_index]) == type('a'):
                    data_sepe_x, data_sepe_y = self.seperate_data_type(self.X, self.y)
                    #print('sep:',len(data_sepe_x),self.using_index)
                    for name in data_sepe_x:
                        if len(data_sepe_x[name]) > 0:
                            self.sub_trees[name] = Decision_tree_node(self.rest, data_sepe_x[name], data_sepe_y[name])
                            self.sub_trees[name].train()
            else:
                self.pos = self.index_max
        #print('\tremove\t',used_features,self.using_index)
        time_end = time.time()
        used_features.remove(self.using_index)
        time_counter['train'] += time_end - time_begin
    
    def predict(self,x):
        #print(self.using_index)
        if self.pos != None:
            return self.pos
        elif type(x[self.using_index]) in [type(1),type(0.1)]:
            if x[self.using_index] > self.t:
                return self.sub_trees['pos'].predict(x)
            else:
                return self.sub_trees['neg'].predict(x)
        elif type(x[self.using_index]) == type('a'):
            if self.sub_trees[x[self.using_index]] == None:
                return 2
            else:
                return self.sub_trees[x[self.using_index]].predict(x)
            
                  
dt = Decision_tree()
dt.train(X,y)
from sklearn.metrics import classification_report
y_pred,y_true = dt.predict(X_test), y_test
print(classification_report(y_true, y_pred))