import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, load_npz, linalg, diags, save_npz
from scipy.stats import pearsonr
from scipy import float32
import seaborn as sns
import time
import os
from collections import defaultdict
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from collections import OrderedDict

def truncated_normal(loc, var, mini, maxi):
    while True:
        _ = np.random.normal(loc, var)
        if _ <= maxi and _ >= mini:
            return _

def load_data(p='datasets/collaboration.edgelist.txt'):
    with open(p, 'r') as f:
        txt = f.readlines()
        txt = list(map(lambda x: x.replace('\n', '').split('\t'), txt))
        txt = [i for i in txt if i[0].isdigit()]
        txt = list(map(lambda x: [int(x[0]), int(x[1])], txt))
        txt = np.array(txt)

    num_node = np.max(txt) + 1

    S = dok_matrix((num_node,num_node), dtype=float32)
    for i in txt:
        if i[0] < i[1]:
            S[i[0],i[1]] = 1.
            S[i[1], i[0]] = 1.
    S = S.tocsr()

    save_npz(p.replace('.edgelist.txt', '_in.npz'), S)

    return S, num_node

def load_data_npz(p='datasets/collaboration_in.npz'):
    S = load_npz(p)
    num_node = S.shape[0]
    return S, num_node


def p1(S, num_node, name, max_num_node = 10000):
    # # p1

    # ### (a) degree distribution

    # In[210]:

    # %%time

    start = time.time()

    degrees = np.asarray(S.sum(axis=1)).reshape(-1)



    degree_counts = np.array([[np.log(d + 1), np.log(c)] for (d, c) in Counter(degrees).items()])



    def least_squares(x, y):
        x_ = np.concatenate((x, np.ones(x.shape)), axis=1)
        return np.dot(np.linalg.inv(np.dot(x_.T, x_)), np.dot(x_.T, y))

    x, y = degree_counts[:, 0].reshape(-1, 1), degree_counts[:, 1].reshape(-1, 1)

    [slope, bias] = least_squares(x, y).reshape(-1).tolist()

    def infer_y(x, slope, bias):
        return x * slope + bias

    v_infer_y = np.vectorize(lambda x: infer_y(x, slope, bias))

    inferred_y = v_infer_y(x)

    end = time.time()
    elapased_time = round(end - start, 4)

    plt.scatter(x, y)
    plt.plot(x, inferred_y, c='orange')
    # plt.text(.1, .1, 'slope: {}, bias: {}'.format(str(round(slope, 2)), str(round(bias, 2))),
    #          fontsize=12)
    plt.xlabel('log(degree+1)')
    plt.ylabel('log(count)')
    plt.title('degree distribution plot, slope: {}, bias: {},  \n time: {}'.format(str(round(slope, 2)), str(round(bias, 2)), elapased_time))
    plt.savefig(name + '_degree.png')
    plt.clf()

    print('\n', '=' * 5, 'degree', elapased_time)

    # ### (a.2) subsampling

    # In[71]:

    # rand_idx = np.random.choice(np.array(list(range(num_node))), size = 5000, replace = False)

    # S = S[rand_idx,:][:, rand_idx]

    # num_node = 5000
    # degrees = degrees[rand_idx]

    # ### (b) clustering coefficients

    # In[211]:
    # %%time

    if num_node > max_num_node:
        sub_size = max_num_node
        rand_idx = np.random.choice(np.array(list(range(num_node))), size=sub_size, replace=False)
        S_sub = S[rand_idx, :][:, rand_idx]
        num_node_sub = sub_size
        degrees_sub = degrees[rand_idx]
    else:
        S_sub = S
        num_node_sub = num_node
        degrees_sub = degrees

    scaling = num_node / num_node_sub

    # %%time
    start = time.time()

    S_3 = S_sub.dot(S_sub).dot(S_sub)

    clusterings = np.nan_to_num(S_3.diagonal() / ( 1 + degrees_sub * (degrees_sub - 1)))

    clustering_mean = np.mean(clusterings)

    clustering_counts = np.array([[cl, np.log(c)] for (cl, c) in Counter(clusterings).items()])

    x, y = clustering_counts[:, 0].reshape(-1, 1), clustering_counts[:, 1].reshape(-1, 1)

    plt.scatter(x, y, alpha=0.2)

    end = time.time()
    elapased_time = round(end - start, 4)

    # plt.text(x[0], y[0]  , 'local clustering mean: {}'.format(str(round(clustering_mean,2))),
    #          fontsize=12)
    plt.xlabel('local clustering')
    plt.ylabel('log(count)')
    plt.title('local clustering distribution plot, local clustering mean: {}, \n  time: {}'.format(str(round(clustering_mean, 2)),
                                                                                               elapased_time))
    plt.savefig(name + '_clustering.png')
    plt.clf()
    print('\n', '=' * 5, 'clustering', elapased_time)
    # ### (c) shortest path distribution

    # In[198]:



    # In[213]:
    start = time.time()



    row_nonzero, col_nonzero = S_sub.nonzero()

    aja_list = defaultdict(list)

    for ind, i in enumerate(row_nonzero):
        aja_list[i].append(col_nonzero[ind])



    all_dist = defaultdict(dict)
    for i in tqdm(range(num_node_sub)):
        all_dist[i][i] = 0
        neighs = aja_list[i]
        queue = deque()
        for j in neighs:
            all_dist[i][j] = 1
            queue.append((j, 1))
        if queue:
            while queue:
                cur_node, cur_dis = queue.popleft()
                neighs = aja_list[cur_node]
                for j in neighs:
                    if j not in all_dist[i]:
                        all_dist[i][j] = cur_dis + 1
                        queue.append((j, cur_dis + 1))

    all_dist_vals = []
    for i in all_dist:
        temp = all_dist[i]
        for j in temp:
            all_dist_vals.append(temp[j])

    shortest_path_mean = np.mean(all_dist_vals)

    inf_dist_count = num_node_sub * num_node_sub - len(all_dist_vals)

    all_dist_val_counts = Counter(all_dist_vals)

    all_dist_val_counts = np.array([[np.log(pl + 1), np.log(c)] for (pl, c) in all_dist_val_counts.items()])

    x, y = all_dist_val_counts[:, 0].reshape(-1, 1), all_dist_val_counts[:, 1].reshape(-1, 1)

    end = time.time()
    elapased_time = round(end - start, 4)

    plt.scatter(x, y)
    # plt.text(x[0], y[0] // 2, 'shortest path length mean: {}'.format(round(shortest_path_mean, 2)),
    #          fontsize=12)
    plt.xlabel('log(shortest path length+1)')
    plt.ylabel('log(count)')
    plt.title('all pair shortest path length distribution plot, shortest path length mean: {}, \n  time : {}'.format(
        round(shortest_path_mean, 2), elapased_time))
    plt.savefig(name + '_path.png')
    plt.clf()

    print('\n', '=' * 5, 'paths', elapased_time)

    # ### (d) number of connected components, the portion of nodes that are in the GCC

    # In[201]:

    # %%time
    start = time.time()

    all_comp = {}
    base_comp = 0
    for i in tqdm(range(num_node_sub)):
        if i not in all_comp:
            all_comp[i] = base_comp
            neighs = aja_list[i]
            queue = deque()
            for j in neighs:
                all_comp[j] = base_comp
                queue.append((j, base_comp))
            if queue:
                while queue:
                    cur_node, cur_dis = queue.popleft()
                    neighs = aja_list[cur_node]
                    for j in neighs:
                        if j not in all_comp:
                            all_comp[j] = base_comp
                            queue.append((j, base_comp))
            base_comp += 1

    components = np.array(list(all_comp.values()))

    component_counter = Counter(components)

    print('number of connected component: ', int(len(component_counter) * scaling))
    print('portion of nodes in giant connected component: ', round(max(component_counter.values()) / num_node, 2))

    plt.text(0, 0, 'number of connected component: {}, \n portion of nodes in giant connected component: {}, \n time : {}'.format(
        int(len(component_counter) * scaling), round(max(component_counter.values()) / num_node, 2), elapased_time))
    plt.savefig(name + '_component.png')
    plt.clf()

    end = time.time()
    elapased_time = round(end - start, 4)

    # os.system(
    #     'echo {} >> {}'.format('number of connected component: {}'.format(int(len(component_counter) * scaling)), name + '_component.txt'))
    # os.system('echo {} >> {}'.format('portion of nodes in giant connected component: {}'.format(
    #     round(max(component_counter.values()) / num_node, 2)), name + '_component.txt'))
    # os.system('echo {} >> {}'.format('time: {}'.format(elapased_time), name + '_component.txt'))

    print('\n', '=' * 5, 'connected component', elapased_time)

    # ### (e) eigenvalue distribution (compute the spectral gap)

    # In[214]:

    # %%time

    start = time.time()

    degree_diagonal = diags(degrees, 0)

    laplacian = degree_diagonal - S

    eigen_vals = linalg.eigsh(laplacian, k=20, return_eigenvectors=False)

    eigen_vals = sorted(eigen_vals, reverse=True)

    eigen_gap = eigen_vals[0] - eigen_vals[1]

    end = time.time()
    elapased_time = round(end - start, 4)

    sns.distplot(eigen_vals)
    plt.title('top 20 laplacian eigenvalue distribution, eigengap: {}, \n time: {}'.format(str(round(eigen_gap, 2)), elapased_time))
    plt.xlabel('eigenvalue')
    plt.ylabel('density')
    plt.savefig(name + '_eigen.png')
    plt.clf()

    print('\n', '=' * 5, 'eigen', elapased_time)

    # ### (f) degree correlations (plot as scatter di vs dj , also report the overall correlation).


    # %%time
    start = time.time()

    x = []
    y = []
    for ind, i in enumerate(row_nonzero):
        j = col_nonzero[ind]
        x.append(degrees[i])
        y.append(degrees[j])
    x = np.array(x)
    y = np.array(y)

    asso_coef = pearsonr(x, y)[0]

    end = time.time()
    elapased_time = round(end - start, 4)

    plt.scatter(x, y, alpha=.1)
    plt.title('assortativity plot, coef = {}, \n  time: {}'.format(str(round(asso_coef, 2)), elapased_time))
    plt.xlabel('left degree')
    plt.ylabel('right degree')
    plt.savefig(name + '_degreecorr.png')
    plt.clf()

    print('\n', '=' * 5, 'degree corr', elapased_time)

    # ### (g) degree-clustering coefficient relation (plot as scatter di vs ci)

    # In[218]:

    # %%time
    start = time.time()

    degree_clustering = defaultdict(list)
    for ind, i in enumerate(degrees_sub):
        degree_clustering[int(i)].append(clusterings[ind])
    x = []
    y = []
    for i in degree_clustering:
        x.append(i)
        y.append(np.mean(degree_clustering[i]))
    x = np.array(x)
    y = np.array(y)

    end = time.time()
    elapased_time = round(end - start, 4)

    plt.scatter(x, y)
    plt.title('degree clustering plot, \n  time: {}'.format(elapased_time))
    plt.xlabel('degree')
    plt.ylabel('average local clustering')
    plt.savefig(name + '_degreecluster.png')
    plt.clf()

    print('\n', '=' * 5, 'degree cluster', elapased_time)


def p2_new(S, num_node):
    sample_dict = {}
    m = S.nnz // num_node // 2
    initial_num_node = max(4, m)
    t = num_node - initial_num_node

    S_ba = dok_matrix((num_node,num_node), dtype=float32)
    c = 0
    for i in range(initial_num_node):
        for j in range(initial_num_node):
            sample_dict[c] = i
            c += 1
            sample_dict[c] = j
            c += 1
            S_ba[i,j] = 1
            S_ba[j,i] = 1 
            
    for i in tqdm(range(initial_num_node, initial_num_node + t)):
        ns = np.random.randint(c, size=m)
        ns = [sample_dict[i] for i in ns]
        ns = list(set(ns))
        for n in ns:
            sample_dict[c] = i
            c += 1
            sample_dict[c] = n
            c += 1
            S_ba[i,n] = 1
            S_ba[n,i] = 1

    S_ba = S_ba.tocsr()

    return S_ba, num_node


def p3_new(S, num_node):
    sample_dict = {}
    m = S.nnz // num_node // 2
    initial_num_node = max(4, m)
    t = num_node - initial_num_node

    S_ba = dok_matrix((num_node,num_node), dtype=float32)
    c = 0
    for i in range(initial_num_node):
        for j in range(initial_num_node):
            sample_dict[c] = i
            c += 1
            sample_dict[c] = j
            c += 1
            S_ba[i,j] = 1
            S_ba[j,i] = 1 
            
    for i in tqdm(range(initial_num_node, initial_num_node + t)):
        ns = np.random.randint(c, size=1+int(np.log(i)))
        ns = [sample_dict[i] for i in ns]
        ns = list(set(ns))
        for n in ns:
            sample_dict[c] = i
            c += 1
            sample_dict[c] = n
            c += 1
            S_ba[i,n] = 1
            S_ba[n,i] = 1

    S_ba = S_ba.tocsr()

    return S_ba, num_node


if __name__ == '__main__':


    max_num_node = 5000

    name_dir_dict = OrderedDict()
    _ = [
#         ('collaboration', 'datasets/collaboration_in.npz'),
#         ('citation', 'datasets/citation_in.npz'),
#         ('actor', 'datasets/actor_in.npz'),
#         ('email', 'datasets/email_in.npz'),
#         ('internet', 'datasets/internet_in.npz'),
#         ('metabolic', 'datasets/metabolic_in.npz'),
#         ('phonecalls', 'datasets/phonecalls_in.npz'),
#         ('powergrid', 'datasets/powergrid_in.npz'),
        ('protein', 'datasets/protein_in.npz'),
        ('www', 'datasets/www_in.npz')
    ]
    for i, j in _:
        name_dir_dict[i] = j
        
    for data in name_dir_dict:
        print('\n','-'*50, data)
        S, num_node = load_data_npz(name_dir_dict[data])
        print('num_node: {}, num_edge: {}'.format(num_node, S.nnz))
        p1(S, num_node, 'results/' + data, max_num_node)
        print('\n','*' * 50, 'ba ' + data)
        S_ba, num_node = p2_new(S, num_node)
        p1(S_ba, num_node, 'results/ba_' + data, max_num_node)
        print('\n', '*' * 50, 'modified ba ' + data)
        S_ba, num_node = p3_new(S, num_node)
        p1(S_ba, num_node, 'results/mod_ba_' + data, max_num_node)


