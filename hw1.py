import numpy as np
from scipy.sparse import linalg, diags
from scipy.sparse import dok_matrix, csr_matrix
from scipy import float32
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque
from collections import defaultdict
from tqdm import tqdm
import time
import os

def load_data(p='datasets/collaboration.edgelist.txt'):
    with open(p, 'r') as f:
        txt = f.readlines()
        txt = list(map(lambda x: x.replace('\n', '').split('\t'), txt))
        txt = list(map(lambda x: [int(x[0]), int(x[1])], txt))
        txt = np.array(txt)

    num_node = np.max(txt) + 1

    S = dok_matrix((num_node,num_node), dtype=float32)
    for i in txt:
        S[i[0],i[1]] = 1.
    S = S.tocsr()

    return S, num_node


# # p1

# ### (a) degree distribution

# In[149]:

def p1(S, num_node, name):
    print('='*10, 'degree')

    start = time.time()

    degrees = np.asarray(S.sum(axis=1)).reshape(-1)
    degree_counts = np.array([[np.log(d + 1), np.log(c)] for (d, c) in  Counter(degrees).items()])
    def least_squares(x, y):
        x_ = np.concatenate((x, np.ones(x.shape)), axis=1)
        return np.dot(np.linalg.inv(np.dot(x_.T, x_)), np.dot(x_.T, y))
    x, y = degree_counts[:,0].reshape(-1,1), degree_counts[:,1].reshape(-1,1)
    [slope, bias] = least_squares(x, y).reshape(-1).tolist()
    def infer_y(x, slope, bias):
        return x * slope + bias
    v_infer_y = np.vectorize(lambda x: infer_y(x, slope, bias))
    inferred_y = v_infer_y(x)

    end = time.time()
    elapased_time = round(end - start, 4)

    caption = 'slope: {}, bias: {}, time: {}'.format(round(slope, 2), round(bias, 2), elapased_time)

    plt.scatter(x, y)
    plt.plot(x, inferred_y, c='orange')
    plt.xlabel('log(degree+1)')
    plt.ylabel('log(count)')
    plt.title('degree distribution plot\n' + caption)
    # plt.show()
    plt.savefig(name + '_degree.png')
    plt.clf()



    # ### (b) clustering coefficients

    # In[148]:

    print('=' * 10, 'clustering')
    # %%time
    start = time.time()

    S_3 = S.dot(S).dot(S)

    clusterings = np.nan_to_num(S_3.diagonal() / (degrees * (degrees - 1)))

    clustering_mean = np.mean(clusterings)

    clustering_counts = np.array([[cl, np.log(c)]     for (cl, c) in  Counter(clusterings).items()])

    x, y = clustering_counts[:,0].reshape(-1,1),     clustering_counts[:,1].reshape(-1,1)

    end = time.time()
    elapased_time = round(end - start, 4)

    plt.scatter(x, y)
    caption = 'local clustering mean: {}, time: {}'.format(round(clustering_mean,2), elapased_time)
    plt.xlabel('local clustering')
    plt.ylabel('log(count)')
    plt.title('local clustering distribution plot\n' + caption)
    # plt.show()
    plt.savefig(name + '_clustering.png')
    plt.clf()
    # ### (c) shortest path distribution

    # In[101]:


    # %%time
    print('=' * 10, 'shortest path')
    start = time.time()


    row_nonzero, col_nonzero = S.nonzero()

    aja_list = defaultdict(list)

    for ind, i in enumerate(row_nonzero):
        aja_list[i].append(col_nonzero[ind])



    all_dist = defaultdict(dict)
    for i in tqdm(range(num_node)):
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

    all_dist_val_counts = Counter(all_dist_vals)

    all_dist_val_counts = np.array([[np.log(pl + 1), np.log(c)]     for (pl, c) in  all_dist_val_counts.items()])


    end = time.time()
    elapased_time = round(end - start, 4)

    x, y = all_dist_val_counts[:,0].reshape(-1,1), all_dist_val_counts[:,1].reshape(-1,1)

    plt.scatter(x, y)
    caption = 'shortest path length mean: {}, time: {}'.format(round(shortest_path_mean,2), elapased_time)
    plt.xlabel('log(shortest path length+1)')
    plt.ylabel('log(count)')
    plt.title('all pair shortest path length distribution plot\n' + caption)
    # plt.show()
    plt.savefig(name + '_path.png')
    plt.clf()

    # ### (d) number of connected components, the portion of nodes that are in the GCC

    # In[102]:

    print('=' * 10, 'connected components')
    # %%time
    start = time.time()

    all_comp = {}
    base_comp = 0
    for i in tqdm(range(num_node)):
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


    end = time.time()
    elapased_time = round(end - start, 4)

    print('number of connected component: ', len(component_counter))
    print('portion of nodes in giant connected component: ', round(max(component_counter.values()) / num_node, 2))
    print('time: {}'.format(elapased_time))

    os.system('echo {} >> {}'.format('number of connected component: {}'.format(component_counter), name + '_component.txt'))
    os.system('echo {} >> {}'.format('portion of nodes in giant connected component: {}'.format(
        round(max(component_counter.values()) / num_node, 2)), name + '_component.txt'))
    os.system('echo {} >> {}'.format('time: {}'.format(elapased_time), name + '_component.txt'))

    # ### (f) degree correlations (plot as scatter di vs dj , also report the overall correlation).

    # In[104]:

    print('=' * 10, 'degree correlation')
    # %%time
    start = time.time()

    degree_vector = csr_matrix(degrees.reshape(-1, 1))

    avg_neighbor_degree = np.nan_to_num(np.asarray(S.dot(degree_vector) / degree_vector)).reshape(-1)

    degree_correlation = defaultdict(list)
    for ind, i in enumerate(degrees):
        degree_correlation[int(i)].append(avg_neighbor_degree[ind])
    x = []
    y = []
    for i in degree_correlation:
        x.append(i)
        y.append(np.mean(degree_correlation[i]))
    x = np.array(x)
    y = np.array(y)

    [slope, bias] = least_squares(x.reshape(-1,1), y.reshape(-1,1)).reshape(-1).tolist()

    inferred_y = v_infer_y(x)

    end = time.time()
    elapased_time = round(end - start, 4)

    plt.scatter(x,y)
    plt.plot(x, inferred_y, c='orange')
    plt.xlabel('degree')
    plt.ylabel('average neighbour degree')
    caption = 'degree correlation coefficient: {}, time: {}'.format(round(slope, 2), elapased_time)
    # plt.show()
    plt.title('degree correlation plot\n' + caption)
    plt.savefig(name + '_degreecorr.png')
    plt.clf()

    # ### (g) degree-clustering coefficient relation (plot as scatter di vs ci)

    # In[105]:
    print('=' * 10, 'degree clustering')

    # %%time
    start = time.time()

    degree_clustering = defaultdict(list)
    for ind, i in enumerate(degrees):
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

    plt.scatter(x,y)
    plt.title('degree clustering plot\nused {} seconds'.format(elapased_time))
    plt.xlabel('degree')
    plt.ylabel('average local clustering')
    # plt.show()
    plt.savefig(name + '_degreecluster.png')
    plt.clf()
    # ### (e) eigenvalue distribution (compute the spectral gap)

    # In[103]:

    # %%time
    # print('=' * 10, 'eigen')
    # start = time.time()
    #
    # degree_diagonal = diags(degrees, 0)
    #
    # laplacian = degree_diagonal - S
    #
    # eigen_vals = linalg.eigsh(laplacian, k=num_node - 1, return_eigenvectors=False)
    #
    # eigen_vals = sorted(eigen_vals, reverse=True)
    #
    # pos_eigen = [i for i in eigen_vals if i > 0]
    # eigen_gap = pos_eigen[-1]
    #
    # end = time.time()
    # elapased_time = round(end - start, 4)
    #
    # sns.distplot(eigen_vals)
    # caption = 'eigengap: {}, time: {}'.format(round(eigen_gap, 2), elapased_time)
    # plt.title('laplacian eigenvalue distribution\n' + caption)
    # plt.xlabel('eigenvalue')
    # plt.ylabel('density')
    # # plt.show()
    # plt.savefig(name + '_eigen.png')
    # plt.clf()

def p2(S, num_node):

    m = S.nnz // 2 // num_node
    initial_num_node = max(4, m)
    t = num_node - initial_num_node

    S_ba = dok_matrix((num_node,num_node), dtype=float32)

    degree_ba = np.zeros(num_node)

    idx_ba = np.array(list(range(num_node)))

    for i in range(initial_num_node):
        degree_ba[i] += initial_num_node
        for j in range(initial_num_node):
            S_ba[i,j] = 1
            S_ba[j,i] = 1

    for i in tqdm(range(initial_num_node, initial_num_node + t)):
        ns = np.random.choice(idx_ba, size=m, p=degree_ba/np.sum(degree_ba), replace=False).tolist()
        for n in ns:
            degree_ba[n] = degree_ba[n] + 1
            S_ba[i,n] = 1
            S_ba[n,i] = 1
        degree_ba[i] += m


    S_ba = S_ba.tocsr()

    return S_ba, num_node


if __name__ == '__main__':
    data = 'collaboration'

    name_dir_dict = {
        'collaboration' : 'datasets/collaboration.edgelist.txt',
        'citation': 'datasets/citation.edgelist.txt',
        'actor': 'datasets/actor.edgelist.txt',
        'email': 'datasets/email.edgelist.txt',
        'internet': 'datasets/internet.edgelist.txt',
        'metabolic': 'datasets/metabolic.edgelist.txt',
        'phonecalls': 'datasets/phonecalls.edgelist.txt',
        'powergrid': 'datasets/powergrid.edgelist.txt',
        'protein': 'datasets/protein.edgelist.txt',
        'www': 'datasets/www.edgelist.txt'
    }

    S, num_node = load_data(name_dir_dict[data])
    p1(S, num_node, 'results/' + data)
    S_ba, num_node = p2(S, num_node)
    p1(S_ba, num_node, 'results/ba_' + data)


