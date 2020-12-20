import random
import bisect
import math
import os


def random_choice(values, weights=None , size = 1, replace = True):
    if weights is None:
        i = int(random.random() * len(values))
    else :
        total = 0
        cum_weights = []
        for w in weights:
            total += w
            cum_weights.append(total)
        x = random.random() * total
        i = bisect.bisect(cum_weights, x)
    if size <=1:
        if len(values)>i: return values[i]
        else: return None
    else:
        cval = [values[j] for j in range(len(values)) if replace or i!=j]
        if weights is None: cwei=None
        else: cwei = [weights[j] for j in range(len(weights)) if replace or i!=j]
        tmp= random_choice(cval, cwei, size-1, replace)
        if not isinstance(tmp,list): tmp = [tmp]
        tmp.append(values[i])
        return tmp


class Comms:
     def __init__(self, k):
         self.k = k
         self.groups = [[] for i in range(k)]
         self.memberships = {}

     def add(self, cluster_id, i, s = 1):
         if i not in  [m[0] for m in self.groups[cluster_id]]:
            self.groups[cluster_id].append((i,s))
            if i in self.memberships:
                self.memberships[i].append((cluster_id,s))
            else:
                self.memberships[i] =[(cluster_id,s)]
     def write_groups(self, path):
         with open(path, 'w') as f:
             for g in self.groups:
                 for i,s in g:
                    f.write(str(i) + ' ')
                 f.write('\n')


class Graph:
    def __init__(self,directed=False, weighted=False):
        self.n = 0
        self.counter = 0
        self.max_degree = 0
        self.directed = directed
        self.weighted = weighted
        self.edge_list = []
        self.edge_time = []
        self.deg = []
        self.neigh =  [[]]
        return

    def add_node(self):
        self.deg.append(0)
        self.neigh.append([])
        self.n+=1

    def weight(self, u, v):
        for i,w in self.neigh[u]:
            if i == v: return w
        return 0

    def is_neigh(self, u, v):
        for i,_ in self.neigh[u]:
            if i == v: return True
        return False

    def add_edge(self, u, v, w=1):
        if u==v: return
        if not self.weighted : w =1
        self.edge_list.append((u,v,w) if u<v or self.directed else  (v,u,w))
        self.edge_time.append(self.counter)
        self.counter +=1
        self.neigh[u].append((v,w))
        self.deg[v]+=w
        if self.deg[v]>self.max_degree: self.max_degree = self.deg[v]
        if not self.directed: 
            self.neigh[v].append((u,w))
            self.deg[u]+=w
            if  self.deg[u]>self.max_degree: self.max_degree = self.deg[u]
        return

    def to_nx(self):
        import networkx as nx
        G=nx.Graph()
        for u,v, w in self.edge_list:
            G.add_edge(u, v)
        return G

    def to_nx(self, C):
        import networkx as nx
        G=nx.Graph()
        for i in range(self.n):
            # This line works for networkx 1.10:
            # G.add_node(i, {'c':str(sorted(C.memberships[i]))})
            # This line works for networkx 2.2:
            G.add_node(i, c=str(sorted(C.memberships[i])))
            # G.add_node(i, {'c':int(C.memberships[i][0][0])})
        for i in range(len(self.edge_list)):
        # for u,v, w in self.edge_list:
            u,v, w = self.edge_list[i]
            G.add_edge(u, v, weight=w, capacity=self.edge_time[i])
            # G.add_edges_from(self.edge_list)
        return G

    def to_snap(self, C):
      import snap
      G = snap.TUNGraph.New()
      for i in range(self.n):
        G.AddNode(i)
      for i in range(len(self.edge_list)):
        u,v,w = self.edge_list[i]
        G.AddEdge(u, v)
      return G

    def to_ig(self):
        G=ig.Graph()
        G.add_edges(self.edge_list)
        return G


    def write_edgelist(self, path):
         with open(path, 'w') as f:
             for i,j,w in self.edge_list:
                 f.write(str(i) + '\t'+str(j) + '\n')


def Q(G, C):
    q = 0.0
    m = 2 * len(G.edge_list)
    for c in C.groups:
        for i,_ in c:
            for j,_ in c:
                q+= G.weight(i,j) - (G.deg[i]*G.deg[j]/(2*m))
    q /= 2*m
    return q


def common_neighbour(i, G, normalize=True):
    p = {}
    for k,wik in G.neigh[i]:
        for j,wjk in G.neigh[k]:
            if j in p: p[j]+=(wik * wjk)
            else: p[j]= (wik * wjk)
    if len(p)<=0 or not normalize: return p
    maxp = p[max(p, key = lambda i: p[i])]
    for j in p:  p[j] = p[j]*1.0 / maxp
    return p


def choose_community(i, G, C, alpha, beta, gamma, epsilon):
    mids =[k for  k,uik in C.memberships[i]]
    if random.random()< beta: #inside
        cids = mids
    else:
        cids = [j for j in range(len(C.groups)) if j not in mids] #:  cids.append(j)

    return cids[ int(random.random()*len(cids))] if len(cids)>0 else None


def degree_similarity(i, ids, G, gamma, normalize = True):
    p = [0]*len(ids)
    for ij,j in enumerate(ids):
        p[ij] =  (G.deg[j] -G.deg[i])**2
    if len(p)<=0 or not normalize: return p
    maxp = max(p)
    if maxp==0: return p
    p = [pi*1.0/maxp if gamma<0 else 1-pi*1.0/maxp for pi in p]
    return p


def combine (a,b,alpha,gamma):
    return (a**alpha) / ((b+1)**gamma)


def choose_node(i,c, G, C, alpha, beta, gamma, epsilon):
    ids = [j for j,_ in C.groups[c] if j !=i ]
    #   also remove nodes that are already connected from the candidate list
    for k,_ in G.neigh[i]:
        if k in ids: ids.remove(k)

    norma = False
    cn = common_neighbour(i, G, normalize=norma)
    trim_ids = [id for id in ids if id in cn]
    dd = degree_similarity(i, trim_ids, G, gamma, normalize=norma)

    if random.random()<epsilon or len(trim_ids)<=0:
        tmp = int(random.random() * len(ids))
        if tmp==0: return  None
        return ids[tmp], epsilon
    else:
        p = [0 for j in range(len(trim_ids))]
        for ind in range(len(trim_ids)):
            j = trim_ids[ind]
            p[ind] = (cn[j]**alpha )/ ((dd[ind]+1)** gamma)

        if(sum(p)==0): return  None
        tmp = random_choice(range(len(p)), p ) #, size=1, replace = False)
        # TODO add weights /direction/attributes
        if tmp is None: return  None
        return trim_ids[tmp], p[tmp]


def connect_neighbor(i, j, pj, c, b,  G, C, beta):
    if b<=0: return
    ids = [id for id,_ in C.groups[c]]
    for k,wjk in G.neigh[j]:
        if (random.random() <b and k!=i and (k in ids or random.random()>beta)):
            G.add_edge(i,k,wjk*pj)


def connect(i, b,  G, C, alpha, beta, gamma, epsilon):
    #Choose community
    c = choose_community(i, G, C, alpha, beta, gamma, epsilon)
    if c is None: return
    #Choose node within community
    tmp = choose_node(i, c, G, C, alpha, beta, gamma, epsilon)
    if tmp is None: return
    j, pj = tmp
    G.add_edge(i,j,pj)
    connect_neighbor(i, j, pj , c, b,  G, C, beta)


def select_node(G, method = 'uniform'):
    if method=='uniform':
        return int(random.random() * G.n) # uniform
    else:
        if method == 'older_less_active': p = [(i+1) for i in range(G.n)] # older less active
        elif method == 'younger_less_active' :  p = [G.n-i for i in range(G.n)] # younger less active
        else:  p = [1 for i in range(G.n)] # uniform
        return  random_choice(range(len(p)), p ) #, size=1, replace = False)[0]


def assign(i, C, e=1, r=1, q = 0.5):
    p = [e +len(c) for c in C.groups]
    id = random_choice(range(C.k),p )
    C.add(id, i)
    for j in range(1,r): #todo add strength for fuzzy
        if (random.random()<q):
              id = random_choice(range(C.k),p )
              C.add(id, i)
    return


def print_setting(n,m,k,alpha,beta,gamma, phi,r,q,epsilon,weighted,directed):
    print('n:',n,'m:', m ,'k:', k,'alpha:', alpha,'beta:', beta,'gamma:', gamma,)
    if phi!=default_FARZ_setting['phi']: print('phi:', phi, )
    if r!=default_FARZ_setting['r']: print('r:', r,)
    if q!=default_FARZ_setting['q']: print('pr:', q, )
    if epsilon!=default_FARZ_setting['epsilon']:'epsilon:', epsilon,
    print('weighted' if weighted else '', 'directed' if directed else '')

def realize(n, m,  k, b=0.0,  alpha=0.4, beta=0.5, gamma=0.1, phi=1, r=1, q = 0.5, epsilon = 0.0000001, weighted =False, directed=False):
    # print_setting(n,m,k,alpha,beta,gamma, phi,r,q,epsilon,weighted,directed)
    G =  Graph()
    C = Comms(k)
    for i in range(n):
    # if i%10==0: print('-- ',G.n, len(G.edge_list))
        G.add_node()
        assign(i, C, phi, r, q)
        connect(i,b, G, C, alpha, beta, gamma, epsilon)
        for e in range(1,m):
            j = select_node(G)
            connect(j, b, G, C, alpha, beta, gamma, epsilon)
    return G,C


def props():
    import plotNets as pltn
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus']=False
    graphs = []
    names = []
    params = default_FARZ_setting.copy()
    for alp, gam in [(0.5,0.5), (0.8,0.2), (.5,-0.5), (0.2,-0.8)]:
        params[ "alpha"]=alp
        params[ "gamma"]=gam
        print(str(params))
        G, C =realize(**params)
        print('n=',G.n,' e=', len(G.edge_list))
        print('Q=',Q(G,C))
        G = G.to_nx(C)
        pltn.printGraphStats(G)
        graphs.append(G.to_undirected())
        # name = 'F'+str(params)
        name = '$\\alpha ='+str(params[ "alpha"]) +',\; \\gamma='+str(params[ "gamma"])+"$"
        names.append(name)
        nx.write_gml(graphs[-1], "farz-"+str(params[ "alpha"])+str(params[ "gamma"])+'.gml')

    pltn.plot_dists(graphs,names)

def write_to_file(G,C,path, name,format,params):
    if not os.path.exists(path+'/'): os.makedirs(path+'/')
    print('n=',G.n,' e=', len(G.edge_list), 'generated, writing to ', path+'/'+name, ' in', format)
    if format == 'gml':
        import networkx as nx
        G = G.to_nx(C)
        if not params['directed']: G = G.to_undirected()
        nx.write_gml(G, path+'/'+name+'.gml')
        return G
    if format == 'list':
        G.write_edgelist(path+'/'+name+'.dat')
        C.write_groups( path+'/'+name+'.lgt')
        return G


default_ranges = {'beta':(0.5,1,0.05), 'k':(2,50,5), 'm':(2,11,1) , 'phi':(1,100,10), 'r':(1,10,1), 'q':(0.0,1,0.1)}
default_FARZ_setting = {"n":1000, "k":4, "m":5, "alpha":0.5,"gamma":0.5, "beta":.8, "phi":1, "o":1, 'q':0.5,  "b":0.0, "epsilon":0.0000001, 'directed':False, 'weighted':False}
default_batch_setting= {'vari':None, 'arange':None, 'repeat':1, 'path':'.', 'net_name':'network', 'format':'gml', 'farz_params':None}
supported_formats = ['gml','list']

def generate_farz( vari =None, arange =None, repeat = 1, path ='.', net_name = 'network',format ='gml', farz_params= default_FARZ_setting.copy()):
    
    def get_range(s,e,i):
        res =[]
        while s<=e + 1e-6:
            res.append(s)
            s +=i
        return res

    if vari is None:
        for r in range(repeat):
            G, C =realize(**farz_params)
            import snap as snap
            G = G.to_snap(C)
            name = net_name+(str(r+1) if repeat>1 else '')
        return G

    if arange ==None:
        arange = default_ranges[vari]

    for i,var in enumerate(get_range(arange[0],arange[1],arange[2])):
        for r in range(repeat):
            farz_params[vari] = var
            print('s',i+1, r+1, str(farz_params))
            G, C =realize(**farz_params)
            name = 'S'+str(i+1)+'-'+net_name+ (str(r+1) if repeat>1 else '')
            write_to_file(G,C,path,name,format,farz_params)
