import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_ppr import personalized_page_rank
from node2vec import Node2Vec
from Algos.clustering import DBscan
import matplotlib.pyplot as plt

def filtering(df, condition_list):
      """Data Filtering

      Args:
          df (pd.DataFrame): raw data
          condition_list (list of tuples): conditions list in pandas format

      Returns:
          df: modify raw data directly
      """
      print('>>> 开始 Data Filtering...')
      print('>> 原数据量 = ',len(df))
      for id,condition in enumerate(condition_list):
            try:
                  df = df[condition]
                  print(f'> Condition {id+1} 过滤完毕...')
            except:
                  print('> 筛选语法存在问题,请检查后重试！')
      print('>> 筛选后数据量 = ',len(df))
      return df

def PR_ranking(df, from_name, to_name, verbose = False):
    """ Pange Rank 排序函数

    Args:
        df(pd.DataFrame)：样本数据框\n
    Returns:
        PR_index(nd.array)：排好序的数据
    """
    print('>>> 开始 Personalised Page Ranking...')
    # 对所有入度和出度卡id拉通进行可迭代重编码
    total_nodes = list(set(df[from_name].to_list() + df[to_name].to_list()))
    marker_dict = {i[1]:i[0] for i in enumerate(total_nodes)}
    
    df[from_name] = df[from_name].apply(lambda x:marker_dict[x])
    df[to_name] = df[to_name].apply(lambda x:marker_dict[x])
    # 表征边的坐标并且转换成pytorch张量
    edge_list = [(i,j) for i,j in zip(df[from_name], df[to_name])]
    edge_index = torch.as_tensor(data = edge_list).t()
    if verbose:
        print('>> 网络边坐标构建完毕...')
        print('>> 独立节点总数 =', len(total_nodes))
        print('>> 有效边总数 =', len(edge_list))
    # Page Rank 排序，结果按行降序排列
    
    PR_array = np.array(personalized_page_rank(edge_index = edge_index, 
                                               add_identity = False))
    PR_index = PR_array.argsort(axis=1)[:,::-1] # 把结果PR邻接矩阵按行降序排列，返回其索引
    print('>>> PR值降序排列完毕...')
    np.savetxt('results/01_PageRank_indices.txt',PR_index,fmt='%s') 
    return PR_index, marker_dict

def K_ranking(df, PR_index, from_name, to_name, weight_name, from_target_name, to_target_name, 
              seed_strategy = 'both', K = 2):
    """ Top K 筛选函数

    Args:
        df (pd.DataFrame): 原数据框
        PR_index (nd.array）：排好序的数据\n
        seed_ids（list, optional）：默认不提供种子节点列表；如果不提供，则设置seed_strategy参数从关系中推断\n
        seed_strategy (str, optional)：种子节点选择策略，'both'代表出度入度节点都被判定为种子，'out'代表只有出度节点，'in'代表只有入度节点. 默认为'both'.\n
        K (int, optional): 选择需要扩展的最相似的样本个数. Defaults to 2.\n
    Returns:
        expanded_ids (list)：扩展后ids\n
        target_data (pd.DataFrame)：扩展筛选后的样本数据框\n
    """
    if seed_strategy == 'both':
        seed_ids = list(set(df[df[from_target_name] == 1][from_name].tolist() + df[df[to_target_name] == 1][to_name].tolist()))
    elif seed_strategy == 'from':
        seed_ids = list(set(df[df[from_target_name] == 1][from_name].tolist()))
    elif seed_strategy == 'to':
        seed_ids = list(set(df[df[to_target_name] == 1][to_name].tolist()))
    # 筛选前K个与种子节点最相似的可疑卡并去重
    suspects_list = []
    for seed_id in seed_ids:
        for suspects_index in PR_index[seed_id, :K]: # PPR算法呈现结果按照ID升序
            suspects_list.append(suspects_index)
    # 将种子卡id与可疑卡id取并集，生成样本扩展集合
    expanded_ids = list(set(seed_ids + suspects_list))
    common_cards = [i for i in seed_ids if i in suspects_list]
    # 根据样本扩展集合筛选出相应数据
    if seed_strategy == 'both':
        target_data = df[(df[from_name].isin(expanded_ids)) | (df[to_name].isin(expanded_ids))][[from_name, to_name, weight_name, from_target_name, to_target_name,]]
    elif seed_strategy == 'from':
        target_data = df[df[from_name].isin(expanded_ids)][[from_name, to_name, weight_name, from_target_name, to_target_name,]]
    elif seed_strategy == 'to':
        target_data = df[df[to_name].isin(expanded_ids)][[from_name, to_name, weight_name, from_target_name, to_target_name,]]
    return seed_ids, expanded_ids, target_data

class node2vec_for_clustering():
    def __init__(self, target_data, from_name, to_name, weight_name = None, worker = 1, seed = 7, verbose = False):
        """ node2vec特征抽取方法实现DBSCAN聚类

        Args:
            target_data (pd.DataFrame): (筛选/排序后)的样本数据框
            weight_name (str, optional): 权重列名，不指定的话即不加权. Defaults to None.
            worker (bool, optional): node2vec训练时的平行处理器个数(根据电脑最大核数选择). Defaults to False.
            seed (int, optional): 随机游走的随机数.
            quiet(bool, optional): 是否打印中间过程，默认为True.
        """
        self.target_data = target_data
        self.from_name = from_name
        self.to_name = to_name
        self.weight_name = weight_name
        self.worker = worker
        self.seed = seed
        self.verbose = verbose
    
    def training(self):
        self.graph = self.graph_preprocess()
        self.num_nodes = len(self.graph.nodes)
        self.num_edges = len(self.graph.edges)
        self.density = nx.density(self.graph) # 后期以density来确定参数基线
        if self.verbose:
            print('>>> 该批次网络节点总数为：',self.num_nodes, '; 边总数为：',self.num_edges, '; 网络密度为：', np.round(self.density,4))
        self.best_labels = self.tuning(self.graph, self.num_nodes, self.density) # node2vec调参 + 聚类
        mapping_df_orig, mapping_df_dest = self.mapping_labels(self.graph, self.best_labels, self.from_name, self.to_name)
        return self.target_data.merge(mapping_df_orig).merge(mapping_df_dest) # 拼表
    
    def mapping_labels(slef, graph, best_labels, from_name, to_name):
        node_mapping_dict = {node[0]:node[1] for node in enumerate(graph.nodes)}
        mapping_array = np.array([[node_mapping_dict[i], best_labels[i]] for i in range(len(graph.nodes))])
        mapping_df_orig = pd.DataFrame(mapping_array, 
                                       columns=[from_name,'cluster_orignode']) # 按出度节点进行映射
        mapping_df_dest = pd.DataFrame(mapping_array, 
                                       columns=[to_name,'cluster_destnode']) # 按入度节点进行映射
        return mapping_df_orig, mapping_df_dest
    
    def tuning(self, graph, num_node, graph_density):  
        if graph_density >= 0.6: # 高密度
            walk_length = 150; num_walks = 20
            p = 0.5; q = 1 # 高密度网络总体策略是让游走更保守、更注意local
            l_range = range(int(walk_length/2), 176, 25)
            n_range = range(int(num_walks/2), 31, 5)
            p_range = np.linspace(0.1,2.5,5)
            q_range = np.linspace(0.2,5,5)
        elif (graph_density >= 0.3) and (graph_density < 0.6): # 中等密度
            walk_length = 80; num_walks = 10
            p = 1; p = 1 # 中等密度的网络一开始不需要做游走博弈
            l_range = range(int(walk_length/2), 113, 18)
            n_range = range(int(num_walks/2), 26, 5)
            p_range = np.linspace(0.2,5,5)
            q_range = np.linspace(0.2,5,5)      
        else: # 稀疏网络
            walk_length = 40
            num_walks = 10
            p = 1; q = 0.5 # 低密度网络总体策略是让游走更激进、更注意global
            l_range = range(int(walk_length/2), 81, 15)
            n_range = range(int(num_walks/2), 26, 5)
            p_range = np.linspace(0.2,5,5)
            q_range = np.linspace(0.1,2.5,5)
                
        # print('=='*15,'第一轮调参','=='*15)
        # print('>>> 开始试验特征维数...')
        choose_d = {}
        for d in np.exp2(range(2,10)).astype(int): # 特征维数范围介于[4, 512]
            emb_model  = Node2Vec(graph, dimensions = d,
                                  walk_length = walk_length, num_walks = num_walks,
                                  p = p, q = q, weight_key = 'weight', quiet = True,
                                  workers = self.worker, seed = self.seed)
            emb_model = emb_model.fit()
            emb_results = np.array([emb_model.wv[i] for i in range(num_node)])
            clu_model = DBscan(emb_results, max_radius=np.linspace(0.05,0.9,5), min_samples=range(3,11))
            labels = clu_model.training(mark='node2vec_tuning',
                                        visualized = False, quiet = True)
            choose_d[d] = clu_model.best_sil_coeff
        self.best_d, best_score = sorted([(k,v) for k,v in choose_d.items()], key = lambda x:x[1], reverse = True)[0]
        # print('>> 最佳特征维数 =', self.best_d, '; 此时DBSCAN聚类轮廓系数 =', best_score)
                                        
        # print('=='*15,'第二轮调参','=='*15)
        # print('>>> 开始试验游走范围参数...')
        choose_l_n = {}
        for parameter in [(l,n) for l in l_range for n in n_range]:
            emb_model  = Node2Vec(graph, dimensions = self.best_d, # 更新特征维数
                                walk_length = parameter[0], num_walks = parameter[1],
                                p = p, q = q, weight_key = 'weight', quiet = True,
                                workers = self.worker, seed = self.seed)
            emb_model = emb_model.fit()
            emb_results = np.array([emb_model.wv[i] for i in range(num_node)])
            clu_model = DBscan(emb_results, max_radius=np.linspace(0.05,0.9,5), min_samples=range(3,11))
            labels = clu_model.training(mark='node2vec_tuning', 
                                        visualized = False, quiet = True)
            choose_l_n[(parameter[0],parameter[1])] = clu_model.best_sil_coeff
        best_l_n, best_score = sorted([(k,v) for k,v in choose_l_n.items()], key = lambda x:x[1], reverse = True)[0]
        self.best_walk_length = best_l_n[0]
        self.best_num_walks = best_l_n[1]
        # print('>> 最佳游走长度 =', self.best_walk_length,'; 最佳节点遍历次数 =', self.best_num_walks,'; 此时DBSCAN聚类轮廓系数 =', best_score)
        
        # print('=='*15,'第三轮调参','=='*15)
        # print('>>> 开始试验游走策略参数...')
        choose_p_q = {}
        for parameter in [(p,q) for p in p_range for q in q_range]:
            emb_model  = Node2Vec(graph, dimensions = self.best_d, 
                                  walk_length = self.best_walk_length, num_walks = self.best_num_walks, # 更新游走范围参数
                                  p = parameter[0], q = parameter[1], weight_key = 'weight', quiet = True,
                                  workers = self.worker, seed = self.seed)
            emb_model = emb_model.fit()
            emb_results = np.array([emb_model.wv[i] for i in range(num_node)])
            clu_model = DBscan(emb_results, max_radius=np.linspace(0.05,0.9,5), min_samples=range(3,11))
            labels = clu_model.training(mark='node2vec_tuning',
                                        visualized = False, quiet = True)
            choose_p_q[(parameter[0],parameter[1])] = [clu_model.best_sil_coeff, labels] # 最后一轮调参把轮廓系数和聚类标签都收集起来
        # np.savetxt('results/03_node2vec_embedding.txt', emb_results, fmt = '%s')
        
        best_p_q, best_results = sorted([(k,v) for k,v in choose_p_q.items()], key = lambda x:x[1][0], reverse = True)[0] # 按照轮廓系数降序排列取最高
        self.best_p = best_p_q[0]
        self.best_q = best_p_q[1]
        self.best_score = best_results[0]
        best_labels = best_results[1]
        if self.verbose:
            print('>>','本轮最佳参数报告：')
            print('特征维数 D =',self.best_d, '; 游走距离 l =',self.best_walk_length, '; 游走步数 n =',self.best_num_walks,
                  '回流系数 p =',self.best_p, '; 出圈系数 q =',self.best_q,'; 轮廓系数 s =', self.best_score)
            print('=='*15)
        # parameter_result = [['dimension',self.best_d], ['walk_length',self.best_walk_length], 
        #                     ['num_walks',self.best_num_walks],['p',self.best_p],['q',self.best_q]]
        # np.savetxt('results/04_node2vec_clustering_parameters.txt', parameter_result, delimiter = '=', fmt = '%s')
        # print('>>> 最佳参数结果已保存.')
        
        return best_labels

    def graph_preprocess(self):
        G = nx.Graph()
        if self.weight_name:
            # 对网络进行边坐标表征
            target_edge_list = [(f,t,w) for f,t,w in zip(self.target_data[self.from_name], self.target_data[self.to_name], self.target_data[self.weight_name])]
            # print('>>> 扩展样本集边坐标转换完毕...')
            G.add_weighted_edges_from(target_edge_list)  
            # print('>>> 加权无向网络构建完毕...')
        else:
            target_edge_list = [(f,t) for f,t in zip(self.target_data[self.from_name], self.target_data[self.to_name])]
            # print('>>> 扩展样本集边坐标转换完毕...')
            G.add_edges_from(target_edge_list)
            # print('>>> 无权无向网络构建完毕...')
        return G

class cluster_describtion():
    def __init__(self, labeled_df, from_name, to_name, K = 1, weight_name = None, visualize_whole = True):
        """聚类簇分组描述

        Args:
            labeled_df (pd.DataFrame): 贴好标签的数据框
            from_name (str): 出度列名
            to_name (str): 入度列名
            K (int): Rranking K, default as 1
            weight_name (str, optional): 选择是否指定加权列. Defaults to None.
            visualize_whole (bool, optional): 选择是否先对所有簇可视化. Default to True.
        """
        self.labeled_df = labeled_df
        self.from_name = from_name
        self.to_name = to_name
        self.K = K
        self.weight_name = weight_name
        if self.weight_name:
            # 将权重列先取整数对数、再继续标准化到[0,1](min-max缩放方法)
            labeled_df[weight_name] = labeled_df[weight_name].apply(np.log10)
            target_max = labeled_df[weight_name].max()
            target_min = labeled_df[weight_name].min()
            labeled_df[weight_name] = labeled_df[weight_name].apply(lambda x:np.round((x - target_min) / (target_max - target_min), 4))
            if visualize_whole == True:
                target_edge_list = [(f,t,w) for f,t,w in zip(labeled_df[from_name], labeled_df[to_name], labeled_df[weight_name])]
                DG = nx.DiGraph()
                DG.add_weighted_edges_from(target_edge_list)
                edgewidth = []; deg = []
                for f,t,w in DG.edges(data=True):
                    edgewidth.append(DG.get_edge_data(f,t)['weight']*2)
                    deg.append(DG.degree())
                    options = {'node_color':'lightblue',
                               'node_size':200,
                               'edge_color':'black',
                               'width':edgewidth}
                plt.figure(figsize=(10,6), dpi = 300)
                nx.draw(DG, **options,  
                        pos = nx.fruchterman_reingold_layout(DG, k = np.sqrt(3 / DG.number_of_nodes())), 
                        with_labels = True, 
                        font_size = 6.5)
                plt.savefig(f'results/Overall_Graph.png')
                plt.show()
        else:
            if visualize_whole == True:
                target_edge_list = [(f,t) for f,t in zip(labeled_df[from_name], labeled_df[to_name])]
                DG = nx.DiGraph()
                DG.add_edges_from(target_edge_list)
                options = {'node_color':'lightblue',
                           'node_size':200,
                           'edge_color':'black',
                           'width':2}
                plt.figure(figsize=(10,6), dpi = 300)
                nx.draw(DG, **options,  
                        pos = nx.fruchterman_reingold_layout(DG, k = np.sqrt(self.K / DG.number_of_nodes())), 
                        with_labels = True, 
                        font_size = 6.5)
                plt.savefig(f'results/Overall_Graph.png')
                plt.show()
        
    def analyze_each_group(self):
        # 为每一个团簇生成数据框 (规则：出度或入度属于某个团簇)
        self.clustered_df_list = [self.labeled_df[[self.from_name, self.to_name, self.weight_name,'cluster_orignode','cluster_destnode']][(self.labeled_df ['cluster_orignode'] == C) | (self.labeled_df ['cluster_destnode'] == C)]
                                  for C in np.unique(self.labeled_df ['cluster_orignode'].tolist() + self.labeled_df ['cluster_destnode'].tolist())] 
        print(f'>>> 按照出度节点 OR 入度节点属于某个簇的逻辑划分，共有{len(self.clustered_df_list)}类团簇...')
        # 处理非离群点
        group_metrics = [] # 存储网络评估指标
        cluster_ID = 0 # 确保团簇的标号从0开始
        for clustered_df in self.clustered_df_list:
            if self.weight_name:
                target_edge_list = [(f,t,w) for f,t,w in zip(clustered_df[self.from_name], # 为每个有权子图创建edge_list
                                                             clustered_df[self.to_name], 
                                                             clustered_df[self.weight_name])]
                DG = nx.DiGraph()
                DG.add_weighted_edges_from(target_edge_list)
            else:
                target_edge_list = [(f,t) for f,t in zip(clustered_df[self.from_name], # 为每个无权子图创建edge_list
                                                            clustered_df[self.to_name])]
                DG = nx.DiGraph()
                DG.add_edges_from(target_edge_list)               
            # COMPUTE METRICS
            group_metric = self.calculate_network_metrics(cluster_ID, DG, clustered_df)
            group_metrics.append(group_metric)
            # VISULIZATION EACH GROUP
            if len(DG.nodes) >= 100:
                node_size = 100; font_size = 5.5
            else:
                node_size = 300; font_size = 12
            if len(clustered_df[(clustered_df['cluster_orignode'] == -1) & (clustered_df['cluster_destnode'] == -1)]) > 0: # 出度和入度都是离群点才被判定为“离群网络”
                self.visualize_each_group(f'outliners_{cluster_ID}', # 对离群网络作标识
                                          DG, node_color = 'blue',
                                          node_size = node_size, edge_color = 'black',
                                          font_size = font_size, with_label = True)
            else:
                self.visualize_each_group(cluster_ID,
                                          DG, node_color = 'lightblue',
                                          node_size = node_size, edge_color = 'black',
                                          font_size = font_size, with_label = True)
            cluster_ID += 1

        return pd.DataFrame(group_metrics, columns = ['cluster_ID','SUS_LIST','density',
                                                      'avg_transitivity','reciprocity','avg_closeness',
                                                      'num_sus_node','avg_amount','avg_degree','avg_indegree','avg_outdegree'])

    def calculate_network_metrics(self, cluster_ID, graph, clustered_df):
        # COMMUNITY-level's Closeness
        density = nx.density(graph)
        avg_transitivity = np.mean([i for i in nx.clustering(graph).values()])
        reciprocity = np.mean(nx.reciprocity(graph))
        avg_closeness = np.mean([i for i in nx.closeness_centrality(graph).values()])
        # COMMUNITY-level's Scale
        num_sus_node = graph.number_of_nodes() # 最后再标准化到[0,1]
        avg_amount = np.mean(clustered_df[self.weight_name]) # 在初始化的时候已经标准化到[0,1]了
        avg_degree = np.mean([i for i in nx.degree_centrality(graph).values()])
        avg_indegree = np.mean([i for i in nx.in_degree_centrality(graph).values()])
        avg_outdegree = np.mean([i for i in nx.out_degree_centrality(graph).values()])
        SUS_LIST = list(graph.nodes)
        return [cluster_ID,SUS_LIST,density,avg_transitivity,reciprocity,avg_closeness,
                num_sus_node,avg_amount,avg_degree,avg_indegree,avg_outdegree]

    def visualize_each_group(self, cluster_ID, graph, node_color, node_size, edge_color, font_size, with_label = True):
        edgewidth = []
        for f,t,w in graph.edges(data=True):
            edgewidth.append(graph.get_edge_data(f,t)['weight']*2)
        options = {'node_color':node_color,
                   'node_size':node_size,
                   'edge_color':edge_color,
                   'width':edgewidth}
        if len(graph.nodes) >= 100:
            plt.figure(figsize=(int(len(graph.nodes)/8), int(len(graph.nodes)/8)))
        elif (len(graph.nodes) >= 30) & (len(graph.nodes) < 100):
            plt.figure(figsize=(10,10))
        elif len(graph.nodes) < 30:
            plt.figure(figsize=(7,7))
        nx.draw(graph, 
                **options, 
                pos = nx.spring_layout(graph), 
                with_labels = with_label, 
                font_size = font_size)
        plt.savefig(f'results/Cluster_{cluster_ID}_graph.png')
