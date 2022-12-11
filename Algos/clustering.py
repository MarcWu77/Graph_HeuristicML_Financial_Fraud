import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 110; plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文宋体
plt.rcParams['axes.unicode_minus'] = False # 取消unicode编码

class DBscan():
    def __init__(self, data, max_radius, min_samples, sample_weight = None):
        """Density-based Spatial Clustering of Application with Noise

        Args:
            data (2D-array): X data with targeted features
            max_radius (list of flaot): maximum searching radius, among (0,1]
            min_samples (list of int): minmum numbers of points near to the core to be considered as next cores
        
        Returns:
            clusters (1D-array): cluster labels for each sample where -1 represents outliners.
        """
        self.data = data
        self.max_radius = max_radius
        self.min_samples = min_samples
        self.sample_weight = sample_weight
    
    def training(self,mark, visualized = False, quiet = True):
        """
        Args:
            mark (str): model name
            visualized (bool, optional): choose to present the tuning process or not. Defaults to False.

        Returns:
            clusters (1D-array): sequential clustering results
        """
        # FIND THE BEST PARAMETERS (eps & min_samples)
        self.best_radius, self.best_samples = self.find_best_parameters(mark, self.data, 
                                                                        visualized = visualized, quiet = quiet)
        # MODELING
        if quiet == False:
            print('>>> Start modeling...')
        self.model = DBSCAN(eps = self.best_radius,
                            min_samples = int(self.best_samples))
        self.model.fit(self.data, sample_weight = self.sample_weight)
        self.clusters = self.model.labels_
        self.num_outliners = np.sum(np.where(self.clusters == -1))
        if len(np.unique(self.clusters)) > 1:
            self.best_sil_coeff = silhouette_score(self.data, self.clusters, random_state=667)
            self.num_clusters = len(np.unique(self.clusters)-1) 
            if quiet == False:
                print(f'>> Silhoutte Coefficient of model {mark} = ', self.best_sil_coeff)
                print(f'>> The Number of Clusters is:', self.num_clusters)
        else:
            self.best_sil_coeff = -1
            self.num_clusters = 1
            print('>> All instances are outliners OR only one cluster detected;Please check if there is anything wrong~')
        return self.clusters
    
    def find_best_parameters(self, mark, data, visualized, quiet):
        # TUNING
        parameters = [[r,s] for r in self.max_radius
                            for s in self.min_samples]
        if quiet == False:
            print('>>> Start comparing parameters...')
        sil_score = []
        for i in parameters:
            clusters = DBSCAN(eps = i[0], min_samples = i[1]).fit(data, sample_weight=self.sample_weight).labels_
            # 防止在搜索半径过小的情况下，发生返回异常
            try: 
                coeff = silhouette_score(data, clusters)
            except:
                coeff = -1 
            sil_score.append([i[0], i[1], coeff])
        result_df = pd.DataFrame(np.array(sil_score), columns=['r','s','coeff']).sort_values(by='coeff',ascending=False)
        best_radius = result_df['r'].iloc[0]
        best_samples = result_df['s'].iloc[0]
        if quiet == False:
            print('>> Best radius:',best_radius,
                '; Best samples:', best_samples,
                '; Max Silhouette Coefficient:', result_df['coeff'].iloc[0])
        if visualized == True:
            # VISUALIZATION
            r_df = pd.DataFrame(np.array(sil_score), columns=['r','s','coeff']).sort_values(by='r')
            r_df.groupby('r')['coeff'].mean().plot(c='red', style='o-',
                                                xlabel='Max Searching Radius', ylabel='Mean Silhouette Coefficient', 
                                                title = 'Find the best MaxRadius')
            plt.show()
            plt.savefig(f'results/{mark}_dbscan_best_radius.png')
            s_df = pd.DataFrame(np.array(sil_score), columns=['r','s','coeff']).sort_values(by='s')
            s_df.groupby('s')['coeff'].mean().plot(c='blue', style='s-',
                                                xlabel='Min Neighbors of Core Points', ylabel='Mean Silhouette Coefficient', 
                                                title = 'Find the best MinSamples')
            plt.show()
            plt.savefig(f'results/{mark}_dbscan_best_samples.png')       
        return best_radius, best_samples
    
