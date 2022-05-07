import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

class DBSCAN:
    def __init__(self, df, epsilon, min_points):
        self.arr = df.sample(frac=1).to_numpy() 
        self.epsilon = epsilon
        self.min_points = min_points
        
        self.n = self.arr.shape[0]
        self.unvisited_points = np.array(list(range(self.n)))
        self.classifications = np.array(list(np.repeat(0, repeats=self.n)))
        self.num_classified_points = 0

        self.current_cluster = 0 # 0 is no cluster, -1 is noise
        self.max_cluster = 0

    ########### utils ##########
    def get_distance(self, p, distance='euclidean'):
        print(self.arr)
        point = self.arr[p].reshape(1, -1)
        print(p)
        print([True if i != p else False for i in range(self.n)])
        other_points = self.arr[[True if i != p else False for i in range(self.n)], :]
        print(point)
        print(other_points)

        if distance == 'euclidean':
            return euclidean_distances(point, other_points) 

    def get_next_point(self, dist):
        """
        Get min distance point in unvisited points.
        """
        print(dist)
        for i in np.argsort(dist).flatten():
            print(np.argsort(dist).flatten())
            print('HERE!@!!!')
            print(i)
            if i in self.unvisited_points:
                return i


    def cluster_logic(self, p, dist):
        """
        Take point and distances. Assign cluster accordingly.
        """
        points_in_cluster = dist[dist <= self.epsilon]
        n_points_in_cluster = len(points_in_cluster)

        if np.min(dist) > self.epsilon: 
            # assign noise
            self.current_cluster = 0
            return -1
        else: 
            has_cluster = self.current_cluster != 0                # current cluster is not unassigned 
            is_core_point = n_points_in_cluster >= self.min_points # can start new cluster

            if has_cluster: 
                # assign current cluster
                return self.current_cluster
            else:
                if is_core_point: 
                    # create new cluster
                    self.max_cluster += 1
                    self.current_cluster = self.max_cluster
                    return self.current_cluster
                else:
                    # assign noise
                    self.current_cluster = 0
                    return self.current_cluster


    ############## run ############
    def run(self):
        p = 0
        while len(self.unvisited_points) > 0:
            print(f'p: {p}')
            # 0. Assign point as visisted
            self.unvisited_points = self.unvisited_points[list(np.where(self.unvisited_points==p, False, True))]

            # 1. Get distance matrix for point p 
            dist = self.get_distance(p)

            # 2. Assign point to cluster
            self.classifications[p] = self.cluster_logic(p, dist)
            self.num_classified_points += 1
            print('classes: ')
            print(self.classifications)

            # 3. Go to next point
            if len(self.unvisited_points) > 0:
                p = self.get_next_point(dist)


df = pd.DataFrame(dict(x=[1,2,3,4,5], y=[5,4,3,2,1]))
dbscan = DBSCAN(df, 0.5, 2)
dbscan.run()




