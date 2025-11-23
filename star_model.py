import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import networkx as nx
from abc import ABC, abstractmethod



class MLModel(ABC):

    def __init__(self):
        self.stardata = pd.read_csv("hyg_v42_updated.csv", nrows=30000)

    @abstractmethod
    def trainmodel(self):
        pass



class Model1(MLModel):
    def __init__(self):
        super().__init__()
        self.parameters = ['ra', 'dec', 'mass', 'radius', 'temp', 'lum']


    def trainmodel(self):
        cleandata = self.stardata[self.stardata['proper'].isnull()]
        cleandata = cleandata.dropna(subset=self.parameters).reset_index(drop=True)

        stargraph = nx.Graph()
        for index, row in cleandata.iterrows():
            stargraph.add_node(index,
                               mass=row['mass'],
                               radius=row['radius'],
                               temp=row['temp'],
                               lum=row['lum'],
                               ra=row['ra'],
                               dec=row['dec'])

            k_neighbors = 5

            coords = cleandata[['ra', 'dec']].values

            nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            for i, neighbors in enumerate(indices):
                for j, neighbor_index in enumerate(neighbors[1:]):
                    if i < neighbor_index:
                        distance = distances[i][j+1]
                        stargraph.add_edge(i, neighbor_index, distance=distance)

            node_features = []
            node_targets = []
            node_indices = []

            for node in stargraph.nodes():
                features = [
                    stargraph.nodes[node]['mass'],
                    stargraph.nodes[node]['radius'],
                    stargraph.nodes[node]['temp']
                ]

                neighbors = list(stargraph.neighbors(node))
                if neighbors:
                    neighbor_masses = [stargraph.nodes[node]['mass'] for node in neighbors]
                    neighbor_radii = [stargraph.nodes[node]['radius'] for node in neighbors]
                    neighbor_temps = [stargraph.nodes[node]['temp'] for node in neighbors]

                    features.extend([
                        np.mean(neighbor_masses),
                        np.mean(neighbor_radii),
                        np.mean(neighbor_temps),
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])

                node_features.append(features)
                node_targets.append(stargraph.nodes[node]['lum'])
                node_indices.append(node)

            X = np.array(node_features)
            y = np.array(node_targets)

            feature_names = ['mass', 'radius', 'temp', 'neighbor_mass_avg', 'neighbor_radius_avg', 'neighbor_temp_avg']

            X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, node_indices, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            fig, ax = plt.subplots(1, 3, figsize=(20, 5))

            sample_nodes = list(stargraph.nodes())[:100]
            subgraph = stargraph.subgraph(sample_nodes)
            pos = {node: (stargraph[node]['ra'], stargraph[node]['dec']) for node in sample_nodes}




if __name__=='__main__':
    model1 = Model1()

    model1.trainmodel()

















# parameters = ['mass','radius','temp']
# target = 'lum'
#
# stardata_cleaned = stardata.dropna(subset=parameters + [target])
#
#
#
#
#
# network_percentage = 0.10
# data_sample = stardata.sample(frac=network_percentage, random_state=42)

