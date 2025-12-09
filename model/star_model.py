#Necessary imports for all models, polymorphism, and graphs
import warnings
import pandas as pd
import numpy as np
from pandas.errors import SettingWithCopyWarning
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import networkx as nx
from abc import ABC, abstractmethod
from pathlib import Path

# Get the script's directory
script_dir = Path(__file__).parent.parent

# Uses the script directory to access the data csv file
file_path = script_dir / 'data' / 'hyg_v42_updated.csv'

# Uses the script directory to access the graphs created by the model
graph_path = script_dir / 'view'


#Abstract base class for ML Models
class MLModel(ABC):
    #Constructor includes filepath to the csv file to be used along with the amount of rows
    #that will be used from the file
    def __init__(self, csv_path=file_path, nrows=None):

        #If there is a given value of rows, read the file up until that row
        if nrows:
            self.stardata = pd.read_csv(csv_path, nrows=nrows)
        else: #Otherwise, read the entire file
            self.stardata = pd.read_csv(csv_path)

        self.model = None #Model name
        self.scaler = None #Scaler to be used for data normalization
        self.X_train = None #Training set
        self.X_test = None #Testing set
        self.y_train = None #Training set
        self.y_test = None #Testing set
        self.y_pred = None #Predictions
        self.feature_names = [] #Features to be analyzed and learned from by the model

    @abstractmethod
    def set_data(self):
        pass
        # Each model defines this


    @abstractmethod
    def train_model(self):
        pass
        # Each model defines this

    def evaluate_model(self):
        if self.y_pred is None: #If no predictions have been made
            raise ValueError("Model has not been trained.")

        r2 = r2_score(self.y_test, self.y_pred) #How closely the model predictions align with the data
        rmse = mean_squared_error(self.y_test, self.y_pred) #Average squared difference between predicted and true values
        mae = mean_absolute_error(self.y_test, self.y_pred) #Average absolute difference between predicted and true values

        # Show results
        print(f"\n{'=' * 60}")
        print(f"Model Evaluation Results:")
        print(f"{'=' * 60}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"{'=' * 60}")

        # Returns a dictionary with the values for further use
        return {'r2': r2, 'rmse': rmse, 'mae': mae}

    def predict(self, input_data):

        #Predict luminosity for a single star or group of stars.

        # Parameters:
        #     input_data : dictionary or pd.DataFrame or np.array
        #     Single row or multiple rows of star data

        #Predictions can only be done if the model has been trained beforehand to make them
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")

        #Predictions can only be carried out if the data has been normalized.
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call train_model() first.")

        # Convert dictionary to array
        if isinstance(input_data, dict):
            input_array = np.array([[input_data[feature] for feature in self.feature_names]])
        # Convert DataFrame to array
        elif isinstance(input_data, pd.DataFrame):
            input_array = input_data[self.feature_names].values
        # Assume it's already an array if not a dictionary or df
        else:
            input_array = np.array(input_data).reshape(1, -1) if len(np.array(input_data).shape) == 1 else input_data

        # Scale the input using the same scaler from training (data normalization)
        input_scaled = self.scaler.transform(input_array)

        # Make prediction
        prediction = self.model.predict(input_scaled)

        # Returns:
        #   float or np.array: Predicted luminosity value(s)
        return prediction[0] if len(prediction) == 1 else prediction


    @abstractmethod
    def visualize_model(self, filename):
        pass
        #Each model will define this

    def get_featimp(self):
        # Used to hold a feature and it's importance
        feature_dict = None

        #Model must be trained for feature importance to be measured
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # If the used model has feature importances, pair feature names with importance value
        # and store it as a dictionary that will then be returned. Otherwise, leave it empty.
        if hasattr(self.model, 'feature_importances_'):
            print(f"\nFeature Importance:")
            for name, importance in zip(self.feature_names, self.model.feature_importances_):
                print(f"  {name}: {importance:.4f}")
            feature_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            print("Model does not support feature importance.")
            feature_dict = None

        return feature_dict

'''================================================================================================================================================================'''
'''================================================================================================================================================================'''
'''================================================================================================================================================================'''

# Traditional, straightforward ML Model. Used in the main project due to having the highest R² value.
# This means that it possesses additional methods for this purpose relative to the other two models.
class TradModel(MLModel):

    def __init__(self, csv_path=file_path, nrows=None):
        super().__init__(csv_path, nrows) #Obtains the csv filepath and the rows to be read from the base class
        self.required_features = ['mass', 'radius', 'temp', 'dist', 'absmag'] #Features to be used in prediction
        self.target = 'lum' #Target to be predicted
        self.feature_names = self.required_features.copy() #Feature names
        # (A copy of the features so the features themselves can be accessed without the names being affected.)


    #Sets the data to be used
    def set_data(self):
        print(f"\nPreparing data for Traditional ML Model...") #Context
        # Cleans the required features and the target so only fully valid stars can be used in the training process.
        cleandata = self.stardata.dropna(subset=self.required_features + [self.target])

        print(f"  Stars after filtering: {len(cleandata)}") #Context

        X = cleandata[self.required_features].values # Feature values
        y = cleandata[self.target].values # Target values

        # Splits the data into features and targets to be trained on and to be tested on.
        # Uses 60/40 split for training and testing respectively
        # Randomness seed is used to randomly pick which values are trained with and which are tested upon
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # Normalizes the features to be analyzed
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        #Context
        print(f"  Training set: {len(self.X_train)} stars")
        print(f"  Testing set: {len(self.X_test)} stars")

    #Trains the model
    def train_model(self):
        if self.X_train is None:
            self.set_data() # Ensures the data is set

        print(f"\nTraining Traditional Random Forest Model...") #Context
        # Uses the RandomForestRegressor model, 100 decision trees, random seed to ensure same results in different runs,
        # uses all available CPU cores for speed (n_jobs=-1)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        # Trains the model on the data
        self.model.fit(self.X_train, self.y_train)

        # Makes predictions based on the acquired training data
        self.y_pred = self.model.predict(self.X_test)

    # Method exclusive to the TradModel due to its use in the main project.
    def input_predict(self, star_id):
        # Reads the stardata
        stardata = pd.read_csv(file_path)

        #Takes the ID of a given star as an input and turns it into a string
        star_row = stardata[stardata['id'].astype(str) == str(star_id)]

        #If there is no ID, says the star has not been found
        if star_row.empty:
            raise ValueError(f"Star with ID {star_id} not found")

        # Saves the chosen star as the one that first appears when a star is searched for.
        # This is done in case ID digits overlap (Ex. 1246 & 5612467)
        star = star_row.iloc[0]

        # Selects the following data from the star to be analyzed (It matches with the required features of the model)
        # as a dictionary
        star_data = {
            'mass': star['mass'],
            'radius': star['radius'],
            'temp': star['temp'],
            'dist': star['dist'],
            'absmag': star['absmag']
        }

        # Predicts the star using the class' predict method
        return self.predict(star_data)

    # Method exclusive to the TradModel due to its use in the main project.
    # Classifies star based on temperature and luminosity (In theory should match the Hertzsprung-Russell diagram)
    def stellar_classification(self, temp, lum):

        # Defaults the classification to Unclassified
        classification = "Unclassified (Data incomplete - Star may be classified in reality)"

        # Each temp-lum range corresponds to a particular classification. This is used to designate a star.
        if 2500 <= temp < 3500:
            if 0.0001 < lum < 0.08:
                classification = "Main Sequence (M)"
        elif 3500 <= temp < 5000:
            if 0.08 < lum < 0.6:
                classification = "Main Sequence (K)"
        elif 5000 <= temp < 6000:
            if 0.6 < lum < 1.5:
                classification = "Main Sequence (G)"
        elif 6000 <= temp < 7500:
            if 1.5 < lum < 5:
                classification = "Main Sequence (F)"
        elif 7500 <= temp < 10000:
            if 5 < lum < 25:
                classification = "Main Sequence (A)"
        elif 10000 <= temp < 30000:
            if 25 < lum < 10000:
                classification = "Main Sequence (B)"
        elif temp >= 30000:
            if lum > 10000:
                classification = "Main Sequence (O)"

        # Giants and Supergiants (high predicted luminosity, cooler temperature)
        if lum > 1000:
            if temp < 6000:
                classification = "Supergiant"
            elif temp >= 6000:
                classification = "Blue Supergiant"
        elif lum > 100:
            if temp < 6000:
                classification = "Giant"
            elif temp >= 6000:
                classification = "Blue Giant"

        # White Dwarfs (low luminosity, high temperature)
        if lum < 0.01 and temp > 8000:
            classification = "White Dwarf"

        # Subgiants (between main sequence and giants)
        if 10 < lum < 100 and temp < 8000:
            classification = "Subgiant"

        return classification

    # Visualizes the model's capabilities using various graphs
    def visualize_model(self, filename='trad_ml_model_results.png'): #Saves graphs to a file
        if self.y_pred is None:
            raise ValueError("Model has not been trained yet.") #Ensures the model is trained

        fig, axes = plt.subplots(1, 3, figsize=(18, 5)) #Plots 3 subplots in the main plot

        #Plot 1
        axes[0].scatter(self.y_test, self.y_pred, alpha=0.5, s=20) # Plots points based on the predicted and true values
        # Plots a line of perfect predictions (The closer the points are to it, the more accurate the prediction)
        axes[0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Luminosity')
        axes[0].set_ylabel('Predicted Luminosity')
        r2 = r2_score(self.y_test, self.y_pred) # R² value
        axes[0].set_title(f'Traditional ML Model (R² = {r2:.4f})')
        axes[0].grid(True, alpha=0.3)

        #Plot 2
        axes[1].barh(self.feature_names, self.model.feature_importances_) # Feature importances
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Feature Importance')
        axes[1].grid(True, alpha=0.3)

        #Plot 3
        residuals = self.y_test - self.y_pred # Shows the errors in the model (Overprediction or Underprediction)
        axes[2].scatter(self.y_pred, residuals, alpha=0.5, s=20) # Plots predictions and if it overpredicted or underpredicted
        axes[2].axhline(y=0, color='r', linestyle='--', lw=2) # Line of perfect predictions
        axes[2].set_xlabel('Predicted Luminosity')
        axes[2].set_ylabel('Residuals')
        axes[2].set_title('Residual Plot')
        axes[2].grid(True, alpha=0.3)

        # Makes sure the graph is well-organized and saves it.
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')


'''================================================================================================================================================================'''
'''================================================================================================================================================================'''
'''================================================================================================================================================================'''

class GraphModel(MLModel):
    # All aspects not explicitly explained are such because they have been explained in the previous model

    #K-Neighbors are the 5 closest neighbors in 3D space
    def __init__(self, csv_path=file_path, nrows=None, k_neighbors=5):
        super().__init__(csv_path, nrows)
        self.required_features = ['ra', 'dec', 'mass', 'radius', 'temp', 'dist', 'absmag', 'lum']
        self.k_neighbors = k_neighbors
        self.graph = None
        # Certain features will be added by the model itself, so their names are specified here.
        self.feature_names = ['mass', 'radius', 'temp', 'absmag', 'neighbor_mass_avg', 'neighbor_radius_avg', 'neighbor_temp_avg', 'neighbor_absmag_avg']



    def set_data(self):
        print(f"\nPreparing data for NetworkX Graph-based ML Model...")

        cleandata = self.stardata.dropna(subset=self.required_features).reset_index(drop=True)

        print(f"  Stars after filtering: {len(cleandata)}")

        print(f"  Building spatial graph (k={self.k_neighbors} neighbors)...")
        self.graph = nx.Graph() # Creates a graph using NetworkX

        # Adds various nodes to the graph, each representing a star and it's required features
        for index, row in cleandata.iterrows():
            self.graph.add_node(index,
                                mass=row['mass'],
                                radius=row['radius'],
                                temp=row['temp'],
                                dist=row['dist'],
                                absmag=row['absmag'],
                                lum=row['lum'],
                                ra=row['ra'],
                                dec=row['dec']
                                )

        # Sets its coordinates in space (x, y, z)
        coords = cleandata[['ra', 'dec', 'dist']].values

        # Finds the nearest neighbors to each given star using the coordinates
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # Adds edges between neighbors based on how close they are to each other in 3D space
        for i, neighbors in enumerate(indices):
            for j, neighbor_index in enumerate(neighbors[1:]):  # Skip self
                if i < neighbor_index:  # Avoid duplicate edges
                    distance = distances[i][j + 1]
                    self.graph.add_edge(i, neighbor_index, distance=distance)

        print(f"  Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # Obtains node features from the graph
        print(f"  Extracting features from graph...")

        node_features = []
        node_targets = []

        for node in self.graph.nodes():
            features = [
                self.graph.nodes[node]['mass'],
                self.graph.nodes[node]['radius'],
                self.graph.nodes[node]['temp'],
                self.graph.nodes[node]['absmag']
            ]

            # Adds neighborhood aggregation features as a list
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                neighbor_masses = [self.graph.nodes[n]['mass'] for n in neighbors]
                neighbor_radii = [self.graph.nodes[n]['radius'] for n in neighbors]
                neighbor_temps = [self.graph.nodes[n]['temp'] for n in neighbors]
                neighbor_absmag = [self.graph.nodes[nodes]['absmag'] for nodes in neighbors]

                features.extend([
                    np.mean(neighbor_masses),
                    np.mean(neighbor_radii),
                    np.mean(neighbor_temps),
                    np.mean(neighbor_absmag)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Adds individual and neighborhood features to the empty feature array to
            # create an array with all the necessary data for prediction.
            # Fills the target array with the lum value of every node.
            node_features.append(features)
            node_targets.append(self.graph.nodes[node]['lum'])

        # Undergoes the main data setting process, splitting the data into training and testing sets and scaling the
        # data for normalization.
        X = np.array(node_features)
        y = np.array(node_targets)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):

        if self.graph is None:
            self.set_data()

        # Trains the data much in the same way the previous model did.
        print(f"\nTraining Graph-Based Random Forest Model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)
        print("Training complete!")

    def visualize_model(self, filename='graph_based_ml_results.png'):
        if self.y_pred is None:
            raise ValueError("Model has not been trained yet.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # The first graph is a plotting of 500 stars from the NetworkX graph and their distance based edges.
        # Its purpose is to show the network of connections that the model used to make its predictions.
        sample_nodes = list(self.graph.nodes())[:500]
        subgraph = self.graph.subgraph(sample_nodes)
        pos = {node: (self.graph.nodes[node]['ra'], self.graph.nodes[node]['dec'])
               for node in subgraph.nodes()}

        axes[0].set_title("Graph Structure (500 Stars)")
        nx.draw_networkx_nodes(subgraph, pos, node_size=20, node_color='lightblue', ax=axes[0])
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, ax=axes[0])
        axes[0].set_xlabel('Right Ascension')
        axes[0].set_ylabel('Declination')

        # R² Graph
        axes[1].scatter(self.y_test, self.y_pred, alpha=0.5, s=20)
        axes[1].plot([self.y_test.min(), self.y_test.max()],
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('True Luminosity')
        axes[1].set_ylabel('Predicted Luminosity')
        r2 = r2_score(self.y_test, self.y_pred)
        axes[1].set_title(f'Graph-Based Model (R² = {r2:.4f})')
        axes[1].grid(True, alpha=0.3)

        # Feature importance graph
        axes[2].barh(self.feature_names, self.model.feature_importances_)
        axes[2].set_xlabel('Importance')
        axes[2].set_title('Feature Importance')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')

'''================================================================================================================================================================'''
'''================================================================================================================================================================'''
'''================================================================================================================================================================'''

class ClusterModel(MLModel):
    def __init__(self, csv_path=file_path, nrows=30000, n_clusters=10):
        super().__init__(csv_path, nrows)
        self.required_features = ['mass', 'radius', 'temp', 'dist', 'ra', 'dec', 'absmag', 'lum']
        self.n_clusters = n_clusters # Sets the amount of clusters it will split stars into
        self.cluster_model = None
        self.cluster_labels = None
        # Names of the required data, including data created by the model itself
        self.feature_names = ['mass', 'radius', 'temp', 'dist', 'ra', 'dec', 'absmag', 'cluster_id',
                              'cluster_avg_mass', 'cluster_avg_radius', 'cluster_avg_temp', 'cluster_avg_dist',
                              'cluster_avg_ra', 'cluster_avg_dec', 'cluster_avg_absmag']




    def set_data(self):

        print(f"\nPreparing data for Cluster-Based ML Model...")

        # Cleans the data much like the previous models did
        cleandata = self.stardata.dropna(subset=self.required_features).reset_index(drop=True)

        print(f"  Stars after filtering: {len(cleandata)}")

        clustering_features = cleandata[['mass', 'radius', 'temp', 'dist', 'ra', 'dec', 'absmag']].values

        # Scales the data for normalization
        scaler_cluster = StandardScaler()
        clustering_features_scaled = scaler_cluster.fit_transform(clustering_features)

        # Groups the stars into clusters based on feature similarities
        print(f"  Performing K-Means clustering (k={self.n_clusters})...")
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.cluster_model.fit_predict(clustering_features_scaled)

        cleandata['cluster'] = self.cluster_labels

        cluster_stats = cleandata.groupby('cluster').agg({
            'mass': 'mean',
            'radius': 'mean',
            'temp': 'mean',
            'dist': 'mean',
            'ra': 'mean',
            'dec': 'mean',
            'absmag': 'mean',
            'lum': ['mean', 'count']
        })

        print(f"\n  Cluster Statistics:")
        print(f"  {'Cluster':<10} {'Count':<8} {'Avg Mass':<12} {'Avg Radius':<12} {'Avg Temp':<12}")
        print(f"  {'-' * 70}")
        for i in range(self.n_clusters):
            count = cluster_stats.loc[i, ('lum', 'count')]
            avg_mass = cluster_stats.loc[i, ('mass', 'mean')]
            avg_radius = cluster_stats.loc[i, ('radius', 'mean')]
            avg_temp = cluster_stats.loc[i, ('temp', 'mean')]
            avg_dist = cluster_stats.loc[i, ('dist', 'mean')]
            avg_ra = cluster_stats.loc[i, ('ra', 'mean')]
            avg_dec = cluster_stats.loc[i, ('dec', 'mean')]
            avg_absmag = cluster_stats.loc[i, ('absmag', 'mean')]
            print(f"  {i:<10} {int(count):<8} {avg_mass:<12.3f} {avg_radius:<12.3f} {avg_temp:<12.1f} "
                  f"{avg_dist:<12.2f} {avg_ra:<12.2f} {avg_dec:<12.2f} {avg_absmag:<12.2f}")

        node_features = []
        node_targets = []

        for idx, row in cleandata.iterrows():
            cluster_id = row['cluster']

            # Get cluster statistics
            cluster_avg_mass = cluster_stats.loc[cluster_id, ('mass', 'mean')]
            cluster_avg_radius = cluster_stats.loc[cluster_id, ('radius', 'mean')]
            cluster_avg_temp = cluster_stats.loc[cluster_id, ('temp', 'mean')]
            cluster_avg_dist = cluster_stats.loc[cluster_id, ('dist', 'mean')]
            cluster_avg_ra = cluster_stats.loc[cluster_id, ('ra', 'mean')]
            cluster_avg_dec = cluster_stats.loc[cluster_id, ('dec', 'mean')]
            cluster_avg_absmag = cluster_stats.loc[cluster_id, ('absmag', 'mean')]

            features = [
                row['mass'],
                row['radius'],
                row['temp'],
                row['dist'],
                row['ra'],
                row['dec'],
                row['absmag'],
                cluster_id,
                cluster_avg_mass,
                cluster_avg_radius,
                cluster_avg_temp,
                cluster_avg_dist,
                cluster_avg_ra,
                cluster_avg_dec,
                cluster_avg_absmag
            ]

            node_features.append(features)
            node_targets.append(row['lum'])

        X = np.array(node_features)
        y = np.array(node_targets)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"\n  Training set: {len(self.X_train)} stars")
        print(f"  Testing set: {len(self.X_test)} stars")

    def train_model(self):

        if self.X_train is None:
            self.set_data()

        print(f"\nTraining Cluster-Based Random Forest Model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)
        print("Training complete!")

    def visualize_model(self, filename='cluster_based_ml_results.png'):

        if self.y_pred is None:
            raise ValueError("Model has not been trained yet.")

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.5)

        ax1 = fig.add_subplot(gs[0, 0])

        cleandata = self.stardata.dropna(subset=self.required_features)
        cleandata['cluster'] = self.cluster_labels

        scatter = ax1.scatter(cleandata['mass'], cleandata['radius'],
                              c=cleandata['cluster'], cmap='tab10',
                              alpha=0.6, s=20)
        ax1.set_xlabel('Mass (Solar Masses)')
        ax1.set_ylabel('Radius (Solar Radii)')
        ax1.set_title('Clusters in Mass-Radius Space')
        ax1.set_xlim(0, cleandata['mass'].quantile(0.95))
        ax1.set_ylim(0, cleandata['radius'].quantile(0.95))
        plt.colorbar(scatter, ax=ax1, label='Cluster ID')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        scatter2 = ax2.scatter(cleandata['temp'], cleandata['lum'],
                               c=cleandata['cluster'], cmap='tab10',
                               alpha=0.6, s=20)
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Luminosity (Solar Luminosities)')
        ax2.set_title('Clusters in Temperature-Luminosity Space')
        ax2.set_xlim(cleandata['temp'].quantile(0.05), cleandata['temp'].quantile(0.95))
        ax2.set_ylim(0, cleandata['lum'].quantile(0.95))
        plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        cluster_counts = cleandata['cluster'].value_counts().sort_index()
        ax3.bar(cluster_counts.index, cluster_counts.values, color='steelblue')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Stars')
        ax3.set_title('Stars per Cluster')
        ax3.grid(True, alpha=0.3, axis='y')

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(self.y_test, self.y_pred, alpha=0.5, s=20, c='purple')
        ax4.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax4.set_xlabel('True Luminosity')
        ax4.set_ylabel('Predicted Luminosity')
        r2 = r2_score(self.y_test, self.y_pred)
        ax4.set_title(f'Cluster-Based Model (R² = {r2:.4f})')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.barh(self.feature_names, self.model.feature_importances_)
        ax5.set_xlabel('Importance')
        ax5.set_title('Feature Importance')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        residuals = self.y_test - self.y_pred
        ax6.scatter(self.y_pred, residuals, alpha=0.5, s=20, c='orange')
        ax6.axhline(y=0, color='r', linestyle='--', lw=2)
        ax6.set_xlabel('Predicted Luminosity')
        ax6.set_ylabel('Residuals')
        ax6.set_title('Residual Plot')
        ax6.grid(True, alpha=0.3)

        plt.savefig(filename, dpi=300, bbox_inches='tight')

'''================================================================================================================================================================'''
'''================================================================================================================================================================'''
'''================================================================================================================================================================'''

class ModelTrainer:

    def __init__(self, strategy: MLModel):
        self.strategy = strategy

    def set_strategy(self, strategy: MLModel):
        """Change the strategy at runtime"""
        self.strategy = strategy

    def run_training_pipeline(self):
        print(f"\n{'=' * 60}")
        print(f"Running {self.strategy.__class__.__name__}")
        print(f"{'=' * 60}")

        # Train model
        self.strategy.train_model()

        # Evaluate
        results = self.strategy.evaluate_model()

        # Show feature importance
        self.strategy.get_featimp()

        # Visualize
        filename = f"{graph_path}//{self.strategy.__class__.__name__.lower()}_results.png"
        self.strategy.visualize_model(filename)

        return results

'''================================================================================================================================================================'''
'''================================================================================================================================================================'''
'''================================================================================================================================================================'''

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

    # Strategy 1: Traditional ML
    print("\n" + "=" * 60)
    print("STRATEGY 1: TRADITIONAL ML MODEL")
    print("=" * 60)

    traditional_model = TradModel(nrows=5000)
    trainer = ModelTrainer(traditional_model)
    results_traditional = trainer.run_training_pipeline()

    # Strategy 2: Graph-Based ML
    print("\n" + "=" * 60)
    print("STRATEGY 2: GRAPH-BASED ML MODEL")
    print("=" * 60)

    graph_model = GraphModel(nrows=5000, k_neighbors=5)
    trainer.set_strategy(graph_model)
    results_graph = trainer.run_training_pipeline()

    # Strategy 3: Cluster-Based ML
    print("\n" + "=" * 60)
    print("STRATEGY 3: CLUSTER-BASED ML MODEL")
    print("=" * 60)

    cluster_model = ClusterModel(nrows=5000, n_clusters=10)
    trainer.set_strategy(cluster_model)
    results_cluster = trainer.run_training_pipeline()

    # Compare results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Traditional ML  - R²: {results_traditional['r2']:.4f}, RMSE: {results_traditional['rmse']:.4f}")
    print(f"Graph-Based ML  - R²: {results_graph['r2']:.4f}, RMSE: {results_graph['rmse']:.4f}")
    print(f"Cluster-Based ML - R²: {results_cluster['r2']:.4f}, RMSE: {results_cluster['rmse']:.4f}")

    # Determine best model
    results_dict = {
        'Traditional ML': results_traditional['r2'],
        'Graph-Based ML': results_graph['r2'],
        'Cluster-Based ML': results_cluster['r2']
    }

    best_model = max(results_dict, key=results_dict.get)
    print(f"\n✓ Best performing model: {best_model} (R² = {results_dict[best_model]:.4f})")




