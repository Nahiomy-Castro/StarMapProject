#Necessary imports for all models
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

# Build the path to your file
file_path = script_dir / 'data' / 'hyg_v42_updated.csv'

graph_path = script_dir / 'view'


#Abstract base class for ML Models
class MLModel(ABC):
    #Constructor includes filepath to be used along with the amount of rows
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
    def execute_model(self):
        pass
        #Each model defines this

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
        rmse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print(f"\n{'=' * 60}")
        print(f"Model Evaluation Results:")
        print(f"{'=' * 60}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"{'=' * 60}")

        return {'r2': r2, 'rmse': rmse, 'mae': mae}

    def predict(self, input_data):
        """
        Predict luminosity for a single star or batch of stars.

        Parameters:
        -----------
        input_data : dict or pd.DataFrame or np.array
            Single row or multiple rows of star data

        Returns:
        --------
        float or np.array : Predicted luminosity value(s)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")

        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call train_model() first.")

        # Convert dict to array
        if isinstance(input_data, dict):
            input_array = np.array([[input_data[feature] for feature in self.feature_names]])
        # Convert DataFrame to array
        elif isinstance(input_data, pd.DataFrame):
            input_array = input_data[self.feature_names].values
        # Assume it's already an array
        else:
            input_array = np.array(input_data).reshape(1, -1) if len(np.array(input_data).shape) == 1 else input_data

        # Scale the input using the same scaler from training
        input_scaled = self.scaler.transform(input_array)

        # Make prediction
        prediction = self.model.predict(input_scaled)

        return prediction[0] if len(prediction) == 1 else prediction


    @abstractmethod
    def visualize_model(self, filename):
        pass

    def get_featimp(self):
        feature_dict = None

        if self.model is None:
            raise ValueError("Model has not been trained yet.")

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

class TradModel(MLModel):

    def __init__(self, csv_path=file_path, nrows=30000):
        super().__init__(csv_path, nrows)
        self.required_features = ['mass', 'radius', 'temp', 'dist', 'absmag']
        self.target = 'lum'
        self.feature_names = self.required_features.copy()

    def execute_model(self):
        print(f"\nExecuting Traditional ML Model...\n")
        self.set_data()
        self.train_model()
        self.evaluate_model()
        self.visualize_model()
        self.get_featimp()
        print('=='*60)

    def set_data(self):
        print(f"\nPreparing data for Traditional ML Model...")
        cleandata = self.stardata.dropna(subset=self.required_features + [self.target])

        print(f"  Stars after filtering: {len(cleandata)}")

        X = cleandata[self.required_features].values
        y = cleandata[self.target].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"  Training set: {len(self.X_train)} stars")
        print(f"  Testing set: {len(self.X_test)} stars")

    def train_model(self):
        if self.X_train is None:
            self.set_data()

        print(f"\nTraining Traditional Random Forest Model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)

    def input_predict(self, star_id):
        stardata = pd.read_csv(file_path)

        star_row = stardata[stardata['id'].astype(str) == str(star_id)]

        if star_row.empty:
            raise ValueError(f"Star with ID {star_id} not found")

        star = star_row.iloc[0]

        star_data = {
            'mass': star['mass'],
            'radius': star['radius'],
            'temp': star['temp'],
            'dist': star['dist'],
            'absmag': star['absmag']
        }

        return self.predict(star_data)

    def stellar_classification(self, temp, predlum):
        classification = "Unclassified"

        if 2500 <= temp < 3500:
            if 0.0001 < predlum < 0.08:
                classification = "Main Sequence (M)"
        elif 3500 <= temp < 5000:
            if 0.08 < predlum < 0.6:
                classification = "Main Sequence (K)"
        elif 5000 <= temp < 6000:
            if 0.6 < predlum < 1.5:
                classification = "Main Sequence (G)"
        elif 6000 <= temp < 7500:
            if 1.5 < predlum < 5:
                classification = "Main Sequence (F)"
        elif 7500 <= temp < 10000:
            if 5 < predlum < 25:
                classification = "Main Sequence (A)"
        elif 10000 <= temp < 30000:
            if 25 < predlum < 10000:
                classification = "Main Sequence (B)"
        elif temp >= 30000:
            if predlum > 10000:
                classification = "Main Sequence (O)"

            # Giants and Supergiants (high predicted luminosity, cooler temperature)
        if predlum > 1000:
            if temp < 6000:
                classification = "Supergiant"
            else:
                classification = "Blue Supergiant"
        elif predlum > 100:
            if temp < 6000:
                classification = "Giant"
            else:
                classification = "Blue Giant"

            # White Dwarfs (low luminosity, high temperature)
        if predlum < 0.01 and temp  > 8000:
            classification = "White Dwarf"

            # Subgiants (between main sequence and giants)
        if 10 < predlum < 100 and temp < 8000:
            classification = "Subgiant"

        return classification

    def visualize_model(self, filename='trad_ml_model_results.png'):
        if self.y_pred is None:
            raise ValueError("Model has not been trained yet.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(self.y_test, self.y_pred, alpha=0.5, s=20)
        axes[0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Luminosity')
        axes[0].set_ylabel('Predicted Luminosity')
        r2 = r2_score(self.y_test, self.y_pred)
        axes[0].set_title(f'Traditional ML Model (R² = {r2:.4f})')
        axes[0].grid(True, alpha=0.3)

        axes[1].barh(self.feature_names, self.model.feature_importances_)
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Feature Importance')
        axes[1].grid(True, alpha=0.3)

        residuals = self.y_test - self.y_pred
        axes[2].scatter(self.y_pred, residuals, alpha=0.5, s=20)
        axes[2].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[2].set_xlabel('Predicted Luminosity')
        axes[2].set_ylabel('Residuals')
        axes[2].set_title('Residual Plot')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')


'''================================================================================================================================================================'''
'''================================================================================================================================================================'''
'''================================================================================================================================================================'''

class GraphModel(MLModel):

    def __init__(self, csv_path=file_path, nrows=30000, k_neighbors=5):
        super().__init__(csv_path, nrows)
        self.required_features = ['ra', 'dec', 'mass', 'radius', 'temp', 'dist', 'absmag', 'lum']
        self.k_neighbors = k_neighbors
        self.graph = None
        self.feature_names = ['mass', 'radius', 'temp', 'dist', 'absmag', 'neighbor_mass_avg', 'neighbor_radius_avg', 'neighbor_temp_avg',
                              'neighbor_dist_avg', 'neighbor_absmag_avg']

    def execute_model(self):
        print(f"\nExecuting NetworkX Graph-based ML Model...\n")
        self.set_data()
        self.train_model()
        self.evaluate_model()
        self.visualize_model()
        self.get_featimp()
        print('=='*60)


    def set_data(self):
        print(f"\nPreparing data for NetworkX Graph-based ML Model...")

        cleandata = self.stardata.dropna(subset=self.required_features).reset_index(drop=True)

        print(f"  Stars after filtering: {len(cleandata)}")

        print(f"  Building spatial graph (k={self.k_neighbors} neighbors)...")
        self.graph = nx.Graph()

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

        coords = cleandata[['ra', 'dec']].values
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        for i, neighbors in enumerate(indices):
            for j, neighbor_index in enumerate(neighbors[1:]):  # Skip self
                if i < neighbor_index:  # Avoid duplicate edges
                    distance = distances[i][j + 1]
                    self.graph.add_edge(i, neighbor_index, distance=distance)

        print(f"  Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # Extract features from graph
        print(f"  Extracting features from graph...")

        node_features = []
        node_targets = []

        for node in self.graph.nodes():
            features = [
                self.graph.nodes[node]['mass'],
                self.graph.nodes[node]['radius'],
                self.graph.nodes[node]['temp'],
                self.graph.nodes[node]['dist'],
                self.graph.nodes[node]['absmag']
            ]

            # Add neighborhood aggregation features
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                neighbor_masses = [self.graph.nodes[n]['mass'] for n in neighbors]
                neighbor_radii = [self.graph.nodes[n]['radius'] for n in neighbors]
                neighbor_temps = [self.graph.nodes[n]['temp'] for n in neighbors]
                neighbor_dist = [self.graph.nodes[n]['dist'] for n in neighbors]
                neighbor_absmag = [self.graph.nodes[nodes]['absmag'] for nodes in neighbors]

                features.extend([
                    np.mean(neighbor_masses),
                    np.mean(neighbor_radii),
                    np.mean(neighbor_temps),
                    np.mean(neighbor_dist),
                    np.mean(neighbor_absmag)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            node_features.append(features)
            node_targets.append(self.graph.nodes[node]['lum'])

        X = np.array(node_features)
        y = np.array(node_targets)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):

        if self.graph is None:
            self.set_data()

        print(f"\nTraining Graph-Based Random Forest Model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)
        print("Training complete!")

    def visualize_model(self, filename='graph_based_ml_results.png'):
        if self.y_pred is None:
            raise ValueError("Model has not been trained yet.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sample_nodes = list(self.graph.nodes())[:1000]
        subgraph = self.graph.subgraph(sample_nodes)
        pos = {node: (self.graph.nodes[node]['ra'], self.graph.nodes[node]['dec'])
               for node in subgraph.nodes()}

        axes[0].set_title("Graph Structure (1000 Stars)")
        nx.draw_networkx_nodes(subgraph, pos, node_size=20, node_color='lightblue', ax=axes[0])
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, ax=axes[0])
        axes[0].set_xlabel('Right Ascension')
        axes[0].set_ylabel('Declination')


        axes[1].scatter(self.y_test, self.y_pred, alpha=0.5, s=20)
        axes[1].plot([self.y_test.min(), self.y_test.max()],
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('True Luminosity')
        axes[1].set_ylabel('Predicted Luminosity')
        r2 = r2_score(self.y_test, self.y_pred)
        axes[1].set_title(f'Graph-Based Model (R² = {r2:.4f})')
        axes[1].grid(True, alpha=0.3)


        axes[2].barh(self.feature_names, self.model.feature_importances_)
        axes[2].set_xlabel('Importance')
        axes[2].set_title('Feature Importance')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        #nx.write_gpickle(self.graph, 'stellar_luminosity_graph.gpickle')

'''================================================================================================================================================================'''
'''================================================================================================================================================================'''
'''================================================================================================================================================================'''

class ClusterModel(MLModel):
    def __init__(self, csv_path=file_path, nrows=30000, n_clusters=10):
        super().__init__(csv_path, nrows)
        self.required_features = ['mass', 'radius', 'temp', 'dist', 'ra', 'dec', 'absmag', 'lum']
        self.n_clusters = n_clusters
        self.cluster_model = None
        self.cluster_labels = None
        self.feature_names = ['mass', 'radius', 'temp', 'dist', 'ra', 'dec', 'absmag', 'cluster_id',
                              'cluster_avg_mass', 'cluster_avg_radius', 'cluster_avg_temp', 'cluster_avg_dist',
                              'cluster_avg_ra', 'cluster_avg_dec', 'cluster_avg_absmag']


    def execute_model(self):
        print(f"\nExecuting NetworkX Graph-based ML Model...\n")
        self.set_data()
        self.train_model()
        self.evaluate_model()
        self.visualize_model()
        self.get_featimp()
        print('==' * 60)


    def set_data(self):

        print(f"\nPreparing data for Cluster-Based ML Model...")

        cleandata = self.stardata.dropna(subset=self.required_features).reset_index(drop=True)

        print(f"  Stars after filtering: {len(cleandata)}")

        clustering_features = cleandata[['mass', 'radius', 'temp', 'dist', 'ra', 'dec', 'absmag']].values

        scaler_cluster = StandardScaler()
        clustering_features_scaled = scaler_cluster.fit_transform(clustering_features)

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
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

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




