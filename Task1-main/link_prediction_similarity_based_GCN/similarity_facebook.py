import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from tqdm import tqdm  
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier\

# Calculate features for each pair of nodes
def common_neighbors(x, y, subgraph):
    return len(list(nx.common_neighbors(subgraph, x, y)))

def adamic_adar(x, y, subgraph):
    common_neighbors_list = list(nx.common_neighbors(subgraph, x, y))
    if len(common_neighbors_list) == 0:
        return 1e-6  # 如果没有共同邻居，返回一个小的正数
    return sum(1/np.log(len(list(subgraph.neighbors(z)))) for z in common_neighbors_list)

def jaccard_coefficient(x, y, subgraph):
    common_neighbors_list = list(nx.common_neighbors(subgraph, x, y))
    if len(common_neighbors_list) == 0:
        return 1e-6  # 如果没有共同邻居，返回一个小的正数
    return len(common_neighbors_list) / (len(set(subgraph.neighbors(x))) + len(set(subgraph.neighbors(y))) - len(common_neighbors_list))

def resource_allocation(x, y, subgraph):
    common_neighbors_list = list(nx.common_neighbors(subgraph, x, y))
    if len(common_neighbors_list) == 0:
        return 1e-6  # 如果没有共同邻居，返回一个小的正数
    return sum(1/len(list(subgraph.neighbors(z))) for z in common_neighbors_list)

def preferential_attachment(x, y, subgraph):
    neighbors_x = list(subgraph.neighbors(x))
    neighbors_y = list(subgraph.neighbors(y))
    if len(neighbors_x) == 0 or len(neighbors_y) == 0:
        return 1e-6  # 如果任一节点没有邻居，返回一个小的正数
    return len(neighbors_x) * len(neighbors_y)

def dice_coefficient(x, y, subgraph):
    neighbors_x = set(subgraph.neighbors(x))
    neighbors_y = set(subgraph.neighbors(y))
    
    intersection_size = len(neighbors_x & neighbors_y)
    if len(neighbors_x) + len(neighbors_y) == 0:
        return 1e-6  # 如果邻居集合为空，返回一个小的正数
    return 2 * intersection_size / (len(neighbors_x) + len(neighbors_y))

def cosine_similarity(x, y, subgraph):
    neighbors_x = set(subgraph.neighbors(x))
    neighbors_y = set(subgraph.neighbors(y))
    
    intersection_size = len(neighbors_x & neighbors_y)
    if len(neighbors_x) == 0 or len(neighbors_y) == 0:
        return 1e-6  # 如果任一节点没有邻居，返回一个小的正数
    return intersection_size / (np.sqrt(len(neighbors_x)) * np.sqrt(len(neighbors_y)))

def pearson_correlation(x, y, subgraph):
    degree_x = len(list(subgraph.neighbors(x)))
    degree_y = len(list(subgraph.neighbors(y)))

    neighbors_x = list(subgraph.neighbors(x))
    neighbors_y = list(subgraph.neighbors(y))
    
    if len(neighbors_x) == 0 or len(neighbors_y) == 0:
        return 1e-6  # 如果任一节点没有邻居，返回一个小的正数
    
    mean_degree_x = np.mean([len(list(subgraph.neighbors(neighbor))) for neighbor in neighbors_x])
    mean_degree_y = np.mean([len(list(subgraph.neighbors(neighbor))) for neighbor in neighbors_y])
    
    # Pearson formula
    numerator = (degree_x - mean_degree_x) * (degree_y - mean_degree_y)
    denominator = np.sqrt((degree_x - mean_degree_x)**2 * (degree_y - mean_degree_y)**2)

    # Return Pearson Correlation (avoid division by zero)
    if denominator != 0:
        return numerator / denominator
    else:
        return 1e-6  # 如果没有变化，返回一个小的正数

# Function to calculate ROC AUC and Average Precision (AP)
def evaluate_model(model, X_test_scaled, y_test):
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # We are interested in the probability for class '1'

    # Calculate ROC AUC and AP
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    return roc_auc, ap

# Step 1: Load the graph from the edge list
G = nx.read_edgelist("facebook_combined.txt", nodetype=int)

# Step 2: Generate positive (existing edges) and negative (non-existing edges) samples
edges = list(G.edges())  # Positive edges
non_edges = list(nx.non_edges(G))  # Negative edges

# Randomly sample the negative edges to match the number of positive edges
non_edges_sample = np.random.choice(len(non_edges), size=len(edges), replace=False)
non_edges_sample = [non_edges[i] for i in non_edges_sample]

# Combine positive and negative samples
samples = edges + non_edges_sample
labels = [1] * len(edges) + [0] * len(non_edges_sample)

# Step 2.1: Shuffle the samples and labels
# Combine the samples and labels into a single list of tuples (sample, label)
data = list(zip(samples, labels))

# Shuffle the data
np.random.shuffle(data)

# Unzip the shuffled data back into samples and labels
samples, labels = zip(*data)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.25, random_state=42)

# Step 4: Create subgraphs for train and test separately
# Create subgraph for the training set (only include edges from the training set)
train_subgraph = G.edge_subgraph(X_train).copy()

# Create subgraph for the test set (only include edges from the test set)
test_subgraph = G.edge_subgraph(X_test).copy()

# Step 5: Calculate features only for training set and test set separately

# Train features
train_features = []
for edge in tqdm(X_train, desc="Calculating train features", unit="pair"):
    x, y = edge
    train_features.append([
        common_neighbors(x, y, train_subgraph),
        adamic_adar(x, y, train_subgraph),
        jaccard_coefficient(x, y, train_subgraph),
        resource_allocation(x, y, train_subgraph),
        preferential_attachment(x, y, train_subgraph),
        dice_coefficient(x, y, train_subgraph),     
        cosine_similarity(x, y, train_subgraph),
        pearson_correlation(x, y, train_subgraph)     
    ])

# Test features
test_features = []
for edge in tqdm(X_test, desc="Calculating test features", unit="pair"):
    x, y = edge
    test_features.append([
        common_neighbors(x, y, test_subgraph),
        adamic_adar(x, y, test_subgraph),
        jaccard_coefficient(x, y, test_subgraph),
        resource_allocation(x, y, test_subgraph),
        preferential_attachment(x, y, test_subgraph),
        dice_coefficient(x, y, test_subgraph),     
        cosine_similarity(x, y, test_subgraph),
        pearson_correlation(x, y, test_subgraph)     
    ])

# Convert features and labels into DataFrame for easier handling
X_train = np.array(train_features)
X_test = np.array(test_features)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Step 6: Standardize the features (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Only transform the test data

# 1. Naive Bayes (GaussianNB)
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)  # Note: Naive Bayes usually doesn't require scaling
nb_roc_auc, nb_ap = evaluate_model(nb_model, X_test_scaled, y_test)
print("Naive Bayes ROC AUC:", nb_roc_auc)
print("Naive Bayes AP:", nb_ap)

# 2. Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', gamma='scale', random_state=42, probability=True)  # Ensure probability=True to get probas
svm_model.fit(X_train_scaled, y_train)
svm_roc_auc, svm_ap = evaluate_model(svm_model, X_test_scaled, y_test)
print("SVM ROC AUC:", svm_roc_auc)
print("SVM AP:", svm_ap)

# 3. Logistic Regression
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_scaled, y_train)
log_reg_roc_auc, log_reg_ap = evaluate_model(log_reg_model, X_test_scaled, y_test)
print("Logistic Regression ROC AUC:", log_reg_roc_auc)
print("Logistic Regression AP:", log_reg_ap)

# 4. Multi-layer Perceptron (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
mlp_roc_auc, mlp_ap = evaluate_model(mlp_model, X_test_scaled, y_test)
print("MLP ROC AUC:", mlp_roc_auc)
print("MLP AP:", mlp_ap)

# 5. K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)  # 选择合适的k值
knn_model.fit(X_train_scaled, y_train)
knn_roc_auc, knn_ap = evaluate_model(knn_model, X_test_scaled, y_test)
print("KNN ROC AUC:", knn_roc_auc)
print("KNN AP:", knn_ap)

# 6. Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_roc_auc, dt_ap = evaluate_model(dt_model, X_test_scaled, y_test)
print("Decision Tree ROC AUC:", dt_roc_auc)
print("Decision Tree AP:", dt_ap)