import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors

# interactive node
#srun -n 4 --time=100:00:00 --pty bash -i
#BIGMEM: srun -p bigmem -n 4 --time=100:00:00 --pty bash -i


BASE_PATH = "/Users/zeus_10/Desktop/Projects/V1.0/SC-RNAseq/Single-cell-Analysis-of-Cystic-Fibrosis"

# Input files
file1 = f"{BASE_PATH}/chunk10_metadata.csv"
file2 = f"{BASE_PATH}/chunk10_counts.npz"

# Load data
df = pd.read_csv(file1)
counts = np.load(file2)

# Extracting matrix/counts from npz file
keys = counts.files
row_array = counts['row']
col_array = counts['col']
data_array = counts['data']
count_matrix = np.zeros((row_array.max() + 1, col_array.max() + 1), dtype=data_array.dtype)
count_matrix[row_array, col_array] = data_array

# Extracting metadata of cystic fibrosis and normal
cystic = df[df["disease"] == "pulmonary fibrosis"]
normal = df[df["disease"] == "normal"]

# Extracting index related to cystic fibrosis and normal
c_index = cystic.index
n_index = normal.index

# Extracting count matrix related to cystic fibrosis and normal
cystic_counts = count_matrix[:, c_index]
normal_counts = count_matrix[:, n_index]

# Transposing the matrix for KNN
cystic_counts = cystic_counts.T
normal_counts = normal_counts.T

# output
file1 = f"{BASE_PATH}/model_output_2024-04-17-19-45-31.npz"
file2 = f"{BASE_PATH}/model_output_2024-04-17-20-34-07.npz"

# Load model outputs
counts1 = np.load(file1)
counts2 = np.load(file2)

# Extracting matrix from model outputs
keys1 = counts1.files
keys2 = counts2.files
matrix1 = counts1['arr_0']
matrix2 = counts2['arr_0']

# Function to determine best k using cross-validation
def find_best_k(normal_counts, cystic_counts, k_range=range(1, 21)):
    best_k = None
    best_score = float('inf')  # Minimize the score
    for k in k_range:
        knn = NearestNeighbors(n_neighbors=k, metric='manhattan')
        knn.fit(normal_counts)
        distances, indices = knn.kneighbors(cystic_counts)
        mean_distance = np.mean(distances)
        if mean_distance < best_score:
            best_score = mean_distance
            best_k = k
    return best_k

# Find the best k
best_k = find_best_k(normal_counts, cystic_counts)

# Using the best k, calculate distances
knn = NearestNeighbors(n_neighbors=best_k, metric='manhattan')
knn.fit(normal_counts)
distances, indices = knn.kneighbors(cystic_counts)

# Min-max normalization
def normalize_distances(distances):
    min_distance = min(distances)
    max_distance = max(distances)
    normalized_distances = [(d - min_distance) / (max_distance - min_distance) for d in distances]
    return normalized_distances

# Create dataframe of indices and distances
distances_normalized = normalize_distances(distances)
df_distances = pd.DataFrame(distances_normalized, columns=["Distance"])
df_indices = pd.DataFrame(indices, columns=['NN_Index'])
df_output = pd.concat([df_distances, df_indices], axis=1)

# Tissue-type information extraction
tissue = df['tissue']
tissue_index = tissue[df_output['NN_Index']]
tissue_index = pd.DataFrame(tissue_index)
tissue_index.reset_index(inplace=True)
tissue_index.rename(columns={'index': 'NN_Index'}, inplace=True)
merged_data = pd.merge(df_output, tissue_index, on="NN_Index", how="inner")
grouped_data = merged_data.groupby('tissue')
mean_values = grouped_data.mean()

# Cell Type information extraction
cell_type = df['cell_type']
cell_type_index = cell_type[df_output['NN_Index']]
cell_type_index = pd.DataFrame(cell_type_index)
cell_type_index.reset_index(inplace=True)
cell_type_index.rename(columns={'index': 'NN_Index'}, inplace=True)
merged_data_cell = pd.merge(df_output, cell_type_index, on="NN_Index", how="inner")
grouped_data_cell = merged_data_cell.groupby('cell_type')
mean_values_cell_type = grouped_data_cell['Distance'].mean()
df_2 = pd.DataFrame(mean_values_cell_type)
pivot_table = df_2.pivot_table(values='Distance', index='cell_type', aggfunc='mean')
pivot_table = pivot_table.sort_values(by='Distance', ascending=False)


# DOT plot of affected cell types
plt.figure(figsize=(10, 12))
pivot_table1 = pivot_table.sort_values(by='Distance', ascending=True)
sizes = 50 + (pivot_table1['Distance'] - pivot_table1['Distance'].min()) * 450

colors = []
for i in range(len(pivot_table1)):
    if i < 5:
        colors.append('green')
    elif i >= len(pivot_table1) - 5:
        colors.append('red')
    else:
        colors.append('blue')

plt.scatter(pivot_table1['Distance'], pivot_table1['cell_type'], color=colors, alpha=0.7, s=sizes)
plt.xlabel('Relative Distance', fontsize=14, fontweight='semibold')
plt.ylabel('Cell Types', fontsize=14, fontweight='semibold')
plt.xticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
plt.text(0.5, 1.05, 'Relative Differences Between', ha='center', va='bottom', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.5, 1.02, 'Cystic Fibrosis and Normal Cell Types', ha='center', va='bottom', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)

legend = plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Most affected', markerfacecolor='red', markersize=10),
                             plt.Line2D([0], [0], marker='o', color='w', label='Least affected', markerfacecolor='green', markersize=10)],
                   loc='upper left', fontsize=12, frameon=True)
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_edgecolor('black')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{BASE_PATH}/dot_plot_minmax.png")

# Bar Graph of affected tissue type information
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=['lung', 'lung parenchyma', 'nose', 'respiratory airway'], y='Distance', data=mean_values, color='skyblue')
plt.xlabel('Tissues')
plt.ylabel('Differences')
plt.title('Mean Difference of Cystic Fibrosis and Normal Tissue')
plt.savefig(f"{BASE_PATH}/Bar_Graphs_minmax.png")