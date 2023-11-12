# Created using chatgpt-4

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Load color data from CSV
df = pd.read_csv('color_counts.csv')

# Assuming the CSV has 'Color' in the format '(r, g, b)' and a 'Count' column
# We need to convert 'Color' from string to a list of integers
df['Color'] = df['Color'].apply(lambda x: [int(i) for i in x.strip('()').split(', ')])

# Extract RGB values and weights (counts)
colors = np.array(df['Color'].tolist())
weights = df['Count'].values

# Choose the number of clusters
k = 7  # for example, you can change this number based on your requirements

# Perform weighted k-means clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(colors, sample_weight=weights)

# Get the centroids of the clusters
centroids = kmeans.cluster_centers_

# You can also assign each original color to its cluster
df['Cluster'] = kmeans.predict(colors)

# Save the DataFrame with cluster assignments to a new CSV file
df.to_csv('color_clusters_small.csv', index=False)

# Normalize the RGB values to be between 0 and 1
normalized_colors = [[r/255, g/255, b/255] for r, g, b in centroids]

# Now centroids contain the 'average' color of each cluster
print(normalized_colors)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Add colors to the plot
for i, color in enumerate(normalized_colors):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

# Remove the axes for a cleaner look
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

# Save the plot as a PNG file
plt.savefig('color_visualization.png')

# Show the plot
plt.show()
