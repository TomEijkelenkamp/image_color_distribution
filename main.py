# Created using chatgpt-4
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def extract_histogram_from_image(filename, bins=32):
    with Image.open(filename) as img:
        img = img.convert('RGB')
        img_data = np.array(img)
        hist = np.histogramdd(img_data.reshape(-1, 3), bins=bins, range=[(0, 256), (0, 256), (0, 256)])
        return hist[1], hist[0].flatten()

def aggregate_histograms(folder_path, keyword="cardboard", bins=32):
    filenames = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)
                 if fname.endswith(('.png', '.jpg', '.jpeg')) and keyword in fname]
    histograms = [extract_histogram_from_image(f, bins=bins) for f in tqdm(filenames, desc="Processing Images")]
    bins = histograms[0][0]
    aggregated_histogram = np.sum([hist[1] for hist in histograms], axis=0)
    return bins, aggregated_histogram

def cluster_histogram(hist_data, k=10):
    # Extract non-zero indices and their corresponding frequencies
    nonzero_indices = np.nonzero(hist_data)[0]
    frequencies = hist_data[nonzero_indices]

    # Extract RGB values from the indices
    samples = np.array([[int(i/(256*256)), (i//256)%256, i%256] for i in nonzero_indices])

    # Use the frequencies as weights in the k-means clustering
    kmeans = KMeans(n_clusters=k).fit(samples, sample_weight=frequencies)

    return kmeans.cluster_centers_

def plot_color_histogram(cluster_centers):
    plt.figure(figsize=(12, 6))
    
    for i, color in enumerate(cluster_centers):
        plt.bar(str(i), 1, color=color/255.0)
    
    plt.xlabel('Cluster')
    plt.ylabel('Frequency')
    plt.title('Color Distribution')
    plt.show()

# Usage:
folder_path =  "C:\\Users\\tomei\\Documents\\Python\\Internship\\test_week_3\\generated_images"
bins, aggregated_histogram = aggregate_histograms(folder_path, keyword="cardboard")
cluster_centers = cluster_histogram(aggregated_histogram)
plot_color_histogram(cluster_centers)
