# Created using chatgpt-4
from PIL import Image
from collections import Counter
import pandas as pd
import os
from tqdm import tqdm

def count_colors(image_paths):
    color_counter = Counter()

    for path in tqdm(image_paths, desc="Processing images"):
        image = Image.open(path)
        # Convert the image to RGB if it's not
        image = image.convert('RGB')
        colors = image.getdata()
        color_counter.update(colors)

    return color_counter

# Example usage:
# Replace 'path_to_your_images' with the path to the directory containing your images.
image_folder = "C:\\Users\\tomei\\Documents\\Python\\image_similarity_clustering\\output_similar_images"
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Count colors across all images
color_counts = count_colors(image_files)

# Convert the color counts to a DataFrame
color_df = pd.DataFrame.from_records(list(color_counts.items()), columns=['Color', 'Count'])

# Sort the DataFrame based on the count, in descending order
color_df = color_df.sort_values(by='Count', ascending=False)

# Save the DataFrame to a CSV file
color_df.to_csv('color_counts.csv', index=False)
