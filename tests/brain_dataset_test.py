import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import brain_dataset
import matplotlib.pyplot as plt
from matplotlib import colors

train_set = brain_dataset.BrainImageSet('dataset/training_images', 'dataset/training_labels')

train_batch_size = 4
images, labels = train_set.get_random_batch(train_batch_size)

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))

for i in range(train_batch_size):
    image = images[i][0]
    label = labels[i]
    # Display images
    axes[0, i].imshow(image, cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title('Image {}'.format(i + 1))

    # Display label maps
    axes[1, i].imshow(label, cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']))
    axes[1, i].axis('off')
    axes[1, i].set_title('Label Map {}'.format(i + 1))

plt.show()