import numpy as np
import matplotlib.pyplot as plt
img = plt.imread('jaimd.jpg')

plt.imshow(img)
plt.axis('off')
plt.show()

print(img.shape)
all_pixels = img.reshape((-1,3))
print(all_pixels)

all_pixels.shape
from sklearn.cluster import KMeans

k = 4
model = KMeans(n_clusters=k)
model.fit(all_pixels)

centers = model.cluster_centers_
centers = np.array(centers,dtype='uint8')
centers

for color in centers:
    box = np.zeros((100,100,3),dtype='uint8')
    box[:,:,:] = color

    plt.imshow(box)
    plt.axis("off")
    plt.show()
    
W,H,_ = img.shape
new_img = np.zeros((W*H,3),dtype='uint8')
centers
labels = model.labels_

for p in range(new_img.shape[0]):
    color_idx = labels[p]
    new_img[p] = centers[color_idx]

new_img = new_img.reshape((img.shape))
plt.imshow(new_img)
