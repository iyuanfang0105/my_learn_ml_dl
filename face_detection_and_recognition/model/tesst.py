import pickle
import random
import matplotlib.pyplot as plt

file_path = 'data/wind.plk'
with open(file_path, 'rb') as fin:
    wind_faces = pickle.load(fin)

fig = plt.figure(figsize=(15, 15))
for i in range(0, 25):
    fig.add_subplot(5, 5, i+1)
    plt.imshow(random.choice(wind_faces)['face'])
plt.show()