import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from get_data import data
from utils import get_winner_neuron

def plot_map(k,grid,countries):
    values = np.zeros((k,k), int)
    
    index = 0
    for entrie in data:
        min_position = (None,None)
        min_dist = 999
        for row in grid:
            for col in row:
                w = col.weights
                dist = np.linalg.norm(entrie-w)
                if dist < min_dist:
                    min_position = col.position
                    min_dist = dist
        grid[min_position[0], min_position[1]].count += 1
        grid[min_position[0], min_position[1]].add_element(countries[index])
        values[min_position[0], min_position[1]] += 1
        index += 1

    fig, ax = plt.subplots(figsize=(15,10))

    i = 0
    for col in grid:
        for j in range(len(col)):
            print('Neurona (',i,',',j,') tiene a: ', grid[i][j].elements)
            label = ''
            for e in grid[i][j].elements:
                label = label + str(e) + '\n'
            ax.text(j + 0.1,i  + 0.75, label, color='w')
        i += 1
                
    sns.heatmap(values, annot=True, ax=ax, cmap='viridis')
    plt.savefig('map.png')
    plt.show(block=False)

def get_neighbors(i,j):
  return [(i,j+1), (i+1,j), (i+1,j+1), (i,j-1), (i-1,j), (i-1,j-1), (i-1, j+1), (i+1, j-1)]

def plot_u_matrix(k,grid):
    u_values = np.zeros((k,k),float)
    plt.figure(figsize=(15,10))
    
    for i in range(k):
        for j in range(k):
            aux = np.linalg.norm(grid[i,j].weights)
            w = grid[i,j].weights / aux
            neighbors = get_neighbors(i,j)
            distances = []
            for n in neighbors:
                x, y = n[0], n[1]
                if x >= 0 and y >= 0 and x < k and y < k:
                    aux = np.linalg.norm(np.array(grid[x,y].weights))
                    neighbor_neuron_w = grid[x,y].weights / aux
                    dist = np.linalg.norm(w - neighbor_neuron_w)
                    distances.append(dist)
                    
            u_values[i][j] = np.mean(distances)

    sns.heatmap(u_values, annot=True, cmap='viridis')

    plt.savefig('matriz_u.png')
    plt.show()