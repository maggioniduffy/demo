
import matplotlib.pyplot as plt
import numpy as np

def plotLatentSpace(latentPoints, labels, generated = None):
    print('POINTS LEN: ', np.array(latentPoints).shape)
    x = [point[0] for point in latentPoints]
    y = [point[1] for point in latentPoints]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, cmap='viridis')

    if labels != None:
        for i in range(len(x)):
            plt.text(x=x[i] + 0.005, y=y[i] + 0.005, s=labels[i])
    
    if generated:
        for g in generated:
            aux = g[0]
            plt.plot(aux[0], aux[1], 'o', color='red')
            rect = [-1,1]
            ax = [aux[1],aux[1]]
            plt.plot(rect,ax,linestyle='--', color='black', linewidth = 0.2)
            ax = [aux[0],aux[0]]
            plt.plot(ax,rect,linestyle='--', color='black', linewidth = 0.2)
    plt.show()
    
def plotError(errorPoints):
    x = np.linspace(0, len(errorPoints), len(errorPoints))
    plt.plot(x, errorPoints)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()
    
def plotLetter(input,noised,predicted):
    plt.imshow(input, cmap='hot', interpolation='nearest')
    plt.title('INPUT')
    plt.show()

    plt.imshow(noised, cmap='hot', interpolation='nearest')
    plt.title('NOISED')
    plt.show()

    plt.imshow(predicted, cmap='hot', interpolation='nearest')
    plt.title('PREDICTED')
    plt.show()