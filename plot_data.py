import matplotlib.pyplot as plt

def plot(U, i):
    plt.imshow(U, cmap=plt.cm.copper,
              interpolation='bilinear',
              extent=[-1, 1, -1, 1])
    plt.title(f'$t={i:.2f}$')
    plt.show()

