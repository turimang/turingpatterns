import matplotlib.pyplot as plt
import json
import os

a = json.load(open(os.path.join('MCMCresults', 'a_values.json')))
b = json.load(open(os.path.join('MCMCresults','b_values.json')))


axes = plt.subplot(1, 2, 1)
axes.hist(a, bins=15)
axes.set_title('Distribution of parameter a')
axes.set_xlabel('Concentration')
plt.axvline(x=0.00028, c='red', label='Input value')
plt.legend()
axes = plt.subplot(1, 2, 2)
axes.set_title('Distribution of parameter b')
axes.set_xlabel('Concentration')
axes.hist(b, bins=15)
plt.axvline(x=0.005, c='red', label='Input value')
plt.legend()
plt.tight_layout()
plt.show()
