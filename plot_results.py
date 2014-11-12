import matplotlib.pyplot as plt
from test_results import *

x = c_130723_3
y = scores_130723_3

max     = max(y)
max_i   = y.index(max)
best_x  = x[max_i]

print(max)
print(best_x)

plt.plot(x,y)
plt.plot(best_x, max, 'ro')
plt.ylabel('AUC')
plt.xlabel('C (strength of regularisation)')
plt.title('AUC score by C value (for feature selected, 7 level interactions)')
plt.show()