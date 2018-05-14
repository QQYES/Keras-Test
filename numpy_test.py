import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)
sum2 = np.sum(a, axis=2)

print('sum0:{}'.format(sum0))
print('sum1:{}'.format(sum1))
print('sum2:{}'.format(sum2))
