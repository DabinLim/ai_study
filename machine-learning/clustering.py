import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')

apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

# plt.hist(np.mean(apple, axis=1), alpha=0.8)  # type: ignore
# plt.hist(np.mean(pineapple, axis=1), alpha=0.8)  # type: ignore
# plt.hist(np.mean(banana, axis=1), alpha=0.8)  # type: ignore
# plt.legend(['apple', 'pineapple', 'banana'])
# plt.show()
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))  # type: ignore
axs[1].bar(range(10000), np.mean(pineapple, axis=0))  # type: ignore
axs[2].bar(range(10000), np.mean(banana, axis=0))  # type: ignore
# plt.show()
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))

banana_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[banana_index[i*10 + j]],  # type: ignore
                         cmap='gray_r')
        axs[i, j].axis('off')  # type: ignore

plt.show()
