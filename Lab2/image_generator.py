import numpy as np

w = 1000
h = 1000
count = w * h
data = np.random.rand(count)
power = 2 ** 32
data *= power
data = data.astype('uint32')
result = np.append([w, h], data).astype('uint32')
file = open(f'{w}x{h}.data', 'wb')
file.write(result.tobytes())
file.close()
print(f'Created {w}x{h} image')