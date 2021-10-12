import numpy as np

f = open('out.data', 'rb')
raw_data = f.read()
f.close()

data = np.frombuffer(raw_data, dtype='<u4')
print(data)

print(' '.join([f'{i:08x}' for i in data]))