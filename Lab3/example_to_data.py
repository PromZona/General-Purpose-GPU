import numpy as np

w = 8
h = 8

input = "D2E27500 CFF65200 D3ED5700 D6E76900 C8F35B00 8E168200 CFF45000 AE977600 D3DC7100 7D1E7B00 AB9A8000 D9E58600 AB967E00 AE9D8000 87058200 D0F95B00 AB967E00 AE9D8000 87058200 D0F95B00 74148000 D0F55900 86136C00 85077400 D6E27700 D3609F00 D1609F00 CC5EA100 CC739D00 7C127F00 AA988800 AFA07D00 D0E37700 7D117A00 D6EB5900 D6E37C00 C9F85700 D655A100 D7EA7400 93127D00 D35BA400 D4DD7900 B0A18400 D6DE7500 D765A900 AD928400 D0D87C00 D7E97F00 CD509E00 CAF85200 CFF75600 CEF45E00 D0E86900 D1D17F00 AD928100 AFA18300 D4DB5C00 88077D00 C6F75700 7D127D00 A99A8E00 C8609E00 D15DA500 AB957E00 AE9A8000 79218100 D065A100 A99E9A00"
data = np.fromiter((int(x, 16) for x in input.split(' ')), dtype='<u4')
print(data)
print(' '.join([hex(i) for i in data]))

result = np.append([w, h], data).astype('uint32')
file = open(f'{w}x{h}.data', 'wb')
file.write(result.tobytes())
file.close()
print(f'Created {w}x{h} image')