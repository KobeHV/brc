import scipy.io as sio
import numpy as np

# python保存.mat文件
save_fn = './restore/a.mat'


save_array = np.array([1, 2, 3, 4])
print(save_array)
sio.savemat(save_fn, {'array': save_array})  # 和上面的一样，存在了array变量的第一行

# python加载.mat文件
load_fn = './restore/a.mat'
load_data = sio.loadmat(load_fn)
