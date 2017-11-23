import caffe
import numpy as np

net = caffe.Net('./deploy_p3d_resnet_sports1m.prototxt',
        './p3d_resnet_sports1m_iter_150000.caffemodel', caffe.TEST)
np.random.seed(1)
input_data = np.random.rand(1,3,16,160,160)
net.blobs['data'].data[:] = input_data
np.save('input_data.npy', input_data)
net.forward()

for data in net.blobs:
    np_data = net.blobs[data].data
    np.save('{}_data.npy'.format(data), np_data)

asd = 0
