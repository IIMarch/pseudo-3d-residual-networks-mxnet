import mxnet as mx
import numpy as np

name = 'prob'
sym, arg_params, aux_params = mx.model.load_checkpoint('./deploy_p3d_resnet_sports1m', 0)
sym1 = sym.get_internals()['{}_output'.format(name)]
sym2 = sym.get_internals()['res2a_branch1_output']
sym = mx.sym.Group([sym1,sym2])
model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=['prob_label'])
model.bind(data_shapes=[('data', (1, 3, 16, 160, 160))], label_shapes=[('prob_label', (1,))])
model.set_params(arg_params=arg_params, aux_params=aux_params)
input_data = np.load('./input_data.npy')
output_data = np.load('./{}_data.npy'.format('prob'))
input_data[:, [0,2], :,:,:] = input_data[:, [2,0], :,:,:]

batch = mx.io.DataBatch(
    data = [mx.nd.array(input_data)],
    provide_data= [mx.io.DataDesc('data', (1,3,16,160,160))],
    )
model.forward(batch, is_train=False)

outputs = [output.asnumpy() for output in model.get_outputs()]
print np.sum(np.abs(outputs[0] - output_data))

