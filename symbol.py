import mxnet as mx
from config import config

def netvlad(feature_len, num_centers, num_output, **kwargs):
	input_data = mx.symbol.Variable(name="data")

	#        input_data = mx.symbol.BatchNorm(input_data)
	input_centers = mx.symbol.Variable(name="centers", shape=(num_centers, feature_len), init=mx.init.Normal(1))

	w = mx.symbol.Variable('weights_vlad',
	                       shape=[num_centers, feature_len], init=mx.initializer.Normal(0.1))
	b = mx.symbol.Variable('biases', shape=[1, num_centers], init=mx.initializer.Constant(1e-4))

	weights = mx.symbol.dot(name='w', lhs=input_data, rhs=w, transpose_b=True)
	weights = mx.symbol.broadcast_add(weights, b)

	softmax_weights = mx.symbol.softmax(data=weights, axis=2, name='softmax_vald')
	#	softmax_weights = mx.symbol.SoftmaxOutput(data=weights, axis=0,name='softmax_vald')

	vari_lib = []

	for i in range(num_centers):
		y = mx.symbol.slice_axis(name='slice_center_{}'.format(i), data=input_centers, axis=0, begin=i, end=i + 1)
		temp_w = mx.symbol.slice_axis(name='temp_w_{}'.format(i), data=softmax_weights, axis=2, begin=i, end=i + 1)
		element_sub = mx.symbol.broadcast_sub(input_data, y, name='cast_sub_{}'.format(i))
		vari_lib.append(mx.symbol.batch_dot(element_sub, temp_w, transpose_a=True, name='batch_dot_{}'.format(i)))

		#     group = mx.sym.Group(vari_lib)
	concat = []
	concat.append(vari_lib[0])
	# concat = mx.symbol.concat(data= vari_lib,dim=2,num_args=5,name = 'concat')
	for i in range(len(vari_lib) - 1):
		concat.append(mx.symbol.concat(concat[i], vari_lib[i + 1], dim=2, name='concat_{}'.format(i)))

	norm = mx.symbol.L2Normalization(concat[len(concat) - 1], mode='channel')
	norm = mx.symbol.Flatten(norm)
	#        norm = mx.symbol.max(input_data,axis =1)
	norm = mx.symbol.L2Normalization(norm)
	norm = mx.symbol.Dropout(norm, p=config.DROP_OUT_RATIO)

	weights_out = mx.symbol.FullyConnected(name='w_pre', data=norm, num_hidden=num_output)
	softmax_label = mx.symbol.SoftmaxOutput(data=weights_out, name='softmax')

	#	group = mx.symbol.Group([softmax_label, mx.symbol.BlockGrad(softmax_weights)])

	return softmax_label

def netvlad_mutil(feature_len, num_centers, num_output, **kwargs):
	input_data = mx.symbol.Variable(name="data")
	input_data_half = mx.symbol.Reshape(data=input_data, shape=(0, -1, feature_len / 2))
	input_data_four = mx.symbol.Reshape(data=input_data, shape=(0, -1, feature_len / 4))
	input_data_eight = mx.symbol.Reshape(data=input_data, shape=(0, -1, feature_len / 8))
	#        input_data = mx.symbol.BatchNorm(input_data)
	input_centers = mx.symbol.Variable(name="centers", shape=(num_centers / 2, feature_len), init=mx.init.Normal(1))

	w = mx.symbol.Variable('weights_vlad',
	                       shape=[num_centers / 2, feature_len], init=mx.initializer.Normal(0.1))
	b = mx.symbol.Variable('biases', shape=[1, num_centers / 2], init=mx.initializer.Constant(1e-4))

	weights = mx.symbol.dot(name='w', lhs=input_data, rhs=w, transpose_b=True)
	weights = mx.symbol.broadcast_add(weights, b)

	softmax_weights = mx.symbol.softmax(data=weights, axis=2, name='softmax_vald')
	#	softmax_weights = mx.symbol.SoftmaxOutput(data=weights, axis=0,name='softmax_vald')

	vari_lib = []

	for i in range(num_centers / 2):
		y = mx.symbol.slice_axis(name='slice_center_{}'.format(i), data=input_centers, axis=0, begin=i, end=i + 1)
		temp_w = mx.symbol.slice_axis(name='temp_w_{}'.format(i), data=softmax_weights, axis=2, begin=i, end=i + 1)
		element_sub = mx.symbol.broadcast_sub(input_data, y, name='cast_sub_{}'.format(i))
		vari_lib.append(mx.symbol.batch_dot(element_sub, temp_w, transpose_a=True, name='batch_dot_{}'.format(i)))

		#     group = mx.sym.Group(vari_lib)
		concat = []
		concat.append(vari_lib[0])
	# concat = mx.symbol.concat(data= vari_lib,dim=2,num_args=5,name = 'concat')
	for i in range(len(vari_lib) - 1):
		concat.append(mx.symbol.concat(concat[i], vari_lib[i + 1], dim=2, name='concat_{}'.format(i)))

		netvlad_ori = concat[len(concat) - 1]

	netvlad_ori = mx.symbol.L2Normalization(netvlad_ori, mode='channel')
	netvlad_ori = mx.symbol.Reshape(data=netvlad_ori, shape=(0, -1, feature_len))
	#################################original part ####################
	input_centers_half = mx.symbol.Variable(name="centers_half", shape=(num_centers / 2, feature_len / 2),
	                                        init=mx.init.Normal(1))

	w_half = mx.symbol.Variable('weights_vlad_half',
	                            shape=[num_centers / 2, feature_len / 2], init=mx.initializer.Normal(0.1))
	b_half = mx.symbol.Variable('biases_half', shape=[1, num_centers / 2], init=mx.initializer.Constant(1e-4))

	weights_half = mx.symbol.dot(name='w_half', lhs=input_data_half, rhs=w_half, transpose_b=True)
	weights_half = mx.symbol.broadcast_add(weights_half, b_half)

	softmax_weights_half = mx.symbol.softmax(data=weights_half, axis=2, name='softmax_vald_half')
	#       softmax_weights = mx.symbol.SoftmaxOutput(data=weights, axis=0,name='softmax_vald')

	vari_lib_half = []
	for i in range(num_centers / 2):
		y_half = mx.symbol.slice_axis(name='slice_center_half_{}'.format(i), data=input_centers_half, axis=0, begin=i,
		                              end=i + 1)
		temp_w_half = mx.symbol.slice_axis(name='temp_w_half{}'.format(i), data=softmax_weights_half, axis=2, begin=i,
		                                   end=i + 1)
		element_sub_half = mx.symbol.broadcast_sub(input_data_half, y_half, name='cast_sub_half_{}'.format(i))
		vari_lib_half.append(
			mx.symbol.batch_dot(element_sub_half, temp_w_half, transpose_a=True, name='batch_dot_half_{}'.format(i)))

	# group = mx.sym.Group(vari_lib)
	concat_half = []
	concat_half.append(vari_lib_half[0])
	#        concat = mx.symbol.concat(data= vari_lib,dim=2,num_args=5,name = 'concat')
	for i in range(len(vari_lib_half) - 1):
		concat_half.append(
			mx.symbol.concat(concat_half[i], vari_lib_half[i + 1], dim=2, name='concat_half{}'.format(i)))

	netvlad_half = concat_half[len(concat_half) - 1]
	netvlad_half = mx.symbol.L2Normalization(netvlad_half, mode='channel')
	netvlad_half = mx.symbol.Reshape(data=netvlad_half, shape=(0, -1, feature_len))
	########################## 1/4 feature size #############################

	input_centers_four = mx.symbol.Variable(name="centers_four", shape=(num_centers / 2, feature_len / 4),
	                                        init=mx.init.Normal(1))

	w_four = mx.symbol.Variable('weights_vlad_four',
	                            shape=[num_centers / 2, feature_len / 4], init=mx.initializer.Normal(0.1))
	b_four = mx.symbol.Variable('biases_four', shape=[1, num_centers / 2], init=mx.initializer.Constant(1e-4))

	weights_four = mx.symbol.dot(name='w_four', lhs=input_data_four, rhs=w_four, transpose_b=True)
	weights_four = mx.symbol.broadcast_add(weights_four, b_four)

	softmax_weights_four = mx.symbol.softmax(data=weights_four, axis=2, name='softmax_vald_four')
	#       softmax_weights = mx.symbol.SoftmaxOutput(data=weights, axis=0,name='softmax_vald')

	vari_lib_four = []
	for i in range(num_centers / 2):
		y_four = mx.symbol.slice_axis(name='slice_center_four_{}'.format(i), data=input_centers_four, axis=0, begin=i,
		                              end=i + 1)
		temp_w_four = mx.symbol.slice_axis(name='temp_w_four{}'.format(i), data=softmax_weights_four, axis=2, begin=i,
		                                   end=i + 1)
		element_sub_four = mx.symbol.broadcast_sub(input_data_four, y_four, name='cast_sub_four_{}'.format(i))
		vari_lib_four.append(
			mx.symbol.batch_dot(element_sub_four, temp_w_four, transpose_a=True, name='batch_dot_four_{}'.format(i)))

	# group = mx.sym.Group(vari_lib)
	concat_four = []
	concat_four.append(vari_lib_four[0])
	#        concat = mx.symbol.concat(data= vari_lib,dim=2,num_args=5,name = 'concat')
	for i in range(len(vari_lib_four) - 1):
		concat_four.append(
			mx.symbol.concat(concat_four[i], vari_lib_four[i + 1], dim=2, name='concat_four{}'.format(i)))

	netvlad_four = concat_four[len(concat_four) - 1]
	netvlad_four = mx.symbol.L2Normalization(netvlad_four, mode='channel')
	netvlad_four = mx.symbol.Reshape(data=netvlad_four, shape=(0, -1, feature_len))
	######################## 1/8 feature ##################################

	input_centers_eight = mx.symbol.Variable(name="centers_eight", shape=(num_centers, feature_len / 8),
	                                         init=mx.init.Normal(1))

	w_eight = mx.symbol.Variable('weights_vlad_eight',
	                             shape=[num_centers, feature_len / 8], init=mx.initializer.Normal(0.1))
	b_eight = mx.symbol.Variable('biases_eight', shape=[1, num_centers], init=mx.initializer.Constant(1e-4))

	weights_eight = mx.symbol.dot(name='w_eight', lhs=input_data_eight, rhs=w_eight, transpose_b=True)
	weights_eight = mx.symbol.broadcast_add(weights_eight, b_eight)

	softmax_weights_eight = mx.symbol.softmax(data=weights_eight, axis=2, name='softmax_vald_eight')
	#       softmax_weights = mx.symbol.SoftmaxOutput(data=weights, axis=0,name='softmax_vald')

	vari_lib_eight = []
	for i in range(num_centers):
		y_eight = mx.symbol.slice_axis(name='slice_center_eight_{}'.format(i), data=input_centers_eight, axis=0,
		                               begin=i, end=i + 1)
		temp_w_eight = mx.symbol.slice_axis(name='temp_w_eight{}'.format(i), data=softmax_weights_eight, axis=2,
		                                    begin=i, end=i + 1)
		element_sub_eight = mx.symbol.broadcast_sub(input_data_eight, y_eight, name='cast_sub_eight_{}'.format(i))
		vari_lib_eight.append(
			mx.symbol.batch_dot(element_sub_eight, temp_w_eight, transpose_a=True, name='batch_dot_eight_{}'.format(i)))

	# group = mx.sym.Group(vari_lib)
	concat_eight = []
	concat_eight.append(vari_lib_eight[0])
	#        concat = mx.symbol.concat(data= vari_lib,dim=2,num_args=5,name = 'concat')
	for i in range(len(vari_lib_eight) - 1):
		concat_eight.append(
			mx.symbol.concat(concat_eight[i], vari_lib_eight[i + 1], dim=2, name='concat_eight{}'.format(i)))

	netvlad_eight = concat_eight[len(concat_eight) - 1]

	netvlad_eight = mx.symbol.L2Normalization(netvlad_eight, mode='channel')
	netvlad_eight = mx.symbol.Reshape(data=netvlad_eight, shape=(0, -1, feature_len))

	norm = mx.symbol.concat(netvlad_ori, netvlad_half, netvlad_four, netvlad_eight, dim=1, name='total_concat')

	#        return norm

	# norm = mx.symbol.Flatten(norm)
	# norm = mx.symbol.L2Normalization(norm,mode='channel')
	norm = mx.symbol.Flatten(norm)
	#        norm = mx.symbol.max(input_data,axis =1)
	norm = mx.symbol.L2Normalization(norm)
	norm = mx.symbol.Dropout(norm, p=config.DROP_OUT_RATIO)

	weights_out = mx.symbol.FullyConnected(name='w_pre', data=norm, num_hidden=num_output)
	softmax_label = mx.symbol.SoftmaxOutput(data=weights_out, name='softmax')

	#	group = mx.symbol.Group([softmax_label, mx.symbol.BlockGrad(softmax_weights)])

	return softmax_label