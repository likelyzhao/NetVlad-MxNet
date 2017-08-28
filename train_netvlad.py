import mxnet as mx
import numpy as np
import os

from easydict import EasyDict as edict

config = edict()
config.NUM_VLAD_CENTERS =10
config.NUM_LABEL =500
config.LEARNING_RATE =0.1
config.FeaLen = 4096


def _save_model(model_prefix, rank=0):
	import os
	if model_prefix is None:
		return None
	dst_dir = os.path.dirname(model_prefix)
	if not os.path.isdir(dst_dir):
		os.mkdir(dst_dir)
	return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d" % (
		model_prefix, rank))


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor


class FeaDataIter(mx.io.DataIter):
	def __init__(self, filelist, batchsize, ctx, num_classes, data_shape, dtype = 'float32', work_load_list =None):
		self.batch_size = batchsize
		self.cur_iter = 0
#		self.max_iter = max_iter
		self.dtype = dtype
		self.ctx = ctx
		self.work_load_list = work_load_list
		self.featuredb =[]
		if not os.exsit(filelist):
			raise Exception('Sorry, filelist {} not exsit.'.format(filelist))
		f = open(filelist)
		self.featuredb = f.readlines()
		f.close()
		self.max_iter = len(self.featuredb)
		self.num_classes = num_classes

#		label = np.random.randint(0, num_classes, [self.batch_size,])
#		data = np.random.uniform(-1, 1, data_shape)
#		self.data = mx.nd.array(data, dtype=self.dtype)
#		self.label = mx.nd.array(label, dtype=self.dtype)
	def __iter__(self):
		return self
	@property
	def provide_data(self):
		return [mx.io.DataDesc('data', self.data.shape, self.dtype)]
	@property
	def provide_label(self):
		return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]

	def iter_next(self):
		return self.cur + self.batch_size <= self.size

	def next(self):
		if self.iter_next():
			self.get_batch()
			self.cur += self.batch_size
#			return self.im_info, \
			return  mx.io.DataBatch(data=self.data, label=self.label,
			                       pad=self.getpad(), index=self.getindex(),
			                       provide_data=self.provide_data, provide_label=self.provide_label)
		else:
			raise StopIteration

	def __next__(self):
		return self.next()
	def reset(self):
		self.cur_iter = 0


	def get_data_label(self,iroidb):
		label_array =[]
		data_array =[]
		for line in iroidb:
			datapath  = line.split('/t')[0]
			label_array.appen(line.split("/t")[1])
			data_array.append(np.formfile(datapath,dtype='float').reshape(-1,config.FeaLen))

		return data_array,label_array



	def get_batch(self):
		# slice roidb
		cur_from = self.cur
		cur_to = min(cur_from + self.batch_size, self.size)
		roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

		# decide multi device slice
		work_load_list = self.work_load_list
		ctx = self.ctx
		if work_load_list is None:
			work_load_list = [1] * len(ctx)
		assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
			"Invalid settings for work load. "
		slices = mx.executor_manager._split_input_slice(self.batch_size, work_load_list)

		# get testing data for multigpu
		# each element in the list is the data used by different gpu
		data_list = []
		label_list = []
		for islice in slices:
			iroidb = [self.featuredb[i] for i in range(islice.start, islice.stop)]
			data, label = self.get_data_label(iroidb)
			data_list.append(data)
			label_list.append(label)

		# pad data first and then assign anchor (read label)
		data_tensor = tensor_vstack([batch for batch in data_list])
		label_tensor = tensor_vstack([batch for batch in data_list])

		self.data = mx.nd.array(data_tensor)
		self.label = mx.nd.array(label_tensor)


def netvlad(num_centers, num_output,**kwargs):
	input_data = mx.symbol.Variable(name="data")
	input_centers = mx.symbol.Variable(name="centers")

	weights = mx.symbol.FullyConnected(name='w', data=input_data, num_hidden=num_centers)
	softmax_weights = mx.symbol.Softmax(data=weights, axis=0,name='softmax')

	vari_lib =[]

	for i in range(num_centers):
		y = mx.symbol.slice_axis(input_centers,0,i,None)
		temp_w = mx.symbol.slice_axis(softmax_weights,0,i,None)
		element_sub = mx.symbol.broadcast_sub(input_data, y)
		vari_lib.append(mx.symbol.broadcast_mul(element_sub, temp_w))

	for i in len(vari_lib):
		vari_lib[0] +=vari_lib[i]

	norm = mx.symbol.L2Normalization(vari_lib[0],mode='instance')
	norm = mx.symbol.Flatten(norm)
	norm = mx.symbol.L2Normalization(norm)

	weights = mx.symbol.FullyConnected(name='w', data=norm, num_hidden=num_output)
	softmax_label = mx.symbol.SoftmaxOutput(data=weights, axis=0,name='softmax_label')

	group = mx.symbol.Group([softmax_label, mx.symbol.BlockGrad(softmax_weights)])

	return group


def _load_model(model_prefix,load_epoch,rank=0):
	import os
	assert model_prefix is not None
	if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
		model_prefix += "-%d" % (rank)
	sym, arg_params, aux_params = mx.model.load_checkpoint(
		model_prefix, load_epoch)
 #   logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
	return (sym, arg_params, aux_params)

def _get_lr_scheduler(lr, lr_factor=None, begin_epoch = 0 ,lr_step_epochs='',epoch_size=0):
	if not lr_factor or lr_factor >= 1:
		return (lr, None)

	step_epochs = [int(l) for l in lr_step_epochs.split(',')]
	adjustlr =lr
	for s in step_epochs:
		if begin_epoch >= s:
			adjustlr *= lr_factor
	if lr != adjustlr:
		logging.info('Adjust learning rate to %e for epoch %d' % (lr, begin_epoch))

	steps = [epoch_size * (x - begin_epoch) for x in step_epochs if x - begin_epoch > 0]
	return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor))



def train():
	kv_store = 'device'
	# kvstore
	kv = mx.kvstore.create(kv_store)

	model_prefix = 'model/netvlad'
	optimizer = 'sgd'
	wd =0.05


	load_epoch =0

	train_data = FeaDataIter("training path")
	val_data  = FeaDataIter("val path")

	lr, lr_scheduler = _get_lr_scheduler(config.LEARNING_RATE, 0.1,0,'2,5',train_data.total)

	optimizer_params = {
		'learning_rate': lr,
		'wd': wd,
		'lr_scheduler': lr_scheduler}

	gpus = 1,2,3,4
	top_k = 0
	batch_size =32
	disp_batches =40

	devs = mx.cpu() if gpus is None or gpus is '' else [
		mx.gpu(int(i)) for i in gpus.split(',')]

	checkpoint = _save_model(model_prefix, kv.rank)

	# create model
	model = mx.mod.Module(
		context=devs,
		symbol=netvlad(config.NUM_VLAD_CENTERS,config.NUM_LABEL)
	)

	initializer = mx.init.Xavier(
		rnd_type='gaussian', factor_type="in", magnitude=2)


	eval_metrics = ['accuracy']
	if top_k > 0:
		eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=top_k))

	if optimizer == 'sgd':
		optimizer_params['multi_precision'] = True

	batch_end_callbacks = [mx.callback.Speedometer(batch_size, disp_batches)]

#	monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None
	monitor = None

	data_shape_dict = dict(train_data.provide_data + train_data.provide_label)

	arg_shape, out_shape, aux_shape = model[1].infer_shape(**data_shape_dict)
	fea_len = out_shape[1]
	center = mx.nd.array(config.NUM_VLAD_CENTERS,fea_len)

#	arg_shape_dict = dict(zip(train_data.list_arguments(), arg_shape))
#	out_shape_dict = dict(zip(train_data.list_outputs(), out_shape))
#	aux_shape_dict = dict(zip(train_data.list_auxiliary_states(), aux_shape))

#	sym,arg_params,aux_params = _load_model()


	model.fit(train_data,
			  begin_epoch=load_epoch if load_epoch else 0,
			  num_epoch=10,
			  eval_data=val_data,
			  eval_metric=eval_metrics,
			  kvstore=kv_store,
			  optimizer=optimizer,
			  optimizer_params=optimizer_params,
			  initializer=initializer,
			  arg_params=None,
			  aux_params=None,
			  batch_end_callback=batch_end_callbacks,
			  epoch_end_callback=checkpoint,
			  allow_missing=True,
			  monitor=monitor)



if __name__ is '__main__':
	train()

