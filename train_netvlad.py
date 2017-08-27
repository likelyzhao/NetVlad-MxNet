import mxnet as mx
import numpy as np


def _save_model(args, rank=0):
	import os
	if args.model_prefix is None:
		return None
	dst_dir = os.path.dirname(args.model_prefix)
	if not os.path.isdir(dst_dir):
		os.mkdir(dst_dir)
	return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
		args.model_prefix, rank))


class FeaDataIter(mx.io.DataIter):
	def __init__(self, num_classes, data_shape, max_iter, dtype):
		self.batch_size = data_shape[0]
		self.cur_iter = 0
		self.max_iter = max_iter
		self.dtype = dtype
		label = np.random.randint(0, num_classes, [self.batch_size,])
		data = np.random.uniform(-1, 1, data_shape)
		self.data = mx.nd.array(data, dtype=self.dtype)
		self.label = mx.nd.array(label, dtype=self.dtype)
	def __iter__(self):
		return self
	@property
	def provide_data(self):
		return [mx.io.DataDesc('data', self.data.shape, self.dtype)]
	@property
	def provide_label(self):
		return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]
	def next(self):
		self.cur_iter += 1
		if self.cur_iter <= self.max_iter:
			return mx.io.DataBatch(data=(self.data,),
							 label=(self.label,),
							 pad=0,
							 index=None,
							 provide_data=self.provide_data,
							 provide_label=self.provide_label)
		else:
			raise StopIteration
	def __next__(self):
		return self.next()
	def reset(self):
		self.cur_iter = 0




def netvlad(num_classes, num_centers, **kwargs):
	input_data = mx.symbol.Variable(name="data")
	input_centers = mx.symbol.Variable(name="centers")

	weights = mx.symbol.FullyConnected(name='w', data=input_data, num_hidden=num_centers)
	softmax_weights = mx.symbol.SoftmaxOutput(data=weights, axis=0,name='softmax')

	vari_lib =[]

	for i in range(num_centers):
		y = mx.symbol.slice_axis(input_centers,0,i,None)
		vari_lib.append(mx.symbol.broadcast_sub(input_data, y))

	for i in len(vari_lib):
		vari_lib[0] +=vari_lib[i]

	norm = mx.symbol.L2Normalization(vari_lib[0],mode='instance')
	norm = mx.symbol.Flatten(norm)
	norm = mx.symbol.L2Normalization(norm)

	group = mx.symbol.Group([norm , mx.symbol.BlockGrad(softmax_weights)])

	return group


def _load_model(args, rank=0):
	import os
	if 'load_epoch' not in args or args.load_epoch is None:
		return (None, None, None)
	assert args.model_prefix is not None
	model_prefix = args.model_prefix
	if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
		model_prefix += "-%d" % (rank)
	sym, arg_params, aux_params = mx.model.load_checkpoint(
		model_prefix, args.load_epoch)
 #   logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
	return (sym, arg_params, aux_params)



def train():
	optimizer = 'sgd'
	wd =0.05
	lr, lr_scheduler = _get_lr_scheduler(args, kv)
	optimizer_params = {
		'learning_rate': 0.01,
		'wd': wd,
		'lr_scheduler': lr_scheduler}
	load_epoch =0
	(train, val) = FeaDataIter(args, kv)

	gpus = 1,2,3,4
	top_k = 0
	batch_size =32
	disp_batches =40

	devs = mx.cpu() if gpus is None or gpus is '' else [
		mx.gpu(int(i)) for i in gpus.split(',')]

	checkpoint = _save_model(args, kv.rank)

	# create model
	model = mx.mod.Module(
		context=devs,
		symbol=netvlad
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

	argout = model.infer_shape()
	center = mx.nd.array()


	model.fit(train,
			  begin_epoch=load_epoch if load_epoch else 0,
			  num_epoch=10,
			  eval_data=val,
			  eval_metric=eval_metrics,
#			  kvstore=kv,
			  optimizer=optimizer,
			  optimizer_params=optimizer_params,
			  initializer=initializer,
			  arg_params=arg_params,
			  aux_params=aux_params,
			  batch_end_callback=batch_end_callbacks,
			  epoch_end_callback=checkpoint,
			  allow_missing=True,
			  monitor=monitor)



if __name__ is '__main__':
	train()

