import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler


def build_model(input, num_dense_layers,list_nodes):
	input = tf.transpose(input, [0, 1])
	h1 = input
	print("zzz",np.shape(h1))

	for i in range(num_dense_layers):
		print(list_nodes[i+1])
		name = 'layer_dense_{0}'.format(i + 1)
		dense1 = tf.layers.dense(inputs=h1, name=name, units=list_nodes[i+1], activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
		h1 = dense1

	print('done')
	w6 = tf.get_variable("W_end", shape=[list_nodes[num_dense_layers],list_nodes[-1] ],initializer=tf.contrib.layers.xavier_initializer())
	b6 = tf.Variable(tf.zeros([list_nodes[-1]]), name="Bias_end", dtype=tf.float32)
	h6 = tf.matmul(h1, w6) + b6

	w_rwd = tf.get_variable("W_rwd", shape=[list_nodes[num_dense_layers],1 ],initializer=tf.contrib.layers.xavier_initializer())
	b_rwd = tf.Variable(tf.zeros([1]), name="Bias_rwd", dtype=tf.float32)
	h_rwd = tf.matmul(h1, w_rwd) + b_rwd

	return h6,h_rwd

train = 1
test = 1

def plotter(Y_pred,Y_truth,type):
	# print(len(Y_pred[0]))00
	stateinf = ['gamma','theta1','theta2','der_x','der_y','der_gamma','der_theta_1','der_theta_2']
	for i in range(len(stateinf)):
		fig, ax = plt.subplots(figsize=(7,3))
		agent_name=str('State information estimated = '+stateinf[i]+', Type = ' + type)
		# plt.subplot(12, 2, 2*i + j)
		ax.plot(Y_pred[:,i],label='NN prediction', linewidth=2, color='g')
		# plt.subplot(12, 2, 2*i + j)
		ax.plot(Y_truth[:,i],label='Ground Truth', linewidth=2, color='b')

		ax.grid(True, which="both", ls="-")
		# ax.set_ylim([-1,2])

		ax.set_xlabel('Steps')
		ax.set_ylabel(stateinf[i])
		ax.set_title(agent_name)

		ax.legend(loc='lower right')
		fig.savefig('Results/'+agent_name + '_comparison_NN.png', dpi=200, bbox_inches='tight')
		plt.show()

	# plt.tight_layout()
	
	

def train_network(list_nodes,train_data,train_label,test_data,test_label,valid_data,valid_label,train_rwd,test_rwd,val_rwd,dimension,learning_rate=0.0001,epochs=500,mini_batch_num=256):	
	
	tf.set_random_seed(seed=1)
	action_size = 2
	num_layers = len(list_nodes) + 2
	list_nodes = [dimension] + list_nodes + [dimension-action_size]

	# Features and Labels
	features = tf.placeholder(shape = [None, dimension],dtype = tf.float32)
	states = tf.placeholder(shape = [None, dimension-2],dtype = tf.float32)
	true_reward = tf.placeholder(tf.float32,[None,1],name="true_reward")
	output_pred, pred_rwd =  build_model(features,num_layers-2, list_nodes)
	logits = tf.reduce_mean(tf.square(output_pred - states) + tf.square(true_reward - pred_rwd))
	optimizer =tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(logits)
	
	init = tf.global_variables_initializer()

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	saver = tf.train.Saver()

	sess.run(init)
	
	if train:

		hyper_loss = 0
		hyper_valid_loss = 0

		for i in range(epochs):
			tr_loss = 0
			for j in range(int(len(train_data) / mini_batch_num)):
				batch_data = train_data[j * mini_batch_num:(j + 1) * mini_batch_num]
				batch_label = train_label[j * mini_batch_num:(j + 1) * mini_batch_num]
				batch_rwd = train_rwd[j * mini_batch_num:(j + 1) * mini_batch_num]

				# batch_reward = train_reward[j * mini_batch_num:(j + 1) * mini_batch_num]
				# batch_done = train_done[j * mini_batch_num:(j + 1) * mini_batch_num]

				tr_loss_part,_,_ = sess.run([logits,output_pred,optimizer],feed_dict={features:batch_data, states : batch_label, true_reward: batch_rwd})
				hyper_loss += (tr_loss_part ) / (int(len(train_data) / mini_batch_num))
				# print('o2',np.shape(O__2))

				tr_loss += tr_loss_part
			train_idx = list(range(len(train_data)))
			# print(train_idx)
			np.random.shuffle(train_idx)
			train_data = train_data[train_idx]
			train_label = train_label[train_idx]
			
			# train_reward = train_reward[train_idx]
			# train_done = train_done[train_idx]
			val_loss = 0
			for j in range(int(len(valid_data) / mini_batch_num)):
				batch_data = valid_data[j * mini_batch_num:(j + 1) * mini_batch_num]
				batch_label = valid_label[j * mini_batch_num:(j + 1) * mini_batch_num]
				batch_rwd = val_rwd[j * mini_batch_num:(j + 1) * mini_batch_num]
				# batch_reward = val_reward[j * mini_batch_num:(j + 1) * mini_batch_num]
				# batch_done = val_done[j * mini_batch_num:(j + 1) * mini_batch_num]

				# if(j==0):
				#  summary,val_loss_part,estimated=sess.run([merged,mse,P],feed_dict={O:batch_data,Rr:Rr_data,Rs:Rs_data,Ra:Ra_data,P_label:batch_label,X:X_data})
				#  writer.add_summary(summary,i)
				# else:
				val_loss_part,_,_ = sess.run([logits,output_pred,optimizer],feed_dict={features:batch_data, states : batch_label, true_reward:batch_rwd})

				hyper_valid_loss += (val_loss_part ) / (int(len(valid_data) / mini_batch_num))
				val_loss += val_loss_part
			val_idx = list(range(len(valid_data)))
			np.random.shuffle(val_idx)
			valid_data = valid_data[val_idx]
			valid_label = valid_label[val_idx]

			print("Epoch " + str(i + 1) + " Training MSE: " + str(
				tr_loss / (int(len(train_data) / mini_batch_num))) + " Validation MSE: " + str(
				val_loss / (int(len(valid_data) / mini_batch_num))))

		SAVE_PATH='./models/Graph_free_Model_based_Swimmer'
		saver.save(tf.get_default_session(), SAVE_PATH)

	if test:

		SAVE_PATH = './models/Graph_free_Model_based_Swimmer'
		saver.restore(tf.get_default_session(), SAVE_PATH)
		one_step=1
		recursive=1


		if one_step:

			test_loss = 0
			batch_data = test_data[:mini_batch_num]
			batch_label = test_label[:mini_batch_num]
			batch_rwd = test_rwd[:mini_batch_num]
			test_loss_part,estimated = sess.run([logits,output_pred],feed_dict={features:batch_data, states : batch_label,true_reward:batch_rwd})

			print('test loss: ',test_loss_part)
			plotter(estimated,batch_label,'one-step')

		if recursive:

			test_loss = 0
			batch_data = test_data[:1]
			batch_label = test_label[:1]
			batch_rwd = test_rwd[:1]
			# test_loss = 0
			true_out = []
			pred_out = []
			# start_obs=test_data[:1]
			# next_state= test_label[:1]
			for j in range(49):
				
				test_loss_part,estimated = sess.run([logits,output_pred],feed_dict={features:batch_data, states : batch_label, true_reward:batch_rwd})
				
				batch_data = [np.concatenate((estimated[0],test_data[j+1,6:8]))]
				batch_data =scaler.transform(batch_data)
				batch_label = [test_label[j+1]]

				test_loss+=test_loss_part

				true_out.append(test_label[j])
				pred_out.append(estimated[0])


			true_out=np.array(true_out)
			pred_out=np.array(pred_out)
			print('test loss: ',test_loss_part)
			print (np.shape(pred_out),np.shape(true_out))
			plotter(pred_out,true_out,'rec')

def load_data():

	set_num = 10000

	train_features = np.load('Data/train_input_s_a.npy')[:set_num]
	train_output = np.load('Data/train_output_s.npy')[:set_num]
	train_rwd = train_output[:,-1]
	train_rwd = np.reshape(train_rwd,(set_num,1))
	train_output = train_output[:,:-1]
	

	test_features = np.load('Data/test_input_s_a.npy')[:set_num]
	test_output = np.load('Data/test_output_s.npy')[:set_num]
	test_rwd = test_output[:,-1]
	test_rwd = np.reshape(test_rwd,(set_num,1))
	test_output = test_output[:,:-1]

	val_features = np.load('Data/val_input_s_a.npy')[:set_num]
	val_output = np.load('Data/val_output_s.npy')[:set_num]
	val_rwd = val_output[:,-1]
	val_rwd = np.reshape(val_rwd,(set_num,1))
	val_output = val_output[:,:-1]

	scaler=MinMaxScaler()
	norm_train_data=scaler.fit_transform(train_features)
	norm_valid_data=scaler.transform(test_features)
	norm_test_data=scaler.transform(val_features)

	return scaler,norm_train_data,train_output,norm_test_data,test_output,norm_valid_data,val_output,train_rwd,test_rwd,val_rwd


scaler,train_data,train_label,test_data,test_label,valid_data,valid_label,train_rwd,test_rwd,val_rwd = load_data()
train_network([128,128],train_data,train_label,test_data,test_label,valid_data,valid_label,train_rwd,test_rwd,val_rwd,dimension=10)