import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
#import sonnet as snt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler

def load_data():
	train_state_trans = np.load('train_state_trans.npy')
	train_actions = np.load('train_actions.npy')

	test_state_trans = np.load('test_state_trans.npy')
	test_actions = np.load('test_actions.npy')

	val_state_trans = np.load('val_state_trans.npy')
	val_actions = np.load('val_actions.npy')

	set_num = 1000

	state_train = train_state_trans[:int(.6 * set_num)]
	action_train = train_actions[:int(.6 * set_num)]

	state_test = test_state_trans[:int(.2* set_num)]
	action_test = test_actions[:int(.2 * set_num)]
	
	state_val = val_state_trans[:int(.2 * set_num)]
	action_val = val_actions[:int(.2 *set_num)]

	# print (np.shape(state_train))

	return state_train,action_train,state_test,action_test,state_val,action_val



def shape(norm_train_data,new_shape):

	train_data=np.reshape(norm_train_data,(new_shape[0],new_shape[1]*new_shape[2]))

	return train_data


def shape_inverse(norm_train_data,new_shape):
	print (new_shape)
	train_data=np.reshape(norm_train_data,(new_shape[0],new_shape[1],new_shape[2]))
	# train_data=np.transpose(train_data)
	return train_data

def shuffle_data(norm_train_data,edges_train):

	total_idx_train = list(range(len(norm_train_data)))
	np.random.shuffle(total_idx_train)
	norm_train_data=norm_train_data[total_idx_train]
	edges_train=edges_train[total_idx_train]

	return norm_train_data,edges_train





def preprocess(state_train,action_train,state_test,action_test,state_val,action_val):

	# set_num=50000
	# total_state=49

	train_data = state_train[:,0]
	valid_data = state_val[:,0]
	test_data = state_test[:,0]

	flat_train_data=shape_inverse(train_data,np.shape(train_data))
	flat_valid_data=shape_inverse(valid_data,np.shape(valid_data))
	flat_test_data=shape_inverse(test_data,np.shape(test_data))

	train_label = state_train[:,1]
	valid_label = state_val[:,1]
	test_label = state_test[:,1]

	flat_train_data=shape(flat_train_data,np.shape(train_data))
	flat_valid_data=shape(flat_valid_data,np.shape(valid_data))
	flat_test_data=shape(flat_test_data,np.shape(test_data))

	scaler = 1
	# scaler = StandardScaler()
	# scaler = Normalizer()
	scaler=MinMaxScaler()
	norm_train_data=scaler.fit_transform(flat_train_data)
	norm_valid_data=scaler.transform(flat_valid_data)
	norm_test_data=scaler.transform(flat_test_data)

	norm_train_data=shape_inverse(norm_train_data,np.shape(train_data))[:-1]
	norm_valid_data=shape_inverse(norm_valid_data,np.shape(valid_data))[:-1]
	norm_test_data=shape_inverse(norm_test_data,np.shape(test_data))[:-1]


	print(np.shape(train_label))
	print(np.shape(norm_train_data))

	return scaler,norm_train_data,norm_test_data,norm_valid_data,train_label,valid_label,test_label

def plotter(Y_pred,Y_truth):
	
	for i in range(Ds):
		for j in range(No):
			fig, ax = plt.subplots(figsize=(7,3))
			# ax.plot(trajectory[:200,0,index],label='MLP', linewidth=2, color='g')
			# ax.plot(test[:200,0,index], label='ground truth',linewidth=2, color='b')
			#
			agent_name='node: '+str('Feature = '+str(i)+','+'Object = '+str(j))
			# plt.subplot(12, 2, 2*i + j)
			ax.plot(Y_pred[:,i,j],label='GCN', linewidth=2, color='g')
			# plt.subplot(12, 2, 2*i + j)
			ax.plot(Y_truth[:,i,j],label='ground truth', linewidth=2, color='b')

			ax.grid(True, which="both", ls="-")
			# ax.set_ylim([-1,2])

			ax.set_xlabel('Steps')
			ax.set_ylabel('X')
			ax.set_title(agent_name)

			ax.legend(loc='lower right')
			fig.savefig(agent_name + '_comparison.png', dpi=200, bbox_inches='tight')

	plt.tight_layout()
	plt.show(fig)
	
	




Ds=4			#state rep to 3
No=3			#number of objects
Nr=No*(No-1)
Dr=1			#check
Dx=1			#action applied on each of the object, 0 for 1st and "angle change" for others
De=3			#latent rep of edge			
Dp=Ds
Da=1
max_epoches=100
mini_batch_num= 100

train=1
test=1


def m(O, Rr, Rs, Ra):
    # return tf.concat([(tf.matmul(O,Rr)-tf.matmul(O,Rs)),Ra],1)
    B = tf.concat([tf.matmul(O, Rr), tf.matmul(O, Rs), Ra], 1)
    return tf.concat([tf.matmul(O, Rr), tf.matmul(O, Rs), Ra], 1)


def phi_R(B, num_dense_layers_phi_R, num_dense_nodes_hb, De):
    h_size = num_dense_nodes_hb

    B_trans = tf.transpose(B, [0, 2, 1])

    B_trans = tf.reshape(B_trans, [-1, ( (2 * Ds + Dr))])
    # B_trans=tf.reshape(B,[-1,((2*Ds+Dr))])

    # w1 = tf.Variable(tf.truncated_normal([(Nr*(2*Ds+Dr)), h_size], stddev=0.1), name="r_w1", dtype=tf.float32)
    w1 = tf.get_variable("W1R", shape=[( (2 * Ds + Dr)), h_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([h_size]), name="r_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1)

    for i in range(num_dense_layers_phi_R):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_phi_R_{0}'.format(i + 1)
        dense1 = tf.layers.dense(inputs=h1, name=name, units=h_size, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        h1 = dense1

    w5 = tf.get_variable("W5R", shape=[h_size,  De],
                         initializer=tf.contrib.layers.xavier_initializer())
    # w5 = tf.Variable(tf.truncated_normal([h_size, Nr*De], stddev=0.1), name="r_w5", dtype=tf.float32)
    b5 = tf.Variable(tf.zeros([ De]), name="r_b5", dtype=tf.float32)
    h5 = tf.matmul(h1, w5) + b5

    h5_trans = tf.reshape(h5, [-1, Nr, De])
    h5_trans = tf.transpose(h5_trans, [0, 2, 1])



	# w6 = tf.get_variable("W5R", shape=[h_size,  1],
	#                  initializer=tf.contrib.layers.xavier_initializer())
	# # w5 = tf.Variable(tf.truncated_normal([h_size, Nr*De], stddev=0.1), name="r_w5", dtype=tf.float32)
	# b6 = tf.Variable(tf.zeros([ 1]), name="r_b6", dtype=tf.float32)
	# h6 = tf.matmul(h1, w6) + b6

	# h6_trans = tf.reshape(h6, [-1, Nr, 1])
	# h6_trans = tf.transpose(h6_trans, [0, 2, 1])

    return (h5_trans)


def a(O, Rr, X, E):
    E_bar = tf.matmul(E, tf.transpose(Rr, [0, 2, 1]))
    # O_2=tf.stack(tf.unstack(O,Ds,1),1)
    # O_2=tf.unstack(O,Ds,1)
    # return (tf.concat([O_2,X,E_bar],1))
    return (tf.concat([O, X, E_bar], 1))


def phi_O(C, num_dense_layers_phi_o, num_dense_nodes_hc, De):
    h_size = num_dense_nodes_hc
    C_trans = tf.transpose(C, [0, 2, 1])
    C_trans = tf.reshape(C_trans, [-1, ( (Ds + Dx + De))])

    # w1 = tf.Variable(tf.truncated_normal([(No*(Ds+Dx+De)), h_size], stddev=0.1), name="o_w1", dtype=tf.float32)
    w1 = tf.get_variable("W1O", shape=[( (Ds + Dx + De)), h_size],
                         initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.zeros([h_size]), name="o_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1)

    for i in range(num_dense_layers_phi_o):
        name = 'layer_dense_phi_o_{0}'.format(i + 1)
        dense1 = tf.layers.dense(inputs=h1, name=name, units=h_size, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        h1 = dense1

    # w6 = tf.Variable(tf.truncated_normal([h_size, No*Dp], stddev=0.1), name="o_w2", dtype=tf.float32)
    w6 = tf.get_variable("W6O", shape=[h_size,  Dp],
                         initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.zeros([ Dp]), name="o_b2", dtype=tf.float32)
    h6 = tf.matmul(h1, w6) + b6
    h6_trans = tf.reshape(h6, [-1, No, Dp])
    h6_trans = tf.transpose(h6_trans, [0, 2, 1])

    return h6_trans


def fitness(num_dense_layers_phi_R ,
num_dense_layers_phi_o ,
# num_dense_layers_phi_rew ,
num_dense_nodes_hb,
num_dense_nodes_hc,
# num_dense_nodes_hox,
            De):



	state_train,action_train,state_test,action_test,state_val,action_val = load_data()
	scaler,norm_train_data,norm_test_data,norm_valid_data,norm_train_label,norm_valid_label,norm_test_label = preprocess(state_train,action_train,state_test,action_test,state_val,action_val)



	train_data=norm_train_data
	valid_data=norm_valid_data
	test_data=norm_test_data

	train_label=norm_train_label
	valid_label=norm_valid_label
	test_label=norm_test_label

	train_X = action_train
	valid_X  = action_val
	test_X = action_test

	tf.set_random_seed(seed=1)

	# Object Matrix
	learning_rate=0.001
	O = tf.placeholder(tf.float32, [None, Ds, No], name="O")
	# Relation Matrics R=<Rr,Rs,Ra>
	Rr = tf.placeholder(tf.float32, [None, No, Nr], name="Rr")
	Rs = tf.placeholder(tf.float32, [None, No, Nr], name="Rs")
	Ra = tf.placeholder(tf.float32, [None, Dr, Nr], name="Ra")
	# next velocities
	P_label = tf.placeholder(tf.float32, [None, Dp, No], name="P_label")
	# External Effects
	X = tf.placeholder(tf.float32, [None, Dx, No], name="X")

	# reward_label = tf.placeholder(tf.float32, [None, Da], name="reward_label")

	# done_label = tf.placeholder(tf.float32, [None, Da], name="reward_label")

	# marshalling function, m(G)=B, G=<O,R>
	B = m(O, Rr, Rs, Ra)

	# relational modeling phi_R(B)=E
	E = phi_R(B, num_dense_layers_phi_R, num_dense_nodes_hb, De)

	# aggregator
	C = a(O, Rr, X, E)

	# object modeling phi_O(C)=P
	P = phi_O(C, num_dense_layers_phi_o, num_dense_nodes_hc, De)

	# q, d = phi_rew_done(O, X, num_dense_layers_phi_rew, num_dense_nodes_hox)

	mse = tf.reduce_mean(tf.square(P - P_label))

	loss = 0.001 * (tf.nn.l2_loss(E))

	# for i in params_list:
	#     loss += 0.001 * tf.nn.l2_loss(i)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	trainer = optimizer.minimize(mse+loss )


	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	saver = tf.train.Saver()


	Rr_data=np.zeros((int(mini_batch_num),No,Nr),dtype=float)
	Rs_data=np.zeros((int(mini_batch_num),No,Nr),dtype=float)
	Ra_data=np.ones((int(mini_batch_num),Dr,Nr),dtype=float)
	# X_data=np.zeros((mini_batch_num,Dx,No),dtype=float)



	cnt=0
	for i in range(No):
	    for j in range(No):
	      if(i!=j):
	        Rr_data[:,i,cnt]=1.0
	        Rs_data[:,j,cnt]=1.0
	        cnt+=1




	if train:

		hyper_loss = 0
		hyper_valid_loss = 0
		# max_epoches=150
		for i in range(max_epoches):
		    tr_loss = 0
		    for j in range(int(len(train_data) / mini_batch_num)):
		        batch_data = train_data[j * mini_batch_num:(j + 1) * mini_batch_num]
		        batch_label = train_label[j * mini_batch_num:(j + 1) * mini_batch_num]

		        batch_X = train_X[j * mini_batch_num:(j + 1) * mini_batch_num]

		        # batch_reward = train_reward[j * mini_batch_num:(j + 1) * mini_batch_num]
		        # batch_done = train_done[j * mini_batch_num:(j + 1) * mini_batch_num]

		        tr_loss_part, _ = sess.run([mse,trainer],
		                                                          feed_dict={O: batch_data, Rr: Rr_data, Rs: Rs_data,
		                                                                     Ra: Ra_data, P_label: batch_label, X: batch_X,
		                                                                 })
		        hyper_loss += (tr_loss_part ) / (int(len(train_data) / mini_batch_num))
		        # print('o2',np.shape(O__2))

		        tr_loss += tr_loss_part
		    train_idx = list(range(len(train_data)))
		    # print(train_idx)
		    np.random.shuffle(train_idx)
		    train_data = train_data[train_idx]
		    train_label = train_label[train_idx]
		    train_X = train_X[train_idx]
		    # train_reward = train_reward[train_idx]
		    # train_done = train_done[train_idx]
		    val_loss = 0
		    for j in range(int(len(valid_data) / mini_batch_num)):
		        batch_data = valid_data[j * mini_batch_num:(j + 1) * mini_batch_num]
		        batch_label = valid_label[j * mini_batch_num:(j + 1) * mini_batch_num]
		        batch_X = valid_X[j * mini_batch_num:(j + 1) * mini_batch_num]
		        # batch_reward = val_reward[j * mini_batch_num:(j + 1) * mini_batch_num]
		        # batch_done = val_done[j * mini_batch_num:(j + 1) * mini_batch_num]

		        # if(j==0):
		        #  summary,val_loss_part,estimated=sess.run([merged,mse,P],feed_dict={O:batch_data,Rr:Rr_data,Rs:Rs_data,Ra:Ra_data,P_label:batch_label,X:X_data})
		        #  writer.add_summary(summary,i)
		        # else:
		        val_loss_part,  estimated = sess.run([mse, P],
		                                                                   feed_dict={O: batch_data, Rr: Rr_data,
		                                                                              Rs: Rs_data, Ra: Ra_data,
		                                                                              P_label: batch_label, X: batch_X,
		                                                                             })
		        hyper_valid_loss += (val_loss_part ) / (int(len(valid_data) / mini_batch_num))
		        val_loss += val_loss_part
		    val_idx = list(range(len(valid_data)))
		    np.random.shuffle(val_idx)
		    valid_data = valid_data[val_idx]
		    valid_label = valid_label[val_idx]
		    valid_X = valid_X[val_idx]

		    print("Epoch " + str(i + 1) + " Training MSE: " + str(
		        tr_loss / (int(len(train_data) / mini_batch_num))) + " Validation MSE: " + str(
		        val_loss / (int(len(valid_data) / mini_batch_num))))

		SAVE_PATH='./models/planetary_4_IN'
		saver.save(tf.get_default_session(), SAVE_PATH)

	if test:

		SAVE_PATH = './models/planetary_4_IN'
		saver.restore(tf.get_default_session(), SAVE_PATH)
		one_step=1
		recursive=1


		if one_step:

					test_loss = 0
					output_true=[]
					output_pred=[]
					# for j in range(int(len(norm_test_data) / mini_batch_num)):
					for j in range(1):

						batch_data = test_data[j * mini_batch_num:(j + 1) * mini_batch_num]
						batch_label = test_label[j * mini_batch_num:(j + 1) * mini_batch_num]
						batch_X = valid_X[j * mini_batch_num:(j + 1) * mini_batch_num]

						# batch_A=edges_test[j * mini_batch_num:(j + 1) * mini_batch_num]

						test_loss_part,  estimated = sess.run([mse, P],feed_dict={O: batch_data, Rr: Rr_data,
		                                                                              Rs: Rs_data, Ra: Ra_data,
		                                                                              P_label: batch_label, X: batch_X,
		                                                                             })
						test_loss+=test_loss_part
						output_true.append(batch_label)
						output_pred.append(estimated)


					print('test loss: ',test_loss_part)
					print(np.shape(np.array(output_pred)))
					plotter(estimated,batch_label)

		if recursive:

					test_loss = 0
					output_true=[]
					output_pred=[]
					# for j in range(int(len(norm_test_data) / mini_batch_num)):
					# mini_batch_num=1
					start_obs=test_data[0]
					for j in range(49):

						# batch_data = test_data[j :(j + 1) ]
						# batch_label = test_label[j:(j + 1) ]
						# batch_X = valid_X[j:(j + 1) ]

						# batch_A=edges_test[j * mini_batch_num:(j + 1) * mini_batch_num]

						test_loss_part,  estimated = sess.run([mse, P],feed_dict={O: [start_obs], Rr: [Rr_data[j]],
		                                                                              Rs: [Rs_data[j]], Ra: [Ra_data[j]],
		                                                                              P_label: [test_label[j]], X: [test_X[j]],
		                                                                             })

						# print(np.shape(estimated))

						pred=np.reshape(estimated,(np.shape(estimated)[0],12))
						pred=scaler.transform(pred)
						pred=np.reshape(pred,(np.shape(pred)[0],4,3))
						start_obs=pred[0]

						test_loss+=test_loss_part

						output_true.append(test_label[j])
						output_pred.append(estimated[0])


					output_pred=np.array(output_pred)
					output_true=np.array(output_true)
					print('test loss: ',test_loss_part)
					print(np.shape((output_pred)))
					plotter(output_pred,output_true)


fitness(2 ,2,50, 50,1)

