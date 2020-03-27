import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env

class Model:
    
    def __init__(self, environment, num_dense_layers = 2, lr_Adam = 0.001, list_dense_nodes = [128,128]):
        # Parameters.
        self.num_dense_layers = num_dense_layers
        self.environment = environment
        self.list_dense_nodes = list_dense_nodes
        self.num_layers = self.num_dense_layers + 2
        env = gym.make(self.environment)
        self.input_dim = len(env.reset()) + len(env.action_space.sample())
        self.output_dim = len(env.reset())
        self.list_nodes = [self.input_dim] + self.list_dense_nodes + [self.output_dim]
        self.learning_rate = lr_Adam

        # Features and Labels
        self.features = tf.placeholder(shape = [None, self.input_dim],dtype = tf.float32)
        self.states = tf.placeholder(shape = [None, self.output_dim],dtype = tf.float32)
        self.true_reward = tf.placeholder(tf.float32,[None,1],name="true_reward")
        self.output_pred, self.pred_rwd =  self.build_model(self.features, self.list_nodes)
        self.logits = tf.reduce_mean(tf.square(self.output_pred - self.states) + tf.square(self.true_reward - self.pred_rwd))
        self.optimizer =tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.logits)
        
        tf.set_random_seed(seed=1)
        
        self.init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.sess.run(self.init)

    def build_model(self,input, list_nodes):
        input = tf.transpose(input, [0, 1])
        h1 = input
    
        for i in range(self.num_dense_layers):
            name = 'layer_dense_{0}'.format(i + 1)
            dense1 = tf.layers.dense(inputs=h1, name=name, units=list_nodes[i+1], activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            h1 = dense1

        print('done')
        w6 = tf.get_variable("W_end", shape=[list_nodes[self.num_dense_layers],list_nodes[-1] ],initializer=tf.contrib.layers.xavier_initializer())
        b6 = tf.Variable(tf.zeros([list_nodes[-1]]), name="Bias_end", dtype=tf.float32)
        h6 = tf.matmul(h1, w6) + b6

        w_rwd = tf.get_variable("W_rwd", shape=[list_nodes[self.num_dense_layers],1 ],initializer=tf.contrib.layers.xavier_initializer())
        b_rwd = tf.Variable(tf.zeros([1]), name="Bias_rwd", dtype=tf.float32)
        h_rwd = tf.matmul(h1, w_rwd) + b_rwd

        return h6,h_rwd

    def train_network(self, train_data, train_label, train_rwd, algorithm_id = 'gen_model', learning_rate=0.0001, epochs=500, mini_batch_num=50):	
        
        saver = tf.train.Saver()

        scaler = MinMaxScaler()
        train_data=scaler.fit_transform(train_data)

        tr_loss = 0
        for i in range(epochs):
            for j in range(int(len(train_data) / mini_batch_num)):
                batch_data = train_data[j * mini_batch_num:(j + 1) * mini_batch_num]
                batch_label = train_label[j * mini_batch_num:(j + 1) * mini_batch_num]
                batch_rwd = train_rwd[j * mini_batch_num:(j + 1) * mini_batch_num]

                tr_loss_part,_,_ = self.sess.run([self.logits,self.output_pred,self.optimizer],feed_dict={self.features:batch_data, self.states : batch_label, self.true_reward: batch_rwd})
                tr_loss += tr_loss_part

            train_idx = list(range(len(train_data)))
            np.random.shuffle(train_idx)
            train_data = train_data[train_idx]
            train_label = train_label[train_idx]

        SAVE_PATH='./models/NN_Model_based_Swimmer_' + algorithm_id
        saver.save(tf.get_default_session(), SAVE_PATH)

        return tr_loss


    def model_step(self, obs, action):
        s_a_input = np.hstack((obs,action))
        next_state, rwd =  self.sess.run([self.output_pred,self.pred_rwd],feed_dict={self.features:[s_a_input]})
        return next_state[0],rwd[0,0],0            #done = 0 for specific envts only



'''
How to use this Model
'''
# set_num = 1000
# train_features = np.load('Data/train_input_s_a.npy')[:set_num]
# train_output = np.load('Data/train_output_s.npy')[:set_num]
# train_rwd = train_output[:,-1]
# train_rwd = np.reshape(train_rwd,(set_num,1))
# train_output = train_output[:,:-1]

# x = Model(environment = 'Swimmer-v2')
# x.train_network(train_features,train_output,train_rwd)
# print (x.model_step(train_features[0,0:8],train_features[0,8:10]))