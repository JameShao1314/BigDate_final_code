
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io
import cPickle
import configuration
import caption_generator_rawlstm

class Evaluate(object):

  def __init__(self):

        x = cPickle.load(open("./data/mscoco/data.p","rb"))
	train, val, test = x[0], x[1], x[2]
	wordtoix, ixtoword = x[3], x[4]
	del x
	n_words = len(ixtoword)
	    
	x = cPickle.load(open("./data/mscoco/word2vec.p","rb"))
	W = x[0]
	del x
	data = scipy.io.loadmat('./data/mscoco/resnet_feats.mat')
	img_feats = data['feats'].astype(float)
        print("finish loading data")
        self.val = test #test the test images
        self.img_feats = img_feats
	self.ixtoword = ixtoword
      
  def evaluate(self):
        g = tf.Graph()
	with g.as_default():
            model_config = configuration.ModelConfig()
	    training_config = configuration.TrainingConfig()

	    #initializer method
	    initializer = tf.random_uniform_initializer(
		minval=-model_config.initializer_scale,
		maxval=model_config.initializer_scale)

	    seq_embeddings = None
	    image_feed = tf.placeholder(dtype=tf.float32, shape=[2048], name="image_feed")
	    input_feed = tf.placeholder(dtype=tf.int32,
			                  shape=[None],  # batch_size
			                  name="input_feed")
	    

	    # Process image and insert batch dimensions.
	    image_fea = tf.expand_dims(image_feed, 0)
	    #input_seqs = tf.expand_dims(input_feed, 1)
	    input_seqs = input_feed
	    #image_fea = image_feed

	    with tf.variable_scope("seq_embedding"), tf.device("/gpu:0"):
	      embedding_map = tf.get_variable(
		  name="map",
		  shape=[model_config.vocab_size, model_config.embedding_size],
		  initializer=initializer)
	    seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

	    with tf.variable_scope("image_embedding") as scope:
	      image_embeddings = tf.contrib.layers.fully_connected(
		  inputs=image_fea,
		  num_outputs=model_config.embedding_size,
		  activation_fn=None,
		  weights_initializer=initializer,
		  biases_initializer=None,
		  scope=scope)

	    W = tf.get_variable('W', shape=[4, model_config.num_lstm_units, model_config.num_lstm_units], initializer=initializer)
            U = tf.get_variable('U', shape=[4, model_config.num_lstm_units, model_config.num_lstm_units], initializer=initializer)
	    def step(prev, x):
                # gather previous internal state and output state
                st_1, ct_1 = tf.unstack(prev)
                ####
                # GATES
                #
                #  input gate
                i = tf.sigmoid(tf.matmul(x,U[0]) + tf.matmul(st_1,W[0]))
                #  forget gate
                f = tf.sigmoid(tf.matmul(x,U[1]) + tf.matmul(st_1,W[1]))
                #  output gate
                o = tf.sigmoid(tf.matmul(x,U[2]) + tf.matmul(st_1,W[2]))
                #  gate weights
                g = tf.tanh(tf.matmul(x,U[3]) + tf.matmul(st_1,W[3]))
                ###
                # new internal cell state
                ct = ct_1*f + g*i
                # output state
                st = tf.tanh(ct)*o
                return tf.stack([st, ct])
	    image_embeddings = tf.stack([image_embeddings,image_embeddings])

	    #image_embeddings = tf.expand_dims(image_embeddings, 0)
	    image_embeddings = tf.transpose(image_embeddings,[1,0,2],name='initial_state')
	    state_feed = tf.placeholder(dtype=tf.float32,
		                            shape=[None,2, model_config.num_lstm_units],
		                            name="state_feed")
	    #state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
	    state_feed = tf.transpose(state_feed,[1,0,2])
	    seq_embeddings = tf.reshape(seq_embeddings,[-1,model_config.num_lstm_units])
	    states = step(state_feed,seq_embeddings)
	    
	    #states = tf.scan(step, 
                    #tf.transpose(seq_embeddings, [1,0,2]),
		    #initializer=state_feed)
	    tf.transpose(states, [1,0,2],name='state')
	    #states = tf.Print(states, ["lstm states shape:",tf.shape(states)])
	    states = states[0]
	    #states = tf.Print(states, ["lstm states REshape:",tf.shape(states)])
	    lstm_outputs = tf.reshape(states, [-1, model_config.num_lstm_units])
	    #lstm_outputs = tf.Print(lstm_outputs, [tf.shape(lstm_outputs), "lstm_outputs"])

	    with tf.variable_scope("logits") as logits_scope:
	      logits = tf.contrib.layers.fully_connected(
		  inputs=lstm_outputs,
		  num_outputs=model_config.vocab_size,
		  activation_fn=None,
		  weights_initializer=initializer,
		  scope=logits_scope)

	   
	    tf.nn.softmax(logits, name="softmax")

	    global_step = tf.Variable(
		initial_value=0,
		name="global_step",
		trainable=False,
		collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
            # Set up the Saver for saving and restoring model checkpoints.
            saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
            g.as_default()
            sess = tf.InteractiveSession(graph=g)
            #load the trained model 
            with sess.as_default():
                saver.restore(sess, "log/model.ckpt-19") 

        print("finish initialization")

        x= self.val[0]
        lengths = [len(s) for s in x]
	n_samples = len(x)
	maxlen = np.max(lengths)
        #remove duplicate. Because one image has many captions.
        val_re = []
        for i in range(n_samples):
        	if self.val[1][i] not in val_re:
   			val_re.append(self.val[1][i])
        n_samples = len(val_re)
        print("n_samples:"+str(n_samples)+"maxlen:"+str(maxlen))
	z = np.array([self.img_feats[:,val_re[t]]for t in range(n_samples)])
        cap = np.zeros(( n_samples,maxlen))
        generator = caption_generator_rawlstm.CaptionGenerator()
        #generate captions.feed word one by one to the model.Start with 6800('#').Stop when get 0('.')
        for num in range(n_samples):
                if num%100==0:
			print(num)
		if 1 :
		        captions = generator.beam_search(sess, z[num])
		        for s in range(len(captions[0].sentence)-1):
				cap[num][s]= captions[0].sentence[s+1]
                else:
			initial_state = sess.run(fetches="initial_state:0",
		                     feed_dict={"image_feed:0": z[num]})
		        input_feed = np.array([6800])
		        state_feed = initial_state
		      
			for s in range(maxlen):		
				softmax_output, state_output = sess.run(
					fetches=["softmax:0", "state:0"],
					feed_dict={
					    "input_feed:0": input_feed,
					    "state_feed:0": state_feed,
					})
				#print(softmax_output.shape)
		                softmax_output = softmax_output.reshape(softmax_output.shape[1])
				input_feed = [np.argsort(-softmax_output)[0]]
				#print(softmax_output.shape)
				#print(input_feed)
		                state_feed  = state_output
		                cap[num][s] = input_feed[0]
		                if input_feed[0]==0:
		                        #print(cap[num])
		                        break
		 	
        #get the real word by index
	precaptext=[]
	for i in range(n_samples):
		temcap=[]
		for j in range(maxlen):
			if cap[i][j]!=0:
				temcap.append(self.ixtoword[cap[i][j]])
			else:
				break
		precaptext.append(" ".join(temcap))
        #save the results to 'coco_5k_test.txt'
	print('write generated captions into a text file...')
	open('./coco_5k_test.txt', 'w').write('\n'.join(precaptext))
	
				



