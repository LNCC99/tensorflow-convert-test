# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:53:48 2021

@author: xiaohuaile
"""

import tensorflow.keras as keras
import tensorflow as tf

layernorm_enable = False
'''
dual path rnn block
'''   
class DprnnBlock(keras.layers.Layer):
    
    def __init__(self, numUnits, batch_size, L, width, channel, causal = True, **kwargs):
        super(DprnnBlock, self).__init__(**kwargs)
        '''
        numUnits  hidden layer size in the LSTM
        batch_size 
        L         number of frames, -1 for undefined length
        width     width size output from encoder
        channel   channel size output from encoder
        causal    instant Layer Norm or global Layer Norm
        '''
        self.numUnits = numUnits
        self.batch_size = batch_size
        self.causal = causal
        #self.intra_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.numUnits//2, return_sequences=True,implementation = 2,recurrent_activation = 'hard_sigmoid'))
        self.intra_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.numUnits//2, return_sequences=True,implementation = 2,recurrent_activation = 'sigmoid'))
       
        self.intra_fc = keras.layers.Dense(units = self.numUnits,)
        if layernorm_enable:
            if self.causal:
                self.intra_ln = keras.layers.LayerNormalization(center=True, scale=True, axis = [-1,-2])
            else:
                self.intra_ln = keras.layers.LayerNormalization(center=False, scale=False)

        #self.inter_rnn = keras.layers.LSTM(units=self.numUnits, return_sequences=True,implementation = 2,recurrent_activation = 'hard_sigmoid')
        self.inter_rnn = keras.layers.LSTM(units=self.numUnits, return_sequences=True,implementation = 2,recurrent_activation = 'sigmoid')
        self.inter_fc = keras.layers.Dense(units = self.numUnits,) 
        if layernorm_enable:
            if self.causal:
                self.inter_ln = keras.layers.LayerNormalization(center=True, scale=True, axis = [-1,-2])
            else:
                self.inter_ln = keras.layers.LayerNormalization(center=False, scale=False)

        self.L = L
        self.width = width
        self.channel = channel
        
        
    def call(self, x):

        batch_size = self.batch_size
        L = self.L
        width = self.width
        
        intra_rnn = self.intra_rnn
        intra_fc = self.intra_fc
        if layernorm_enable:
            intra_ln = self.intra_ln
        inter_rnn = self.inter_rnn
        inter_fc = self.inter_fc
        if layernorm_enable:
            inter_ln = self.inter_ln
        channel = self.channel
        causal = self.causal
        
        # Intra-Chunk Processing
        # input shape (bs,T,F,C) --> (bs*T,F,C)
        intra_LSTM_input = tf.reshape(x,[-1,width,channel])
        # (bs*T,F,C)
        intra_LSTM_out = intra_rnn(intra_LSTM_input)
        
        # (bs*T,F,C) channel axis dense
        intra_dense_out = intra_fc(intra_LSTM_out)
        
        if layernorm_enable:   
            if causal:
                # (bs*T,F,C) --> (bs,T,F,C) Freq and channel norm
                intra_ln_input = tf.reshape(intra_dense_out,[batch_size,-1,width,channel])
                intra_out = intra_ln(intra_ln_input)
                
            else:       
                # (bs*T,F,C) --> (bs,T*F*C) global norm
                intra_ln_input = tf.reshape(intra_dense_out,[batch_size,-1])
                intra_ln_out = intra_ln(intra_ln_input)
                intra_out = tf.reshape(intra_ln_out,[batch_size,L,width,channel])
        else:
            intra_ln_input = tf.reshape(intra_dense_out,[batch_size,-1,width,channel])
            intra_out = intra_ln_input
        # (bs,T,F,C)
        intra_out = keras.layers.Add()([x,intra_out])
        #%% Inter-Chunk Processing
        # (bs,T,F,C) --> (bs,F,T,C)
        inter_LSTM_input = tf.transpose(intra_out,[0,2,1,3])
        # (bs,F,T,C) --> (bs*F,T,C)
        inter_LSTM_input = tf.reshape(inter_LSTM_input,[batch_size*width,L,channel])
        
        inter_LSTM_out = inter_rnn(inter_LSTM_input)
        
        # (bs,F,T,C) 
        inter_dense_out = inter_fc(inter_LSTM_out)
        
        inter_dense_out = tf.reshape(inter_dense_out,[batch_size,width,L,channel])
        if layernorm_enable:
            if causal:
                # (bs,F,T,C) --> (bs,T,F,C)
                inter_ln_input = tf.transpose(inter_dense_out,[0,2,1,3])
                inter_out = inter_ln(inter_ln_input)
                
            else:
                # (bs,F,T,C) --> (bs,F*T*C)
                inter_ln_input = tf.reshape(inter_dense_out,[batch_size,-1])
                inter_ln_out = inter_ln(inter_ln_input)
                inter_out = tf.reshape(inter_ln_out,[batch_size,width,L,channel])
                # (bs,F,T,C) --> (bs,T,F,C)
                inter_out = tf.transpose(inter_out,[0,2,1,3])
        else:
            inter_ln_input = tf.transpose(inter_dense_out,[0,2,1,3])
            inter_out = inter_ln_input
        # (bs,T,F,C)
        inter_out = keras.layers.Add()([intra_out,inter_out])
    
        return inter_out           
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numUnits' : self.numUnits,
            'batch_size' : self.batch_size,
            'causal' : self.causal,
            'L' : self.L,
            'width' : self.width,
            'channel' : self.channel,
        })   
        return config  