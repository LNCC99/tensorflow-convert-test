import time
import os
#import tensorrt as trt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Conv2D, BatchNormalization, Conv2DTranspose, Concatenate, LayerNormalization, PReLU, ZeroPadding2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

from random import seed
import numpy as np

from modules import DprnnBlock
from utils import reshape, transpose, ParallelModelCheckpoints
from tensorflow.python.framework.ops import disable_eager_execution

seed(42)

class DPCRN_model():
    '''
    Class to create and train the DPCRN model
    '''
    
    def __init__(self, batch_size = 1,
                       length_in_s = 5,
                       fs = 16000,
                       norm = 'iLN',
                       numUnits = 128,
                       numDP = 2,
                       block_len = 400,
                       block_shift = 200,
                       max_epochs = 200,
                       lr = 1e-3):

        # defining default cost function
        self.cost_function = self.snr_cost
        self.model = None
        # defining default parameters
        self.fs = fs
        self.length_in_s = length_in_s
        self.batch_size = batch_size
        # number of the hidden layer size in the LSTM
        self.numUnits = numUnits
        # number of the DPRNN modules
        self.numDP = numDP
        # frame length and hop length in STFT
        self.block_len = block_len
        self.block_shift = block_shift
        self.lr = lr
        self.max_epochs = max_epochs
        # window for STFT: sine win
        win = np.sin(np.arange(.5,self.block_len-.5+1)/self.block_len*np.pi)
        self.win = tf.constant(win,dtype = 'float32')

        self.L = (16000*length_in_s-self.block_len)//self.block_shift + 1
        
        self.multi_gpu = False
        # iLN for instant Layer norm and BN for Batch norm
        self.input_norm = norm
        
    @staticmethod
    def snr_cost(s_estimate, s_true):
        '''
        Static Method defining the cost function. 
        The negative signal to noise ratio is calculated here. The loss is 
        always calculated over the last dimension. 
        '''
        # calculating the SNR
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
            (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True) + 1e-8)
        # using some more lines, because TF has no log10
        num = tf.math.log(snr + 1e-8) 
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10*(num / (denom))

        return loss
    
    @staticmethod
    def sisnr_cost(s_hat, s):
        '''
        Static Method defining the cost function. 
        The negative signal to noise ratio is calculated here. The loss is 
        always calculated over the last dimension. 
        '''
        def norm(x):
            return tf.reduce_sum(x**2, axis=-1, keepdims=True)

        s_target = tf.reduce_sum(
            s_hat * s, axis=-1, keepdims=True) * s / norm(s)
        upp = norm(s_target)
        low = norm(s_hat - s_target)
  
        return -10 * tf.log(upp /low) / tf.log(10.0)  
    
    def spectrum_loss(self,y_true):
        '''
        spectrum MSE loss 
        '''
        enh_real = self.enh_real
        enh_imag = self.enh_imag
        enh_mag = tf.math.sqrt(enh_real**2 + enh_imag**2 + 1e-8)
        
        true_real,true_imag = self.stftLayer(y_true, mode='real_imag')
        true_mag = tf.math.sqrt(true_real**2 + true_imag**2 + 1e-8)
        
        loss_real = tf.reduce_mean((enh_real - true_real)**2,)
        loss_imag = tf.reduce_mean((enh_imag - true_imag)**2,)
        loss_mag = tf.reduce_mean((enh_mag - true_mag)**2,) 
        
        return loss_real + loss_imag + loss_mag
    
    def spectrum_loss_phasen(self, s_hat,s,gamma = 0.3):
        
        true_real,true_imag = self.stftLayer(s, mode='real_imag')
        hat_real,hat_imag = self.stftLayer(s_hat, mode='real_imag')

        true_mag = tf.math.sqrt(true_real**2 + true_imag**2+1e-9)
        hat_mag = tf.math.sqrt(hat_real**2 + hat_imag**2+1e-9)

        true_real_cprs = (true_real / true_mag )*true_mag**gamma
        true_imag_cprs = (true_imag / true_mag )*true_mag**gamma
        hat_real_cprs = (hat_real / hat_mag )* hat_mag**gamma
        hat_imag_cprs = (hat_imag / hat_mag )* hat_mag**gamma

        loss_mag = tf.reduce_mean((hat_mag**gamma - true_mag**gamma)**2,)         
        loss_real = tf.reduce_mean((hat_real_cprs - true_real_cprs)**2,)
        loss_imag = tf.reduce_mean((hat_imag_cprs - true_imag_cprs)**2,)

        return 0.7 * loss_mag + 0.3 * ( loss_imag + loss_real ) 
    
    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''
        def lossFunction(y_true,y_pred):
            # calculating loss and squeezing single dimensions away
            loss = tf.squeeze(self.cost_function(y_pred,y_true)) 
            mag_loss = tf.math.log(self.spectrum_loss(y_true) + 1e-8)
            # calculate mean over batches
            loss = tf.reduce_mean(loss)
            return loss + mag_loss 
        
        return lossFunction
    
    '''
    In the following some helper layers are defined.
    '''  
    def seg2frame(self, x):
        '''
        split signal x to frames
        '''
        frames = tf.signal.frame(x, self.block_len, self.block_shift)
        if self.win is not None:
            frames = self.win*frames
        return frames
    
    def stftLayer(self, x, mode ='mag_pha'):
        '''
        Method for an STFT helper layer used with a Lambda layer
        mode: 'mag_pha'   return magnitude and phase spectrogram
              'real_imag' return real and imaginary parts
        '''
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.block_len, self.block_shift)
        if self.win is not None:
            frames = self.win*frames
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        output_list = []
        if mode == 'mag_pha':
            mag = tf.math.abs(stft_dat)
            phase = tf.math.angle(stft_dat)
            output_list = [mag, phase]
        elif mode == 'real_imag':
            real = tf.math.real(stft_dat)
            imag = tf.math.imag(stft_dat)
            output_list = [real, imag]            
        # returning magnitude and phase as list
        return output_list
    
    def fftLayer(self, x):
        '''
        Method for an fft helper layer used with a Lambda layer.
        The layer calculates the rFFT on the last dimension and returns
        the magnitude and phase of the STFT.
        '''
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(x)
        # calculating magnitude and phase from the complex signal
        mag = tf.math.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]


    def ifftLayer(self, x,mode = 'mag_pha'):
        '''
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information. 
        As input x a list with [mag,phase] is required.
        '''
        if mode == 'mag_pha':
        # calculating the complex representation
            s1_stft = (tf.cast(x[0], tf.complex64) * 
                        tf.exp( (1j * tf.cast(x[1], tf.complex64))))
        elif mode == 'real_imag':
            s1_stft = tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64)
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)  
    
    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        '''
        # calculating and returning the reconstructed waveform
        '''
        if self.move_dc:
            x = x - tf.expand_dims(tf.reduce_mean(x,axis = -1),2)
        '''
        return tf.signal.overlap_and_add(x, self.block_shift)              
     
    def mk_mask(self,x):
        '''
        Method for complex ratio mask and add helper layer used with a Lambda layer.
        '''
        [noisy_real,noisy_imag,mask] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        mask_real = mask[:,:,:,0]
        mask_imag = mask[:,:,:,1]
        
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real
        
        return [enh_real,enh_imag]
        
    def build_DPCRN_model(self, name = 'model0'):

        # input layer for time signal
        #time_dat = Input(batch_shape=(self.batch_size, None))
        time_dat = Input(batch_shape=(self.batch_size, 172600))
        # calculate STFT
        real,imag = Lambda(self.stftLayer,arguments = {'mode':'real_imag'})(time_dat)
        # normalizing log magnitude stfts to get more robust against level variations
        real = Lambda(reshape,arguments={'axis':[self.batch_size,-1,self.block_len // 2 + 1,1]})(real) #Lambda 将任意表达包装成layer对象
        imag = Lambda(reshape,arguments={'axis':[self.batch_size,-1,self.block_len // 2 + 1,1]})(imag)
        input_complex_spec = Concatenate(axis = -1)([real,imag])
        '''encoder'''

        if self.input_norm == 'iLN':    
            input_complex_spec = LayerNormalization(axis = [-1,-2], name = 'input_norm')(input_complex_spec)
        elif self.input_norm == 'BN':    
            input_complex_spec =BatchNormalization(name = 'input_norm')(input_complex_spec)
        
        # causal padding [1,0],[0,2]
        padding = ZeroPadding2D(padding=((1,0),(0,2)))(input_complex_spec)
        conv_1 = Conv2D(32, (2,5),(1,2),name = name+'_conv_1',padding='valid')(padding)
        #conv_1 = Conv2D(32, (2,5),(1,2),name = name+'_conv_1',padding = [[0,0],[1,0],[0,2],[0,0]])(input_complex_spec)
        bn_1 = BatchNormalization(name = name+'_bn_1')(conv_1)
        out_1 = PReLU(shared_axes=[1,2])(bn_1)
        # causal padding [1,0],[0,1]
        padding = ZeroPadding2D(padding=((1,0),(0,1)))(out_1)
        conv_2 = Conv2D(32, (2,3),(1,2),name = name+'_conv_2',padding='valid')(padding)
        #conv_2 = Conv2D(32, (2,3),(1,2),name = name+'_conv_2',padding = [[0,0],[1,0],[0,1],[0,0]])(out_1)
        bn_2 = BatchNormalization(name = name+'_bn_2')(conv_2)
        out_2 = PReLU(shared_axes=[1,2])(bn_2)
        # causal padding [1,0],[1,1]
        padding = ZeroPadding2D(padding=((1,0),(1,1)))(out_2)
        conv_3 = Conv2D(32, (2,3),(1,1),name = name+'_conv_3',padding ='valid')(padding)
        #conv_3 = Conv2D(32, (2,3),(1,1),name = name+'_conv_3',padding = [[0,0],[1,0],[1,1],[0,0]])(out_2)
        bn_3 = BatchNormalization(name = name+'_bn_3')(conv_3)
        out_3 = PReLU(shared_axes=[1,2])(bn_3)
        # causal padding [1,0],[1,1]
        padding = ZeroPadding2D(padding=((1,0),(1,1)))(out_3)
        conv_4 = Conv2D(64, (2,3),(1,1),name = name+'_conv_4',padding='valid')(padding)
        #conv_4 = Conv2D(64, (2,3),(1,1),name = name+'_conv_4',padding = [[0,0],[1,0],[1,1],[0,0]])(out_3)
        bn_4 = BatchNormalization(name = name+'_bn_4')(conv_4)
        out_4 = PReLU(shared_axes=[1,2])(bn_4)
        # causal padding [1,0],[1,1]
        padding = ZeroPadding2D(padding=((1,0),(1,1)))(out_4)
        conv_5 = Conv2D(128, (2,3),(1,1),name = name+'_conv_5',padding = 'valid')(padding)
        #conv_5 = Conv2D(128, (2,3),(1,1),name = name+'_conv_5',padding = [[0,0],[1,0],[1,1],[0,0]])(out_4)
        bn_5 = BatchNormalization(name = name+'_bn_5')(conv_5)
        out_5 = PReLU(shared_axes=[1,2])(bn_5)
        
        dp_in = out_5
        
        for i in range(self.numDP):
            
            dp_in = DprnnBlock(numUnits = self.numUnits, batch_size = self.batch_size, L = -1,width = 50,channel = 128, causal=True)(dp_in)#self.DPRNN_kernal(dp_in,str(i),last_dp = 0)
       
        dp_out = dp_in
        
        '''decoder'''
        skipcon_1 = Concatenate(axis = -1)([out_5,dp_out])

        deconv_1 = Conv2DTranspose(64,(2,3),(1,1),name = name+'_dconv_1',padding = 'same')(skipcon_1)
        dbn_1 = BatchNormalization(name = name+'_dbn_1')(deconv_1)
        dout_1 = PReLU(shared_axes=[1,2])(dbn_1)

        skipcon_2 = Concatenate(axis = -1)([out_4,dout_1])
        
        deconv_2 = Conv2DTranspose(32,(2,3),(1,1),name = name+'_dconv_2',padding = 'same')(skipcon_2)
        dbn_2 = BatchNormalization(name = name+'_dbn_2')(deconv_2)
        dout_2 = PReLU(shared_axes=[1,2])(dbn_2)
        
        skipcon_3 = Concatenate(axis = -1)([out_3,dout_2])
        
        deconv_3 = Conv2DTranspose(32,(2,3),(1,1),name = name+'_dconv_3',padding = 'same')(skipcon_3)
        dbn_3 = BatchNormalization(name = name+'_dbn_3')(deconv_3)
        dout_3 = PReLU(shared_axes=[1,2])(dbn_3)
        
        skipcon_4 = Concatenate(axis = -1)([out_2,dout_3])

        deconv_4 = Conv2DTranspose(32,(2,3),(1,2),name = name+'_dconv_4',padding = 'same')(skipcon_4)
        dbn_4 = BatchNormalization(name = name+'_dbn_4')(deconv_4)
        dout_4 = PReLU(shared_axes=[1,2])(dbn_4)
        
        skipcon_5 = Concatenate(axis = -1)([out_1,dout_4])
        
        deconv_5 = Conv2DTranspose(2,(2,5),(1,2),name = name+'_dconv_5',padding = 'valid')(skipcon_5)
        
        '''no activation'''        
        deconv_5 = deconv_5[:,:-1,:-2]

        #output_mask = Activation('tanh')(dbn_5)
        output_mask = deconv_5

        enh_spec = Lambda(self.mk_mask)([real,imag,output_mask])
        
        self.enh_real, self.enh_imag = enh_spec[0],enh_spec[1]
        
        enh_frame = Lambda(self.ifftLayer,arguments = {'mode':'real_imag'})(enh_spec)
        enh_frame = enh_frame * self.win
        enh_time = Lambda(self.overlapAddLayer, name = 'enhanced_time')(enh_frame)
        
        self.model = Model(time_dat,enh_time)
        self.model.summary()

        return self.model

    def compile_model(self):
        '''
        Method to compile the model for training
        '''
        # use the Adam optimizer with a clipnorm of 3
        optimizerAdam = keras.optimizers.Adam(lr=self.lr, clipnorm=3.0)
        # compile model with loss function
        self.model.compile(loss=self.lossWrapper(),optimizer=optimizerAdam)

    def train_model(self, runName, data_generator):
        '''
        Method to train the model. 
        '''
        self.compile_model()
        
        # create save path if not existent
        savePath = './models_'+ runName+'/' 
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # create log file writer
        csv_logger = CSVLogger(savePath+ 'training_' +runName+ '.log')
        # create callback for the adaptive learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=10**(-10), cooldown=1)
        # create callback for early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, 
            patience=10,  mode='auto', baseline=None)
        # create model check pointer to save the best model

        checkpointer = ModelCheckpoint(savePath+runName+'model_{epoch:02d}_{val_loss:02f}.h5',
                                       monitor='val_loss',
                                       save_best_only=False,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq='epoch'
                                       )

        # create data generator for training data
        start_time = time.time()
        
        self.model.fit_generator(data_generator.generator(batch_size = self.batch_size,validation = False), 
                                 validation_data = data_generator.generator(batch_size =self.batch_size,validation = True),
                                 epochs = self.max_epochs, 
                                 steps_per_epoch = data_generator.train_length//self.batch_size,
                                 validation_steps = self.batch_size,
                                 #use_multiprocessing=True,
                                 callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping])
        '''
        self.model.fit_generator(data_generator.generator(batch_size = self.batch_size,validation = False), 
                                 #validation_data = data_generator.generator(batch_size =self.batch_size,validation = True),
                                 epochs = self.max_epochs, 
                                 steps_per_epoch = data_generator.train_length//self.batch_size,
                                 #validation_steps = self.batch_size,
                                 #use_multiprocessing=True,
                                 #callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping]
                                 )
       '''
        end_time = time.time()
        print(f"train time:{end_time - start_time} s")
        # clear out garbage
        tf.keras.backend.clear_session()  
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fs' : self.fs,
            'length_in_s' : self.length_in_s,
            'batch_size' : self.batch_size,
            'numUnits' : self.numUnits,
            'numDP' : self.numDP,
            'block_len' : self.block_len,
            'block_shift' : self.block_shift,
            'lr' : self.lr,
            'max_epochs' : self.max_epochs,
            'norm' : self.input_norm,
        }) 
        return config
    


def representative_data_gen():
    for _ in range(100):
        yield [np.random.rand(1,172600).astype(np.float32)]    
if __name__ == '__main__':
                
    dpcrn = DPCRN_model(batch_size = 1, 
                        length_in_s = 5, 
                        lr = 1e-3,
                        block_len = 400,
                        block_shift = 200)
    my_model = dpcrn.build_DPCRN_model()
    converter = tf.lite.TFLiteConverter.from_keras_model(my_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS,tf.lite.OpsSet.TFLITE_BUILTINS_INT8] #
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()

    quant_model_path = './model_tflite.tflite'
    with open(quant_model_path, 'wb') as f:
        f.write(tflite_model)
