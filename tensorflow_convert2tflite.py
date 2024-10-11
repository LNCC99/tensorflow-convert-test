import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tflite_runtime.interpreter as tflite
from dpcrn_model import DPCRN_model
import os
import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
def representative_data_gen():
   for _ in range(2):
      yield [np.random.rand(1,1,201,2).astype(np.float32)]
      

dpcrn = DPCRN_model(batch_size = 1, 
                    length_in_s = 5, 
                    lr = 1e-3,
                    block_len = 400,
                    block_shift = 200)
my_model = dpcrn.build_DPCRN_model()
converter = tf.lite.TFLiteConverter.from_keras_model(my_model)
      
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
#converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8] #
#converter.inference_input_type = tf.int16
#converter.inference_output_type = tf.int16
tflite_model = converter.convert()

quant_model_path = './model_tflite_CTIVATIONS_INT16_WEIGHTS_INT8_InOutInt16.tflite'
with open(quant_model_path, 'wb') as f:
    f.write(tflite_model)
#interpreter = tf.lite.Interpreter(model_path = tflite_model)
interpreter = tflite.Interpreter(model_path=quant_model_path)
interpreter.allocate_tensors() #加载所有的tensor

input_type = interpreter.get_input_details()[0]['dtype']
print(interpreter.get_input_details())
print('input type:',input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print(interpreter.get_output_details())
print('output type:',output_type)
print('/n')
tensor_details = interpreter.get_tensor_details()
for tensor in tensor_details:
    #print(tensor)
    print(f"Tensor Name:{tensor['name']}")
    print(f"Tensor Shape:{tensor['shape']}")
    print(f"Tensor Type：{tensor['dtype']}")
