import numpy as np
from tensorflow.keras import layers
from keras.layers import Dropout, BatchNormalization, Concatenate, RepeatVector, Dense, LSTM, Input, LeakyReLU, Lambda, GRU, SimpleRNN, Flatten, Conv2DTranspose, ConvLSTM2D, MaxPooling3D, TimeDistributed, Reshape, Subtract, Add, Multiply, MultiHeadAttention, LayerNormalization, Conv2D, MaxPooling2D, Convolution2D
from keras.models import Model
from keras import backend as K
from keras import regularizers, activations
from keras.utils import plot_model
from keras.losses import SquaredHinge
from tensorflow.keras import optimizers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from keras import Sequential

import tensorflow as tf

from utilsforminds.containers import merge_dictionaries
from utilsforminds.numpy_array import mask_prob
import utilsforminds.tensors as tensors
import utils

from tqdm import tqdm
from copy import deepcopy
from random import random, sample
import cv2

keras_functions_dict = {"Dense": Dense, "Dropout": Dropout, "BatchNormalization": BatchNormalization, "LSTM": LSTM, "GRU": GRU, "Flatten": Flatten, "Conv2DTranspose": Conv2DTranspose, "ConvLSTM2D": ConvLSTM2D, "MaxPooling3D": MaxPooling3D, "TimeDistributed": TimeDistributed, "Reshape": Reshape, "RandomFourierFeatures": RandomFourierFeatures, "MultiHeadAttention": MultiHeadAttention, "LayerNormalization": LayerNormalization, "Conv2D": Conv2D, "MaxPooling2D": MaxPooling2D}
# keras_optimizers_dict = {"Adam": optimizers.Adam} # (learning_rate= 0.001, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-7)
basic_regularizer = None

class TransformerTemporal(layers.Layer):
    def __init__(self, input_dim, model_kwargs_attention = None, model_kwargs_prior_residual = None, model_kwargs_temporal = None, model_kwargs_post_residual = None, key_model_kwargs = None, query_model_kwargs = None, value_model_kwargs = None, prior_post_layer_norm_kwargs = None, droprate_before_residual= None):
        super(TransformerTemporal, self).__init__()

        ## Set default models.
        # assert(model_kwargs_attention[0] == "MultiHeadAttention")
        if model_kwargs_attention is None: self.model_kwargs_attention = ["MultiHeadAttention", dict(num_heads= 2, key_dim= 32)]
        else: self.model_kwargs_attention = deepcopy(model_kwargs_attention)

        if key_model_kwargs: self.key_model_kwargs = key_model_kwargs
        else: self.key_model_kwargs = dict(model= "LSTM", kwargs= dict(units= 30))

        if query_model_kwargs: self.query_model_kwargs = query_model_kwargs
        else: self.query_model_kwargs = dict(model= "LSTM", kwargs= dict(units= 30))

        if value_model_kwargs: self.value_model_kwargs = value_model_kwargs
        else: self.value_model_kwargs = dict(model= "LSTM", kwargs= dict(units= 30))

        for model_kwargs in [self.key_model_kwargs, self.query_model_kwargs, self.value_model_kwargs]:
            if model_kwargs["model"] in ["LSTM", "GRU", "SimpleRNN"]:
                model_kwargs["kwargs"]["return_sequences"] = True
                model_kwargs["kwargs"]["return_state"] = True

        if model_kwargs_prior_residual is None: self.model_kwargs_prior_residual = []
        else: self.model_kwargs_prior_residual = deepcopy(model_kwargs_prior_residual)

        self.model_kwargs_temporal = deepcopy(model_kwargs_temporal)

        if model_kwargs_post_residual is None: self.model_kwargs_post_residual = []
        else: self.model_kwargs_post_residual = deepcopy(model_kwargs_post_residual)

        if prior_post_layer_norm_kwargs is not None: self.prior_post_layer_norm_kwargs = deepcopy(prior_post_layer_norm_kwargs)
        else: self.prior_post_layer_norm_kwargs = dict(prior= None, post= None)

        ## Build model
        self.key_model = keras_functions_dict[self.key_model_kwargs["model"]](**self.key_model_kwargs["kwargs"])
        self.query_model = keras_functions_dict[self.query_model_kwargs["model"]](**self.query_model_kwargs["kwargs"])
        self.value_model = keras_functions_dict[self.value_model_kwargs["model"]](**self.value_model_kwargs["kwargs"])
        self.att = keras_functions_dict[self.model_kwargs_attention[0]](**self.model_kwargs_attention[1])
        sequential = [keras_functions_dict[self.model_kwargs_prior_residual[i][0]](**self.model_kwargs_prior_residual[i][1]) for i in range(len(self.model_kwargs_prior_residual))]
        if self.prior_post_layer_norm_kwargs["prior"]: sequential.append(Dense(units = input_dim)) ## To match the dimension of input
        if droprate_before_residual is not None: sequential.append(Dropout(rate= droprate_before_residual))
        if len(sequential) > 0: self.prior_residual = Sequential(
            sequential
            )
        if self.model_kwargs_temporal: 
            self.temporal = keras_functions_dict[self.model_kwargs_temporal[0]](return_sequences= True, return_state= True, **self.model_kwargs_temporal[1])
        
        sequential = [keras_functions_dict[self.model_kwargs_post_residual[i][0]](**self.model_kwargs_post_residual[i][1]) for i in range(len(self.model_kwargs_post_residual))]
        if self.prior_post_layer_norm_kwargs["post"]:
            sequential.append(Dense(units = input_dim)) ## To match the dimension
        if droprate_before_residual is not None: sequential.append(Dropout(rate= droprate_before_residual))
        if len(sequential) > 0: self.post_residual = Sequential(sequential)

        if self.prior_post_layer_norm_kwargs["prior"] is not None: self.layer_norm_prior_residual = LayerNormalization(**self.prior_post_layer_norm_kwargs["prior"])
        if self.prior_post_layer_norm_kwargs["post"] is not None:self.layer_norm_post_residual = LayerNormalization(**self.prior_post_layer_norm_kwargs["post"])
    
    def call(self, inputs):
        """Sequence to sequence.
        
        Parameters
        ----------
        inputs: 
            shape is [batch, time step, num features].

        Return
        ------
        shape is [batch, time step, num features].
        """

        outputs = inputs
        if self.key_model_kwargs["model"] in ["LSTM", "GRU", "SimpleRNN"]:
            key, _, _ = self.key_model(inputs)
        else:
            key = self.key_model(inputs)
        if self.query_model_kwargs["model"] in ["LSTM", "GRU", "SimpleRNN"]:
            query, _, _ = self.query_model(inputs)
        else:
            query = self.query_model(inputs)
        if self.value_model_kwargs["model"] in ["LSTM", "GRU", "SimpleRNN"]:
            value, _, _ = self.value_model(inputs)
        else:
            value = self.value_model(inputs)
        if self.model_kwargs_temporal: 
            outputs, vectors_last_hidden_state, vectors_last_cell_state  = self.temporal(outputs)
        # positional_encoding = 
        outputs = self.att(query = query, value = value, key = key)
        if len(self.model_kwargs_prior_residual) > 0 or self.prior_post_layer_norm_kwargs["prior"] is not None: 
            outputs = self.prior_residual(outputs)
        if self.prior_post_layer_norm_kwargs["prior"] is not None: 
            outputs = self.layer_norm_prior_residual(outputs + inputs)
        outputs_residual = outputs
        if len(self.model_kwargs_post_residual) > 0 or self.prior_post_layer_norm_kwargs["post"] is not None:
            outputs = self.post_residual(outputs)
            outputs = self.layer_norm_post_residual(outputs + outputs_residual)
        return outputs

class HighOrderMixture(layers.Layer):
    def __init__(self, num_features, order= 3, init_missing_order = 0.0):
        super(HighOrderMixture, self).__init__()
        self.order = order
        self.init_missing_order = init_missing_order
        self.num_features = num_features
    
    # @tf.function
    def call(self, inputs, timesteps):
        """Sequence to sequence.
        
        Parameters
        ----------
        inputs: 
            shape is [batch= 1, time steps, num features].dtype= tf.float32.

        Return
        ------
        shape is [batch= 1, num features].
        """

        ## Collect items satisfying items, tf.where.

        batch_size = K.shape(inputs)[0] ## = 1

        inputs_expanded = inputs[0] ## Remove dummy batch dim, (timesteps, features)
        timesteps = timesteps[0] ## Remove dummy batch dim,
        # timesteps = K.shape(inputs)[1]

        def get_bools_not_in_previous_timesteps_mixed(previous_timesteps):
            """
            previous_timesteps: [combs, order]
            
            """

            return tf.map_fn(fn= lambda t: not tensors.element_is_in_tensor(previous_timesteps, t), elems= tf.range(timesteps), dtype= tf.bool)

        def true_body(previous_timesteps_mixed, order):
            """

            previous_timesteps_mixed: [combs, order]
            """
            previous_order = order - 1

            # previous_timesteps_mixed = tf.expand_dims(previous_timesteps_mixed, axis= -1) ## [combs, order]
            # num_combinations = tf.shape(previous_timesteps_mixed)[0]
            combs = tf.shape(previous_timesteps_mixed)[0]
            bools_timesteps_not_in_previous_timesteps_mixed = tf.map_fn(fn= lambda c: get_bools_not_in_previous_timesteps_mixed(c), elems= previous_timesteps_mixed, dtype= tf.bool) ## [combs, timesteps]
            indices_timesteps_not_in_previous_timesteps_mixed = tf.map_fn(fn= lambda bools: tf.where(bools), elems= bools_timesteps_not_in_previous_timesteps_mixed, dtype= tf.int64) ## [combs, timesteps - previous_order, 1]
            indices_timesteps_not_in_previous_timesteps_mixed = tf.cast(indices_timesteps_not_in_previous_timesteps_mixed, dtype= tf.int32)
            # indices_timesteps_not_in_previous_timesteps_mixed = tf.reshape(indices_timesteps_not_in_previous_timesteps_mixed, (combs, timesteps - previous_order)) ## [combs, timesteps - order]
            # indices_timesteps_not_in_previous_timesteps_mixed = tf.expand_dims(indices_timesteps_not_in_previous_timesteps_mixed, axis= -1) ## [combs, timesteps - order, 1]
            concat_indices_timesteps_not_in_previous_timesteps_mixed = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
            # for comb_idx in range(combs):
            for comb_idx in tf.range(combs):
                previous_mix = tf.gather(previous_timesteps_mixed, indices= comb_idx) ## [order]
                for step_idx in tf.range(timesteps - previous_order):
                    current_mix = tf.gather(tf.gather(indices_timesteps_not_in_previous_timesteps_mixed, comb_idx), step_idx)
                    concat_indices_timesteps_not_in_previous_timesteps_mixed = concat_indices_timesteps_not_in_previous_timesteps_mixed.write(comb_idx * (timesteps - previous_order) + step_idx, tf.concat([previous_mix, current_mix], axis= 0))
                # concat_indices_timesteps_not_in_previous_timesteps_mixed.write(comb_idx, tf.map_fn(fn= lambda indices_not_included: tf.concat([tf.gather(previous_timesteps_mixed, indices= comb_idx), indices_not_included], axis= 0), elems= tf.gather(indices_timesteps_not_in_previous_timesteps_mixed, indices= comb_idx))) ## [combs, order * (timesteps - order)]
            previous_timesteps_mixed = concat_indices_timesteps_not_in_previous_timesteps_mixed.stack() ## [combs * (timesteps - order), order + 1]
            # previous_timesteps_mixed = tf.concat(previous_timesteps_mixed, axis= 0)
            return previous_timesteps_mixed ## [combs * (timesteps - order), order + 1]

        def get_mixture_of_this_order(previous_timesteps_mixed):
            """
            
            previous_timesteps_mixed shape = [combs, indices]
            """

            inputs_mixed = tf.map_fn(fn= lambda indices: tf.gather(inputs_expanded, indices= indices), elems= previous_timesteps_mixed, dtype= tf.float32) ## [combs, indices, features]
            # inputs_mixed = tf.cast(inputs_mixed, tf.float32)
            inputs_mixed = tf.map_fn(fn= lambda x: tf.scan(fn= lambda a, x: tf.multiply(a, x), elems= x)[-1], elems= inputs_mixed)
            return K.sum(inputs_mixed, axis= 0)
        
        concat_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        previous_timesteps_mixed = tf.range(timesteps)
        previous_timesteps_mixed = tf.expand_dims(previous_timesteps_mixed, axis= -1)

        concat_ta = concat_ta.write(0, K.sum(inputs_expanded, axis= 0))
        for order in tf.range(start= 2, limit= self.order + 1):
        # for order in range(1, self.order):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(previous_timesteps_mixed, tf.TensorShape([None, None]))]
            )
            previous_timesteps_mixed = tf.cond(order <= timesteps, lambda: true_body(previous_timesteps_mixed, order), lambda: previous_timesteps_mixed)
            mixture_of_this_order = tf.cond(order <= timesteps, lambda: get_mixture_of_this_order(previous_timesteps_mixed), lambda: tf.constant(self.init_missing_order, dtype= tf.float32, shape= (self.num_features,)))

            concat_ta = concat_ta.write(order - 1, mixture_of_this_order)
        result = concat_ta.concat()
        # result = concat_ta.stack()
        result = tf.reshape(result, (1, self.order * self.num_features)) ## add batch dim, works in eager mode
        # result = Reshape(target_shape= (1, self.order * self.num_features))(result)

        return result

class TimeDense(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(TimeDense, self).__init__()

        self.dense = Dense(*args, **kwargs)
        self.concat = Concatenate(axis = -1)
    
    def call(self, inputs, RE_DATE):
        return self.dense(self.concat([inputs, RE_DATE]))

class PositionalEncoding(layers.Layer):
    def __init__(self, model_name, model_kwargs):
        self.model = keras_functions_dict[model_name](**model_kwargs)
    
    def call(self, time_stamps):
        return self.model(time_stamps)

class FeatureExtractor(layers.Layer):
    def __init__(self, extractor_kinds_list = None, num_sines_per_extractor = 2, num_extractors = 4, activation = None, dim= 3):
        super(FeatureExtractor, self).__init__()

        # self.extractor_functions_list = []
        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation
        self.num_extractors = num_extractors
        self.dim = dim

        if False:
            if extractor_kinds_list is None:
                self.extractor_kinds_list = [
                        dict(kind = "max"),
                        dict(kind = "min"),
                        dict(kind = "size"),
                    ]
            else:
                self.extractor_kinds_list = extractor_kinds_list

            for dct in self.extractor_kinds_list:
                if dct["kind"] == "max":
                    self.extractor_functions_list.append(lambda vrc, irc: tf.keras.backend.max(vrc))
                elif dct["kind"] == "min":
                    self.extractor_functions_list.append(lambda vrc, irc: tf.keras.backend.min(vrc))
                elif dct["kind"] == "size":
                    self.extractor_functions_list.append(lambda vrc, irc: tf.cast(K.shape(vrc)[0], dtype= tf.float32))

        self.sines_fre_amp_pha = tf.Variable(
            initial_value= tf.random_normal_initializer()(shape= (num_extractors, num_sines_per_extractor, 4), dtype= 'float32'),
            trainable= True
        )
        self.sines_bias = tf.Variable(
            initial_value= tf.zeros_initializer()(shape= (num_extractors,), dtype= 'float32'),
            trainable= True
        )

        # if False:
        #     for i in range(num_extractors):
        #         """
        #             self.sines_bias[i]: scalar,
        #             inputs[0]: scalar,
        #             inputs[1]: (2,)
        #             ext: (4,)

        #         """

        #         def sine_feature_funct(vrc, irc):
        #             def sines_on_pixel(inputs):
        #                 ## inputs = (value, [row-idx, column-idx])
        #                 # def sine_on_pixel(ext):
        #                 #     return ext[2] * tf.math.sin(ext[0] * inputs[1][0] + ext[1] * inputs[1][1] + ext[3])
        #                 # return inputs[0] * tf.reduce_sum(tf.map_fn(fn= sine_on_pixel, elems= self.sines_fre_amp_pha[i]))

        #                 ext = self.sines_fre_amp_pha[i]
        #                 weights = ext[:, 2] * tf.math.sin(ext[:, 0] * inputs[1][0] + ext[:, 1] * inputs[1][1] + ext[:, 3])
        #                 return inputs[0] * tf.reduce_sum(weights)
        #             return tf.reduce_sum(tf.map_fn(fn = sines_on_pixel, elems= (vrc, irc), fn_output_signature= tf.float32)) + self.sines_bias[i] ## dtype= (tf.float32, tf.int32)
        #         self.extractor_functions_list.append(sine_feature_funct)

        #         # self.extractor_functions_list.append(lambda vrc, irc: tf.reduce_sum(tf.map_fn(fn = lambda vrci, irci: vrci * tf.sum(tf.map_fn(fn= lambda ext: ext[2] * tf.math.sin(ext[0] * irci[0] + ext[1] * irci[1] + ext[3]), elems= self.sines_fre_amp_pha[i])), elems= (vrc, irc))) + self.sines_bias[i])

    def row_col_features_funct(self, inputs):
        """vrc, irc -> num_extractors
        """

        if self.dim == 2:
            vrc = inputs[0] ## when self.dim== 2 -> vrc: (width or height), when self.dim== 3 -> vrc: (width or height, num_channels)
            irc = inputs[1] ## (width or height, 2)
        else:
            irc = inputs ## outputs (width or height, num_extractors)
        
        if True:
            if False:
                def sine_feature_funct(inputs_ext):
                    ext = inputs_ext[0] ## (num_sines_per_extractor, 4)
                    bias = inputs_ext[1] ## ()
                    def sines_on_pixel(inputs_1):
                        value = inputs_1[0]
                        idx_0 = inputs_1[1][0]
                        idx_1 = inputs_1[1][1]
                        ## inputs_1 = (value, [row-idx, column-idx])
                        # def sine_on_pixel(ext):
                        #     return ext[2] * tf.math.sin(ext[0] * inputs_1[1][0] + ext[1] * inputs_1[1][1] + ext[3])
                        # return inputs_1[0] * tf.reduce_sum(tf.map_fn(fn= sine_on_pixel, elems= self.sines_fre_amp_pha[i]))

                        # ext = self.sines_fre_amp_pha[i]
                        weights = ext[:, 2] * tf.math.sin(ext[:, 0] * idx_0 + ext[:, 1] * idx_1 + ext[:, 3])
                        return value * tf.reduce_sum(weights)
                    return tf.reduce_sum(tf.map_fn(fn = sines_on_pixel, elems= (vrc, irc), fn_output_signature= tf.float32)) + bias ## dtype= (tf.float32, tf.int32)
            else:
                def sine_feature_funct(inputs_ext):
                    ext = inputs_ext[0] ## (num_sines_per_extractor, 4)
                    bias = inputs_ext[1] ## ()
                    def sines_on_pixel(inputs_1):
                        idx_0 = inputs_1[0]
                        idx_1 = inputs_1[1]

                        weights = ext[:, 2] * tf.math.sin(ext[:, 0] * idx_0 + ext[:, 1] * idx_1 + ext[:, 3])
                        return tf.reduce_sum(weights)
                    return tf.map_fn(fn = sines_on_pixel, elems= irc, fn_output_signature= tf.float32) + bias ## dtype= (tf.float32, tf.int32)

            # self.sines_fre_amp_pha: (num_extractors, num_sines_per_extractor, 4), self.sines_bias: (num_extractors,)
            return tf.map_fn(fn= sine_feature_funct, elems= (self.sines_fre_amp_pha, self.sines_bias), fn_output_signature= tf.float32, parallel_iterations= self.num_extractors) ## per extractor (feature)

    def get_feature_weights(self, input_shape):
        ## When self.dim == 2 inputs: (height, width), when self.dim == 3 inputs: (height, width, num_channels).
        ## xy_idx_pos: (height, width, 2) -> (height, width, self.num_extractors)
        
        X, Y = tf.meshgrid(tf.range(input_shape[1]), tf.range(input_shape[0]))
        output = tf.map_fn(elems= (X, Y), fn_output_signature= tf.float32)

    def call(self, inputs, axis, dim= 3, rel_pos = True):
        ## When self.dim == 2 inputs: (height, width), when self.dim == 3 inputs: (height, width, num_channels).
        input_shape = K.shape(inputs)
        Y, X = tf.meshgrid(tf.range(input_shape[1]), tf.range(input_shape[0]))
        X = tf.cast(X, dtype= tf.float32)
        Y = tf.cast(Y, dtype= tf.float32)
        if rel_pos:
            X = X / tf.cast(input_shape[0], dtype= tf.float32)
            Y = Y / tf.cast(input_shape[1], dtype= tf.float32)
        xy_idx_pos = tf.stack([X, Y], axis= 2)

        if axis == 1 and self.dim == 2:
            if dim== 2:
                inputs = tf.transpose(inputs)
            else:
                inputs = tf.transpose(inputs, perm= [1, 0, 2])
            xy_idx_pos = tf.transpose(xy_idx_pos, perm= [1, 0, 2])
        
        # def row_col_features_funct(inputs):
        #     vrc = inputs[0]
        #     irc = inputs[1]
            
        #     row_col = []
        #     for ext in self.extractor_functions_list:
        #         row_col.append(ext(vrc, irc))
        #     return tf.stack(row_col)
        #     # return tf.map_fn(fn= lambda ext: ext(vrc, irc), elems= self.extractor_functions_list, fn_output_signature= tf.float32)
        if self.dim == 2:
            return self.activation(tf.map_fn(fn= self.row_col_features_funct, elems= (inputs, xy_idx_pos), fn_output_signature= tf.float32))
        else:
            return self.activation(tf.map_fn(fn= self.row_col_features_funct, elems= xy_idx_pos, fn_output_signature= tf.float32)) ## outputs (height, width, num_extractors)

class AttentionReshaper(layers.Layer):
    def __init__(self, target_shape, attention_kwargs = None, extractor_kwargs = None, if_temporal = True, if_use_single_attention_layer_across_channels = True):
        super(AttentionReshaper, self).__init__()
        self.target_shape = target_shape
        if attention_kwargs is None:
            self.attention_kwargs = dict(num_heads= 3, key_dim= 32)
        else:
            self.attention_kwargs = attention_kwargs
        if extractor_kwargs is None:
            self.extractor_kwargs = dict()
        else:
            self.extractor_kwargs = extractor_kwargs
        self.feature_extractors = [FeatureExtractor(**self.extractor_kwargs), FeatureExtractor(**self.extractor_kwargs)]
        self.attention_layers = []
        self.if_temporal = if_temporal
        self.if_use_single_attention_layer_across_channels = if_use_single_attention_layer_across_channels
    
    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        num_attention_models = 1 if self.if_use_single_attention_layer_across_channels else self.num_channels
        for i in range(num_attention_models):
            attention_layers_ch = []
            for axis in (0, 1):
                attention_layers_ch.append(MultiHeadAttention(output_shape= self.target_shape[axis], **self.attention_kwargs))
            self.attention_layers.append(attention_layers_ch)
        # self.dummy_img_vars = tf.Variable(
        #     initial_value= tf.zeros_initializer()(shape= input_shape[1:], dtype= 'float32'),
        #     trainable= True
        # )
        self.dummy_img_init_var = tf.Variable(initial_value= 0.0, trainable= True, dtype= tf.float32)

    def reshape_att_channels(self, input_3d):
        ## input_3d: (height, width, channel)
        assert(self.if_use_single_attention_layer_across_channels)

        ## axis 0
        raise (NotImplementedError)
        extracted_features_channels = None

    def reshape_att(self, input_3d_batch):
        input_3d_batch_chfirst = tf.transpose(input_3d_batch, perm= [3, 0, 1, 2]) ## (batch == 1 (not timesteps), height, width, channel) -> (channel, batch == 1, height, width)
        def reshape_att_single_batch(inputs):
            transform_mat_batch = []
            chi = inputs[1]
            channel_attention_idx = 0 if self.if_use_single_attention_layer_across_channels else chi
            input_2d_batch = inputs[0] ## inputs[0] == (batch == 1, height, width)
            for axis in (0, 1):
                extracted_features_batch = tf.map_fn(fn= lambda input_2d: self.feature_extractors[axis](input_2d, axis= axis), elems= input_2d_batch) ## -> (batch = 1, height/width, features)
                attention_output_batch = self.attention_layers[channel_attention_idx][axis](query= extracted_features_batch, value= extracted_features_batch) ## (batch, width or height varying, fixed), (extracted_features, extracted_features) are (query and value) if they are same then self-attention.
                if axis == 0:
                    attention_output_batch = tf.transpose(attention_output_batch, perm= [0, 2, 1])
                transform_mat_batch.append(attention_output_batch)
            return tf.matmul(tf.matmul(transform_mat_batch[0], input_2d_batch), transform_mat_batch[1])
        return tf.map_fn(fn= reshape_att_single_batch, elems= (input_3d_batch_chfirst, tf.range(self.num_channels)), fn_output_signature= tf.float32, parallel_iterations= self.num_channels) ## -> (channel, batch == 1, height, width)

    
    def create_dummy_img(self, current_shape):
        dummy_imgs = tf.map_fn(lambda inputs: tf.constant(value= 1.0, shape= (self.target_shape[0], self.target_shape[1], self.num_channels), dtype= tf.float32), elems= tf.range(current_shape[0]), fn_output_signature= tf.float32) * self.dummy_img_init_var ## -> (batch, height, width, channel)
        # dummy_imgs = tf.map_fn(lambda inputs: self.dummy_img_vars, elems= tf.range(current_shape[0]), fn_output_signature= tf.float32) ## -> (batch, height, width, channel)
        return tf.transpose(dummy_imgs, perm= [3, 0, 1, 2]) ## -> (channel, batch, height, width)

    def reshape_att_wrap(self, input_3d_batch):
        ## (batch == 1, height, width, channel) ->
        current_shape = K.shape(input_3d_batch)
        return tf.cond(current_shape[1] == 0 or current_shape[2] == 0, lambda: self.create_dummy_img(current_shape), lambda: self.reshape_att(input_3d_batch))
        # return tf.cond(current_shape[1] == 0 or current_shape[2] == 0, lambda: self.create_dummy_img(current_shape), lambda: self.reshape_att_channels(input_3d_batch[0]))

    def call(self, inp):
        ## inp shape = (batch == 1, timesteps == None, height, width, channel)
        if self.if_temporal:
            inp = tf.transpose(inp, perm= [1, 0, 2, 3, 4]) ## (batch == 1, timesteps == None, height, width, channel) -> (timesteps == None, batch == 1, height, width, channel)
            inp = tf.map_fn(fn= self.reshape_att_wrap, elems= inp) ## -> (timesteps, channel, batch, height, width)
            return tf.transpose(inp, perm= [2, 0, 3, 4, 1]) ## -> (batch == 1, timesteps == None, fix-height, fix-width, channel)
        else:
            result = self.reshape_att_wrap(inp) ## (batch == 1, height, width, channel) -> (channel, batch == 1, height, width)
            return tf.transpose(result, perm= [1, 2, 3, 0])

            # output = []
            # for chi in range(self.num_channels):
            #     channel_attention_idx = 0 if self.if_use_single_attention_layer_across_channels else chi
            #     input_2d_batch = input_3d_batch[:, :, :, chi]
            #     transform_mat_batch = []
            #     for axis in [0, 1]:
            #         extracted_features_batch = tf.map_fn(fn= lambda input_2d: self.feature_extractors[axis](input_2d, axis= axis), elems= input_2d_batch)
            #         attention_output_batch = self.attention_layers[channel_attention_idx][axis](query= extracted_features_batch, value= extracted_features_batch) ## (width or height varying, fixed), (extracted_features, extracted_features) are (query and value) if they are same then self-attention.
            #         if axis == 0:
            #             attention_output_batch = tf.transpose(attention_output_batch, perm= [0, 2, 1])
            #         transform_mat_batch.append(attention_output_batch)
            #     output.append(transform_mat_batch[0] @ input_2d_batch @ transform_mat_batch[1])
            # return tf.stack(output, axis= -1)

        # if self.if_temporal:
        #     inp = tf.transpose(inp, perm= [1, 0, 2, 3, 4]) ## (batch == 1, timesteps == None, height, width, channel) -> (timesteps == None, batch == 1, height, width, channel)
        #     inp = tf.map_fn(fn= reshape_att, elems= inp) ## (timesteps == None, batch == 1, var-height, var-width, channel)
        #     return tf.transpose(inp, perm= [1, 0, 2, 3, 4]) ## (batch == 1, timesteps == None, var-height, var-width, channel)
        # else:
        #     return reshape_att(inp) ## inp: (batch == 1, var-height, var-width, channel)

class Conv2DMulti(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Conv2DMulti, self).__init__()
        self.conv_layer = Conv2D(*args, **kwargs)
    
    def call(self, inputs):
        ## input shape: (batch == 1, timesteps == None, height, width, channel)
        return tf.map_fn(fn= self.conv_layer, elems= inputs)

class MaxPooling2DMulti(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MaxPooling2DMulti, self).__init__()
        self.pooling_layer = MaxPooling2D(*args, **kwargs)
    
    def call(self, inputs):
        ## input shape: (batch == 1, timesteps == None, height, width, channel)
        return tf.map_fn(fn= self.pooling_layer, elems= inputs)

keras_functions_dict["TransformerTemporal"] = TransformerTemporal
keras_functions_dict["TimeDense"] = TimeDense
keras_functions_dict["HighOrderMixture"] = HighOrderMixture
keras_functions_dict["AttentionReshaper"] = AttentionReshaper
keras_functions_dict["Conv2DMulti"] = Conv2DMulti
keras_functions_dict["MaxPooling2DMulti"] = MaxPooling2DMulti

class VarShapeIntoFixConv(layers.Layer):
    def __init__(self, list_CNN, if_var_shape):
        super(VarShapeIntoFixConv, self).__init__()
        self.list_CNN_layers = []
        self.if_var_shape = if_var_shape
        self.additional_kwargs = dict(Conv2D= dict(kernel_regularizer= None, bias_regularizer= None))
        def kwargs_gen(key):
            if key in self.additional_kwargs.keys():
                return self.additional_kwargs[key]
            else:
                return dict()

        for elem in list_CNN:
            self.list_CNN_layers.append(keras_functions_dict[elem[0]](**elem[1], **kwargs_gen(elem[0])))
    
    def reshape_conv(self, inputs):
        if self.if_var_shape:
            img = inputs[0] ## inputs[0]: (height, width, channel)
            slice_idx_elem = inputs[1] ## (2, )
            # img = img[:, :slice_idx_elem[0], :slice_idx_elem[1], :]
            img = tf.slice(img, begin= [0, 0, 0], size= [slice_idx_elem[0], slice_idx_elem[1], 1])
        else:
            img = inputs
        img = tf.expand_dims(img, axis= 0)

        for layer in self.list_CNN_layers:
            img = layer(img)
        return img
    
    def call(self, inputs, slice_idx = None):
        """(1, timesteps, height, width, channel) -> """
        if self.if_var_shape:
            output = tf.map_fn(fn= self.reshape_conv, elems= (inputs[0], slice_idx[0]), fn_output_signature= tf.float32)
        else:
            output = tf.map_fn(fn= self.reshape_conv, elems= inputs[0], fn_output_signature= tf.float32)
        ## output == (timesteps, 1, height, width, channel)
        # return tf.expand_dims(output, axis= 0)
        return tf.transpose(output, perm= [1, 0, 2, 3, 4])

keras_functions_dict["VarShapeIntoFixConv"] = VarShapeIntoFixConv

class Autoencoder():
    name = "Autoencoder"
    def __init__(self, debug = 0, verbose = 0, small_delta = 1e-7, whether_use_mask = True, whether_use_static = True, whether_use_RE_DATE = True):
        """RNN Autoencoder Estimator"""

        self.debug = debug
        self.verbose = verbose
        self.small_delta = small_delta
        self.whether_use_mask = whether_use_mask
        self.whether_use_static = whether_use_static
        self.whether_use_RE_DATE = whether_use_RE_DATE
        self.additional_inputs_for_model_call = {key: dict() for key in keras_functions_dict.keys()}
        self.misc_kwargs = {}
        # self.if_ragged_tensor = False
    
    def fit(self, dataset, indices_of_patients, list_of_model_kwargs_RNN_vectors = None, dynamic_vectors_decoder_structure_list = None, predictor_structure_list = None, static_encoder_decoder_structure_dict = None, factors_dict = None, optimizer = None, loss_kind_dict = None, iters = 10, shuffle = True, run_eagerly = True, x_train = None, y_train = None, loss_wrapper_funct = None, loss_factor_funct = None, kwargs_RNN_images = None, conv_encoder_decoder_structure_dict = None, static_conv_encoder_decoder_structure_dict = None):
        """Train Autoencoder.
        
        Parameters
        ----------
        x_train, y_train: None.
            Dummy parameters, just leave them None, do not use.
        indices_of_patients : list of int
            Indices of participants to learn.
        """

        self.if_varying_shape_img = dataset.shape_2D_records is None
        self.if_output_recons_vecs = False
        self.if_output_recons_imgs = False
        # if self.if_varying_shape_img:
        #     shape_2D_records_loc = (None, None)
        # else:
        #     shape_2D_records_loc = dataset.shape_2D_records

        if self.whether_use_mask:
            record_length_factor = 2
            self.data_kind_key = "concat"
        else:
            record_length_factor = 1
            self.data_kind_key = "data"
        # if self.if_varying_shape_img: assert(self.data_kind_key != "concat")
        indices_of_patients_loc = deepcopy(indices_of_patients)
        print(f"Autoencoder fits on {len(indices_of_patients_loc)} samples.")
        output_dict = {"dynamic_vectors_decoder": None, "dynamic_images_decoder": None, "static_decoder": None, "predictor": None}
        inputs_list = []
        concatenated_representations = []

        ## Set default arguments for each kwargs.
        if list_of_model_kwargs_RNN_vectors is None: self.list_of_model_kwargs_RNN_vectors = [dict(model= "LSTM", kwargs= dict(units= 64))]
        elif isinstance(list_of_model_kwargs_RNN_vectors, dict): self.list_of_model_kwargs_RNN_vectors = [list_of_model_kwargs_RNN_vectors]
        else: self.list_of_model_kwargs_RNN_vectors = deepcopy(list_of_model_kwargs_RNN_vectors)
        # assert(("dropout" not in kwargs_RNN_vectors_local.keys() or kwargs_RNN_vectors_local["dropout"] == 0.) and ("recurrent_dropout" not in kwargs_RNN_vectors_local.keys() or kwargs_RNN_vectors_local["recurrent_dropout"] == 0.)) ## dropout largely degrade the performance, dropout may not work with predict_on_batch.
        self.factors_dict_local = merge_dictionaries([{"dynamic vectors reconstruction loss": 1.0, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 1.0, "static image reconstruction loss": 1.0, "prediction": 10.0}, factors_dict])
        self.loss_kind_dict = merge_dictionaries([{"dynamic vectors reconstruction loss": 2, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 2, "prediction": "binary cross entropy", "static image reconstruction loss": 2.0}, loss_kind_dict]) ## binary cross entropy works for multi-class classification with one-hot encoded label.
        if optimizer is None: optimizer = optimizers.Adam()

        ## --- --- Timeseries AE for vector inputs
        ## --- RNN Encoder (for vectors) structure
        vectors_RNN_input = Input(shape= (None, len(dataset.input_to_features_map["RNN_vectors"]) * record_length_factor), batch_size= 1, name= "vectors_RNN_input") ## shape = (batch_size = 1, time_steps = None, num_features), input shape of numpy array should be (1, time_steps, num_features)
        records_outputs = vectors_RNN_input
        num_vectors_hidden_dynamic_features = len(dataset.input_to_features_map["RNN_vectors"]) * record_length_factor

        for model_kwargs in self.list_of_model_kwargs_RNN_vectors:
            if model_kwargs["model"] in ["LSTM", "SimpleRNN"]:
                records_outputs, vectors_last_hidden_state, vectors_last_cell_state = keras_functions_dict[model_kwargs["model"]](return_sequences=True, return_state=True, **model_kwargs["kwargs"])(records_outputs, **self.additional_inputs_for_model_call[model_kwargs["model"]]) ## vectors_last_hidden_state's shape: (1, num_hidden_dynamic_features), the input shape looks like (batch_size = 1, time_steps, units) # batch_input_shape= (1, None, num_features_LSTM_input)
                num_vectors_hidden_dynamic_features = model_kwargs["kwargs"]["units"]
            elif model_kwargs["model"] == "GRU":
                records_outputs, vectors_last_hidden_state = keras_functions_dict[model_kwargs["model"]](return_sequences=True, return_state=True, **model_kwargs["kwargs"])(records_outputs, **self.additional_inputs_for_model_call[model_kwargs["model"]]) ## vectors_last_hidden_state's shape: (1, num_hidden_dynamic_features), the input shape looks like (batch_size = 1, time_steps, units) # batch_input_shape= (1, None, num_features_LSTM_input)
                num_vectors_hidden_dynamic_features = model_kwargs["kwargs"]["units"]
            elif model_kwargs["model"] == "TransformerTemporal":
                records_outputs = TransformerTemporal(input_dim= num_vectors_hidden_dynamic_features, **model_kwargs["kwargs"])(records_outputs, **self.additional_inputs_for_model_call[model_kwargs["model"]])
                vectors_last_hidden_state = records_outputs[:, -1, :]
                if "output_shape" in model_kwargs["kwargs"].keys(): num_vectors_hidden_dynamic_features = model_kwargs["kwargs"]["output_shape"]
            elif model_kwargs["model"] == "MultiInstanceTransformer":
                assert(model_kwargs["model_kwargs_temporal"] is None)
                for key in ["key_model_kwargs", "query_model_kwargs", "value_model_kwargs"]:
                    assert(model_kwargs[key]["model"] == "Dense")
                records_outputs = TransformerTemporal(input_dim= num_vectors_hidden_dynamic_features, **model_kwargs["kwargs"])(records_outputs, **self.additional_inputs_for_model_call[model_kwargs["model"]])
                vectors_last_hidden_state = HighOrderMixture()(inputs = records_outputs)
                if "output_shape" in model_kwargs["kwargs"].keys(): 
                    num_vectors_hidden_dynamic_features = model_kwargs["kwargs"]["output_shape"]
            else:
                raise Exception(NotImplementedError)
        
        ## Copy the last hidden representation. 
        # copies_vectors_last_hidden_state = RepeatVector(n = K.shape(vectors_RNN_input)[1])(last_hidden_state) ## Output shape : 3D tensor of shape (num_samples = 1, n, features), Input shape : 2D tensor of shape (num_samples = 1, features).
        if False:
            def repeat_vector(args): ## https://github.com/keras-team/keras/issues/7949
                layer_to_repeat = args[0]
                sequence_layer = args[1]
                return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)
            copies_vectors_last_hidden_state = Lambda(repeat_vector, output_shape = (None, num_vectors_hidden_dynamic_features))([vectors_last_hidden_state, vectors_RNN_input]) ## vectors_last_hidden_state (None= 1, num_features) -> (None = 1, None = time_steps, num_hidden_dynamic_features)
        else: ## ref: https://stackoverflow.com/questions/57587422/can-keras-repeatvector-repetition-be-specified-dynamically
            # timesteps_lambda = Lambda(
            #     lambda t: tf.shape(t)[1],
            #     name="timesteps_lambda"
            # )(vectors_RNN_input)

            copies_vectors_last_hidden_state = Lambda(
            lambda inputs: tf.tile(tf.expand_dims(inputs[0], 1), (1, tf.shape(inputs[1])[1], 1)),
            name="copies_vectors_last_hidden_state"
            )([vectors_last_hidden_state, vectors_RNN_input])
            if False: ## another method
                def repeat_vector(args): ## https://github.com/keras-team/keras/issues/7949
                    """
                    repeats encoding based on input shape of the model
                    :param input_tensors: [  tensor : encoding ,  tensor : sequential_input]
                    :return: tensor : encoding repeated input-shape[1]times
                    """
                    
                    sequential_input = args[1]
                    to_be_repeated = K.expand_dims(args[0],axis=1)

                    # set the one matrix to shape [ batch_size , sequence_length_based on input, 1]
                    one_matrix = K.ones_like(sequential_input[:,:,:1])
                    
                    # do a mat mul
                    return K.batch_dot(one_matrix,to_be_repeated)

        concatenation = [copies_vectors_last_hidden_state]

        ## --- RNN Decoder (for vectors) structure
        if self.whether_use_RE_DATE:
            arr_RE_DATE = Input(shape = (None, 1), batch_size= 1, name= "arr_RE_DATE") ## (batch_size= 1, time_steps, num_features = 1)
            self.additional_inputs_for_model_call["TimeDense"]["RE_DATE"] = arr_RE_DATE
            concatenation.append(arr_RE_DATE)

        copies_vectors_last_hidden_state_RE_DATE_concatenated = Concatenate(axis = -1)(concatenation) ## (1, time_steps, num_hidden_dynamic_features + 1).

        ### Decoder (for vectors) Reconstruction
        if dynamic_vectors_decoder_structure_list is None: ## Basic structure, if structure is not given.
            self.if_output_recons_vecs = False
            # dynamic_vectors_decoder_structure_list = [["Dense", {"units": 128, "activation": LeakyReLU(alpha=0.01), "activity_regularizer": basic_regularizer}], ["Dense", {"units": 70, "activation": LeakyReLU(alpha=0.01), "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset.input_to_features_map["dynamic_vectors_decoder"]), "activation": LeakyReLU(alpha=0.01), "activity_regularizer":  basic_regularizer}]]
        # assert(dynamic_vectors_decoder_structure_list[-1][0] == "Dense")
        # assert(dynamic_vectors_decoder_structure_list[0][0] == "Dense")
        else:
            self.if_output_recons_vecs = True
            dynamic_vectors_decoder_structure_list[-1][1]["units"] = len(dataset.input_to_features_map["dynamic_vectors_decoder"]) ## output shape (1, time_steps, len(dataset.input_to_features_map["dynamic_vectors_decoder"])).

            output_dict["dynamic_vectors_decoder"] = copies_vectors_last_hidden_state_RE_DATE_concatenated ## https://keras.io/api/layers/core_layers/dense/
            for layer in dynamic_vectors_decoder_structure_list: ## output_dict["dynamic_vectors_decoder"] shape (1, time_steps, len(dataset.input_to_features_map["dynamic_vectors_decoder"])).
                if layer is not None: output_dict["dynamic_vectors_decoder"] = keras_functions_dict[layer[0]](**layer[1])(output_dict["dynamic_vectors_decoder"], **self.additional_inputs_for_model_call[layer[0]]) ## Output of final layer's shape: (batch_size = 1, time_steps, len(dataset.input_to_features_map["dynamic_vectors_decoder"])).

        ## --- --- Timeseries AE for 2D inputs
        if dataset.if_contains_imgs: ## Dataset contains images.
            ## Set default settings
            images_RNN_input = Input(shape= (None, dataset.shape_2D_records[0] * record_length_factor, dataset.shape_2D_records[1], dataset.image_info["RNN_2D"]["num_rgb"]) if not self.if_varying_shape_img else (None, None, None, 1), batch_size= 1, ragged= False, name= "images_RNN_input") ## shape = (batch_size = 1, time_steps = None, num_rows (multiplied twice if concat), num_cols, 1)
            # kwargs_RNN_images_local = merge_dictionaries([{"units": 64}, kwargs_RNN_images])
            if self.if_varying_shape_img:
                slice_idx_RNN_input = Input(shape= (None, 2), batch_size= 1, name= "slice_idx_RNN_input")
                self.additional_inputs_for_model_call["VarShapeIntoFixConv"]["slice_idx"] = slice_idx_RNN_input
            output_dict["dynamic_images_decoder"] = images_RNN_input
            img_timesteps = Input(shape= (), batch_size= 1, name= "img_timesteps", dtype= tf.int32)
            self.additional_inputs_for_model_call["HighOrderMixture"]["timesteps"] = img_timesteps
            inputs_list.append(img_timesteps)
            images_hidden_states_LSTM = None

            if conv_encoder_decoder_structure_dict is None: ## Default argument.
                conv_encoder_decoder_structure_dict = {}
                # conv_encoder_decoder_structure_dict["encoder"] = [["Conv2D", dict(filters= 32, kernel_size= (5, 5), stride= (2, 2), activation= "relu")], ["MaxPooling2D", dict(pool_size= (2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (5, 5), stride= (2, 2), activation= "relu")], ["MaxPooling2D", dict(pool_size= (2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (3, 3), stride= (1, 1), activation= "relu")], ["Flatten", dict()]] # ["Flatten", dict()]

                conv_encoder_decoder_structure_dict["encoder"] = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (2, 2), padding= "valid", activation= "tanh", return_sequences= True)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (2, 2), padding= "valid", activation= "tanh", return_sequences= True)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (2, 2), padding= "valid", activation= "tanh", return_sequences= False)], ["TimeDistributed", dict(layer= Flatten())], ["LSTM", dict(units= 64, return_sequences=True, return_state=True)]] ## For MaxPooling3D, pool_size = (1, 2, 2) means no pooling (1) along time steps, and (2, 2) pooling along width and height.

                conv_encoder_decoder_structure_dict["decoder"] = [["Dense", {"units": 100, "activation": lambda x: activations.relu(x, alpha = 0.1)}], ["Dense", {"units": 169, "activation": lambda x: activations.relu(x, alpha = 0.1)}], ["TimeDistributed", dict(layer = Reshape(target_shape= (13, 13, 1)))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (2, 2), padding= "valid", activation= "tanh"))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5))]] ## Conv2DTranspose, (batch_size (time_steps), rows, cols, channels) -> (batch_size, new_rows, new_cols, filters).
                # ["Reshape", dict(target_shape= (kwargs_RNN_images_local["units"] - kwargs_RNN_images_local["units"] // 2, kwargs_RNN_images_local["units"] // 2))]
            self.conv_encoder_decoder_structure_dict = deepcopy(conv_encoder_decoder_structure_dict)
            self.if_output_recons_imgs = not self.conv_encoder_decoder_structure_dict["decoder"] is None
            
            ## Image Records Encoding
            for layer_idx in range(len(conv_encoder_decoder_structure_dict["encoder"])):
                layer_ = conv_encoder_decoder_structure_dict["encoder"][layer_idx]
                if self.verbose >= 1: print(f"2D Encoding: {layer_[0]}, previous shape: {output_dict['dynamic_images_decoder'].shape}")

                additional_init_kwargs = {}
                additional_inputs_for_model_init_local = {}
                if layer_[0] == "HighOrderMixture":
                    HighOrderMixture_features = output_dict["dynamic_images_decoder"].shape[-1]
                    additional_init_kwargs["num_features"] = output_dict["dynamic_images_decoder"].shape[-1]
                if layer_[0] == "TransformerTemporal":
                    additional_inputs_for_model_init_local["input_dim"] = output_dict["dynamic_images_decoder"].shape[-1]
                if layer_[0] == "Dense" and layer_idx > 1 and conv_encoder_decoder_structure_dict["encoder"][layer_idx - 1][0] == "HighOrderMixture":
                    additional_inputs_for_model_init_local["input_shape"] = (1, HighOrderMixture_features)
                if layer_[0] == "VarShapeIntoFixConv":
                    additional_inputs_for_model_init_local["if_var_shape"] = self.if_varying_shape_img

                output_dict["dynamic_images_decoder"] = keras_functions_dict[layer_[0]](**layer_[1], **additional_init_kwargs, **additional_inputs_for_model_init_local)(output_dict["dynamic_images_decoder"], **self.additional_inputs_for_model_call[layer_[0]])
            if isinstance(output_dict["dynamic_images_decoder"], (tuple, list)):
                images_hidden_states_LSTM, images_last_hidden_state, images_last_cell_state = output_dict["dynamic_images_decoder"]
            else:
                images_last_hidden_state = output_dict["dynamic_images_decoder"]
            if images_hidden_states_LSTM is not None:
                output_dict["images_last_hidden_state"] = images_last_hidden_state

            if self.if_output_recons_imgs and not self.if_varying_shape_img:
                ## Copy the last hidden representation.
                # copies_images_last_hidden_state = Lambda(repeat_vector, output_shape = (None, conv_encoder_decoder_structure_dict["encoder"][-1][1]["units"]))([images_last_hidden_state, images_RNN_input]) ## (None = 1, None = time_steps, num_hidden_dynamic_features)
                if False:
                    concatenation_images_last_hidden_state = [Lambda(repeat_vector, output_shape = (None, conv_encoder_decoder_structure_dict["encoder"][-1][1]["units"]))([images_last_hidden_state, images_RNN_input])] ## (None = 1, None = time_steps, num_hidden_dynamic_features)
                else:
                    concatenation_images_last_hidden_state = [Lambda(
                    lambda inputs: tf.tile(tf.expand_dims(inputs[0], 1), (1, tf.shape(inputs[1])[1], 1)),
                    name="concatenation_images_last_hidden_state"
                    )([images_last_hidden_state, images_RNN_input])]
                    if False: ## another method
                        def repeat_vector(args): ## https://github.com/keras-team/keras/issues/7949
                            """
                            repeats encoding based on input shape of the model
                            :param input_tensors: [  tensor : encoding ,  tensor : sequential_input]
                            :return: tensor : encoding repeated input-shape[1]times
                            """
                            
                            sequential_input = args[1]
                            to_be_repeated = K.expand_dims(args[0],axis=1)

                            # set the one matrix to shape [ batch_size , sequence_length_based on input, 1]
                            one_matrix = K.ones_like(sequential_input[:,:,:1])
                            
                            # do a mat mul
                            return K.batch_dot(one_matrix,to_be_repeated)
                if self.whether_use_RE_DATE:
                    concatenation_images_last_hidden_state.append(arr_RE_DATE)

                ## Additional inputs for image decoder
                if "decoder_input" in conv_encoder_decoder_structure_dict.keys():
                    if conv_encoder_decoder_structure_dict["decoder_input"]["input"] == "images_RNN_input_masked":
                        self.misc_kwargs["keep_rate"] = conv_encoder_decoder_structure_dict["decoder_input"]["keep_rate"]
                        decoder_additional_input = Input(shape= (None, dataset.shape_2D_records[0] * record_length_factor, dataset.shape_2D_records[1], dataset.image_info["RNN_2D"]["num_rgb"]), batch_size= 1, ragged= False, name= "images_RNN_input_masked") ## shape = (batch_size = 1, time_steps = None, num_rows (multiplied twice if concat), num_cols, 1)
                        inputs_list.append(decoder_additional_input)
                    else:
                        raise Exception(NotImplementedError)
                    decoder_additional_input = decoder_additional_input[0] ## remove dummy batch dim.
                    for layer_ in conv_encoder_decoder_structure_dict["decoder_input"]["layers"]:
                        decoder_additional_input = keras_functions_dict[layer_[0]](**layer_[1])(decoder_additional_input, **self.additional_inputs_for_model_call[layer_[0]])
                    decoder_additional_input = K.expand_dims(decoder_additional_input, axis= 0) ## restore dummy batch dim.
                    concatenation_images_last_hidden_state.append(decoder_additional_input)
                
                ## Image Records Decoding
                copies_images_last_hidden_state_RE_DATE_concatenated = Concatenate(axis = -1)(concatenation_images_last_hidden_state) ## (1 (deleted by [0]), time_steps, num_hidden_dynamic_features + 1).
                output_dict["dynamic_images_decoder"] = copies_images_last_hidden_state_RE_DATE_concatenated
                for layer_ in conv_encoder_decoder_structure_dict["decoder"]:
                    if self.verbose >= 1: print(f"2D Decoding: {layer_[0]}, previous shape: {output_dict['dynamic_images_decoder'].shape}")
                    if layer_[0] == "Conv2DTranspose-fit-to-image-shape":
                        if False:
                            strides= [0, 0]
                            kernel_size = layer_[1]["kernel_size"]
                            for axis in [0, 1]:
                                # kernel_size[axis] = dataset.shape_2D_records[axis] % (output_dict["dynamic_images_decoder"].shape[1 + axis] - 1)
                                # strides[axis] = dataset.shape_2D_records[axis] // (output_dict["dynamic_images_decoder"].shape[1 + axis] - 1)
                                strides[axis] = (dataset.shape_2D_records[axis] - kernel_size[axis]) // (output_dict["dynamic_images_decoder"].shape[2 + axis] - 1)
                                if (dataset.shape_2D_records[axis] - kernel_size[axis]) % (output_dict["dynamic_images_decoder"].shape[2 + axis] - 1) != 0: strides[axis] += 1
                        strides = utils.get_strides_to_target_output_shape(output_shape = dataset.shape_2D_records, input_shape= (output_dict["dynamic_images_decoder"].shape[2], output_dict["dynamic_images_decoder"].shape[3]), kernel_size= layer_[1]["kernel_size"])
                        output_dict["dynamic_images_decoder"] = TimeDistributed(layer= Conv2DTranspose(filters= 1, strides = strides, padding= "valid", **layer_[1]))(output_dict["dynamic_images_decoder"]) ## Reconstruct the image with similar (slightly larger or equal) shape as input image.
                    else:
                        output_dict["dynamic_images_decoder"] = keras_functions_dict[layer_[0]](**layer_[1])(output_dict["dynamic_images_decoder"], **self.additional_inputs_for_model_call[layer_[0]])
                output_dict["dynamic_images_decoder"] = output_dict["dynamic_images_decoder"][:, :, :dataset.shape_2D_records[0], :dataset.shape_2D_records[1], :] ## Cut the remaining paddings to keep the same shape as input image.
                reconstructed_imgs = output_dict["dynamic_images_decoder"]
                assert(output_dict["dynamic_images_decoder"].shape[2] == dataset.shape_2D_records[0] and output_dict["dynamic_images_decoder"].shape[3] == dataset.shape_2D_records[1]) ## Same shape as input image.

            concatenated_representations.append(images_last_hidden_state)

        ### Predictor structure
        if predictor_structure_list is None: ## Basic structure, if structure is not given.
            predictor_structure_list = [["Dense", {"units": 80, "activation": LeakyReLU(alpha=0.01), "activity_regularizer": basic_regularizer}], ["Dense", {"units": 40, "activation": LeakyReLU(alpha=0.01), "activity_regularizer":  basic_regularizer}], ["Dense", {"units": 1, "activation": "sigmoid", "activity_regularizer":  basic_regularizer}]]
        assert(predictor_structure_list[-1][0] == "Dense" and predictor_structure_list[-1][1]["units"] == len(dataset.prediction_labels_bag))
        # assert(predictor_structure_list[0][0] == "Dense")

        ## --- --- About static features.
        ## --- Static Encoding.
        ## Set default structure.
        def num_neurons(factor): ## number of neurons for default argument.
            return max(1, round(factor * len(dataset.input_to_features_map["static_encoder"])))
        if static_encoder_decoder_structure_dict is None: ## Default argument.
            static_encoder_decoder_structure_dict = {}
            static_encoder_decoder_structure_dict["encoder"] = [["Dense", {"units": num_neurons(0.6), "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": num_neurons(0.2), "activation": "tanh", "activity_regularizer":  basic_regularizer}],  ["Dense", {"units": num_neurons(0.1), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]
            static_encoder_decoder_structure_dict["decoder"] = [["Dense", {"units": num_neurons(0.2), "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": num_neurons(0.6), "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]
        assert(static_encoder_decoder_structure_dict["decoder"][-1][1]["units"] == len(dataset.input_to_features_map["static_encoder"])) ## For reconstruction.

        ## Start Static Encoding.
        ## Static info is concatenated to input of predictor.
        static_encoder_input = Input(shape = (len(dataset.input_to_features_map["static_encoder"]) * record_length_factor, ), batch_size= 1, name= "static_encoder_input")
        static_encoder_data = Input(shape = (len(dataset.input_to_features_map["static_encoder"]), ), batch_size= 1, name= "static_encoder_data")
        static_encoder_mask = Input(shape = (len(dataset.input_to_features_map["static_encoder"]), ), batch_size= 1, name= "static_encoder_mask")
        encoded_static_vector = static_encoder_input
        if len(dataset.input_to_features_map["static_encoder"]) > 0:
            for layer in static_encoder_decoder_structure_dict["encoder"]:
                encoded_static_vector = keras_functions_dict[layer[0]](**layer[1])(encoded_static_vector, **self.additional_inputs_for_model_call[layer[0]]) ## output encoded_static_vector in shape (1, num_hidden_static_features)
            ## Static Decoding.
            output_dict["static_decoder"] = encoded_static_vector
            for layer in static_encoder_decoder_structure_dict["decoder"]:
                output_dict["static_decoder"] = keras_functions_dict[layer[0]](**layer[1])(output_dict["static_decoder"], **self.additional_inputs_for_model_call[layer[0]]) ## output encoded_static_vector in shape (num_hidden_features, )
            concatenated_representations.append(encoded_static_vector)
        
        ## static IMAGE encode-decode.
        if static_conv_encoder_decoder_structure_dict is not None:
            static_image_encoder_data_shape = deepcopy(list(dataset.shape_2D_record_static))
            if self.data_kind_key == "concat":
                static_image_encoder_data_shape[0] *= 2
            static_image_encoder_data = Input(shape = static_image_encoder_data_shape, batch_size= 1, name= "static_image_encoder_data")
            inputs_list.append(static_image_encoder_data)
            encoded_static_image_vector = static_image_encoder_data
            for layer in static_conv_encoder_decoder_structure_dict["encoder"]:
                encoded_static_image_vector = keras_functions_dict[layer[0]](**layer[1])(encoded_static_image_vector, **self.additional_inputs_for_model_call[layer[0]])
            concatenated_representations.append(encoded_static_image_vector)
            
            ## Static Decoding.
            output_dict["static_image_decoder"] = encoded_static_image_vector
            for layer_ in static_conv_encoder_decoder_structure_dict["decoder"]:
                if self.verbose >= 1: print(f"Static 2D Image Decoding: {layer_[0]}, previous shape: {output_dict['static_image_decoder'].shape}")
                target_shape = list(deepcopy(dataset.shape_2D_record_static))
                if self.data_kind_key == "concat":
                    target_shape[0] *= 2
                if layer_[0] == "Conv2DTranspose-fit-to-image-shape":
                    strides = utils.get_strides_to_target_output_shape(output_shape = (target_shape[0], target_shape[1]), input_shape= (output_dict["static_image_decoder"].shape[1], output_dict["static_image_decoder"].shape[2]), kernel_size= layer_[1]["kernel_size"])
                    Conv2DTranspose(filters= 1, strides = strides, padding= "valid", **layer_[1])(output_dict["static_image_decoder"])
                    output_dict["static_image_decoder"] = Conv2DTranspose(filters= target_shape[2], strides = strides, padding= "valid", **layer_[1])(output_dict["static_image_decoder"]) ## Reconstruct the image with similar (slightly larger or equal) shape as input image.
                else:
                    output_dict["static_image_decoder"] = keras_functions_dict[layer_[0]](**layer_[1])(output_dict["static_image_decoder"], **self.additional_inputs_for_model_call[layer_[0]])
            
            output_dict["static_image_decoder"] = output_dict["static_image_decoder"][:, :target_shape[0], :target_shape[1], :] ## Cut the remaining paddings to keep the same shape as input image.
            # static_reconstructed_image = output_dict["static_image_decoder"]
            assert(output_dict["static_image_decoder"].shape[1] == target_shape[0] and output_dict["static_image_decoder"].shape[2] == target_shape[1]) ## Same shape as input image.

        static_raw = Input(shape = (len(dataset.input_to_features_map["raw"]) * record_length_factor, ), batch_size= 1, name= "static_raw")
        if len(dataset.input_to_features_map["raw"]) > 0 and self.whether_use_static:
            concatenated_representations.append(static_raw)

        ## Concatenate Encoded Static Vector and Raw Vector to input of Predictor.
        concatenated_representations.append(vectors_last_hidden_state)
        
        if len(concatenated_representations) > 1: 
            output_dict["predictor"] = Concatenate(axis = -1)(concatenated_representations) ## encoded_static_vector's shape: (1, num_hidden_static_features), final shape = (1, num_hidden_static_features + num_static_raw_features + num_hidden_dynamic_features). If static_encoder_input is zero shape (,0) then encoded_static_vector is also zero shape, but can be concatenated without syntax error.
        else:
            output_dict["predictor"] = concatenated_representations[0]

        ## Build Predictor.
        for layer in predictor_structure_list:
            if layer is not None: output_dict["predictor"] = keras_functions_dict[layer[0]](**layer[1])(output_dict["predictor"], **self.additional_inputs_for_model_call[layer[0]]) ## Output of final layer's shape: (1, 1).
        
        ## --- --- Define model.
        ## --- Define inputs for labels.
        ## For Decoder.
        if self.if_output_recons_vecs:
            decoder_vectors_labels_data = Input(shape = (None, len(dataset.input_to_features_map["dynamic_vectors_decoder"])), batch_size= 1, name= "decoder_vectors_labels_data") ## (1, time steps, features).
            decoder_vectors_labels_mask = Input(shape = (None, len(dataset.input_to_features_map["dynamic_vectors_decoder"])), batch_size= 1, name= "decoder_vectors_labels_mask") ## (1, time steps, features).
            inputs_list += [decoder_vectors_labels_data, decoder_vectors_labels_mask]
        if self.if_output_recons_imgs:
            if dataset.if_contains_imgs:
                decoder_images_labels_data = Input(shape = (None, dataset.shape_2D_records[0], dataset.shape_2D_records[1], dataset.image_info["RNN_2D"]["num_rgb"]), batch_size= 1, ragged= False, name= "decoder_images_labels_data") ## (1, time steps, features).
                decoder_images_labels_mask = Input(shape = (None, dataset.shape_2D_records[0], dataset.shape_2D_records[1], dataset.image_info["RNN_2D"]["num_rgb"]), batch_size= 1, ragged= False, name= "decoder_images_labels_mask") ## (1, time steps, features).
            if False and dataset.if_contains_static_img:
                decoder_image_static_data = Input(shape = (dataset.shape_2D_records[0], dataset.shape_2D_records[1], 1), batch_size= 1, ragged= False, name= "decoder_image_static_mask") ## (1, time steps, features).
                decoder_image_static_mask = Input(shape = (dataset.shape_2D_records[0], dataset.shape_2D_records[1], 1), batch_size= 1, ragged= False, name= "decoder_image_static_mask") ## (1, time steps, features).

        ## For Predictor.
        predictor_label = Input(shape = (len(dataset.prediction_labels_bag), ), batch_size= 1, name= "predictor_label") ## *** Assuming one-hot encoded label, shape = (1, number of classes).
        predictor_label_observability = Input(shape = (1, ), batch_size= 1, name= "predictor_label_observability") ## shape = (1, 1), {1., 0.}.

        inputs_list += [vectors_RNN_input, predictor_label, predictor_label_observability, static_encoder_input, static_encoder_data, static_encoder_mask, static_raw]
        if dataset.if_partial_label:
            label_tasks_mask = Input(shape = (len(dataset.prediction_labels_bag), ), batch_size= 1, name= "label_tasks_mask")
            inputs_list.append(label_tasks_mask)
        else:
            label_tasks_mask = None
        if self.whether_use_RE_DATE:
            inputs_list.append(arr_RE_DATE)
        outputs_list = [vectors_last_hidden_state, output_dict["predictor"]]
        self.outputs_names_list = ["vectors_last_hidden_state", "predictor"]
        if self.if_output_recons_vecs:
            outputs_list.append(output_dict["dynamic_vectors_decoder"])
            self.outputs_names_list.append("dynamic_vectors_decoder")
        if dataset.if_contains_imgs:
            if self.if_output_recons_imgs:
                inputs_list.append(decoder_images_labels_data)
                inputs_list.append(decoder_images_labels_mask)
                if not self.if_varying_shape_img:
                    outputs_list.append(reconstructed_imgs)
                    self.outputs_names_list.append("reconstructed_imgs")
            inputs_list.append(images_RNN_input)
            outputs_list.append(images_last_hidden_state)
            self.outputs_names_list.append("images_last_hidden_state")
            if self.if_varying_shape_img:
                inputs_list.append(slice_idx_RNN_input)
        
        if dataset.if_contains_static_img and static_conv_encoder_decoder_structure_dict is not None:
            outputs_list.append(output_dict["static_image_decoder"])
            self.outputs_names_list.append("static_image_decoder")
        
        if dataset.if_contains_imgs or (dataset.if_contains_static_img and static_conv_encoder_decoder_structure_dict is not None):
            self.heatmap_model = Model(inputs_list, [output_dict["predictor"]])

        assert(len(outputs_list) == len(self.outputs_names_list))
        self.model = Model(inputs = inputs_list, outputs = outputs_list) ## Output = enriched representation vector, predicted target, reconstructed original representations matrix.
        self.model.trainable = True
        self.model.run_eagerly = run_eagerly ## has not great effect in debugging, self.model.compile(optimizer= optimizer, run_eagerly= run_eagerly) is more important for eager mode.

        ## Plot model, because eager tensor(output tensor calculated by non-Layer function, such as + * - /) cannot be plotted, this should be located before adding losses.
        plot_model(self.model, to_file= "./outputs/misc/model_diagram.png", show_shapes= True)

        ### Define losses.
        if self.if_output_recons_vecs:
            self.model.add_loss(self.dynamic_vectors_reconstruction_loss(decoder_vectors_labels_data = decoder_vectors_labels_data, decoder_vectors_labels_mask = decoder_vectors_labels_mask, decoder_vectors_labels_data_predicted = output_dict["dynamic_vectors_decoder"], factor = self.factors_dict_local["dynamic vectors reconstruction loss"], kind= self.loss_kind_dict["dynamic vectors reconstruction loss"]))
            self.model.add_metric(self.dynamic_vectors_reconstruction_loss(decoder_vectors_labels_data = decoder_vectors_labels_data, decoder_vectors_labels_mask = decoder_vectors_labels_mask, decoder_vectors_labels_data_predicted = output_dict["dynamic_vectors_decoder"], factor = self.factors_dict_local["dynamic vectors reconstruction loss"], kind= self.loss_kind_dict["dynamic vectors reconstruction loss"]), name = "Dynamic Vectors Reconstruction Error")
        if dataset.if_contains_imgs and self.if_output_recons_imgs and not self.if_varying_shape_img:
            self.model.add_loss(self.dynamic_images_reconstruction_loss(decoder_images_labels_data = decoder_images_labels_data, decoder_images_labels_mask = decoder_images_labels_mask, decoder_images_labels_data_predicted = output_dict["dynamic_images_decoder"], factor = self.factors_dict_local["dynamic images reconstruction loss"], kind= self.loss_kind_dict["dynamic images reconstruction loss"]))
            self.model.add_metric(self.dynamic_images_reconstruction_loss(decoder_images_labels_data = decoder_images_labels_data, decoder_images_labels_mask = decoder_images_labels_mask, decoder_images_labels_data_predicted = output_dict["dynamic_images_decoder"], factor = self.factors_dict_local["dynamic images reconstruction loss"], kind= self.loss_kind_dict["dynamic images reconstruction loss"]), name = "Dynamic Images Reconstruction Error")
        if static_conv_encoder_decoder_structure_dict is not None:
            self.model.add_loss(self.static_reconstruction_loss(origina_static_data = static_image_encoder_data, origina_static_mask = None, predicted_static_data = output_dict["static_image_decoder"], factor = self.factors_dict_local["static image reconstruction loss"], kind = self.loss_kind_dict["static image reconstruction loss"]))
            self.model.add_metric(self.static_reconstruction_loss(origina_static_data = static_image_encoder_data, origina_static_mask = None, predicted_static_data = output_dict["static_image_decoder"], factor = self.factors_dict_local["static image reconstruction loss"], kind = self.loss_kind_dict["static image reconstruction loss"]), name = "Static Image Reconstruction Error")
        if len(dataset.input_to_features_map["static_encoder"]) > 0:
            self.model.add_loss(self.static_reconstruction_loss(origina_static_data = static_encoder_data, origina_static_mask = static_encoder_mask, predicted_static_data = output_dict["static_decoder"], factor = self.factors_dict_local["static reconstruction loss"], kind = self.loss_kind_dict["static reconstruction loss"]))
            self.model.add_metric(self.static_reconstruction_loss(origina_static_data = static_encoder_data, origina_static_mask = static_encoder_mask, predicted_static_data = output_dict["static_decoder"], factor = self.factors_dict_local["static reconstruction loss"], kind = self.loss_kind_dict["static reconstruction loss"]), name = "Static Reconstruction Error")
        self.model.add_loss(self.prediction_loss(predictor_label_predicted = output_dict["predictor"], predictor_label = predictor_label, predictor_label_observability = predictor_label_observability, factor = self.factors_dict_local["prediction"], kind= self.loss_kind_dict["prediction"], loss_wrapper_funct = loss_wrapper_funct, loss_factor_funct = loss_factor_funct, label_tasks_mask= label_tasks_mask))
        self.model.add_metric(self.prediction_loss(predictor_label_predicted = output_dict["predictor"], predictor_label = predictor_label, predictor_label_observability = predictor_label_observability, factor = self.factors_dict_local["prediction"], kind= self.loss_kind_dict["prediction"], loss_wrapper_funct = loss_wrapper_funct, loss_factor_funct = loss_factor_funct, label_tasks_mask= label_tasks_mask), name = "Prediction Error")

        ## Compile model.
        self.model.compile(optimizer= optimizer, run_eagerly= run_eagerly) ## This run_eagerly is very useful in debugging tf/keras models.
        if self.verbose >= 2:
            self.model.summary()

        ## Get dictionary of losses to check convergence.
        loss_dict = {name: [] for name in self.model.metrics_names}
        loss_dict["loss"] = []
        ## Train model.
        print("Start autoencoder training..")
        for it in tqdm(range(iters)):
            if shuffle: np.random.shuffle(indices_of_patients_loc) 
            for patient_idx in indices_of_patients_loc:
                # print(f"fit with patientid: {dataset.idx_key_map[patient_idx]}.")
                input_dict = {"vectors_RNN_input": dataset.dicts[patient_idx]["RNN_vectors"][self.data_kind_key], "decoder_vectors_labels_data": dataset.dicts[patient_idx]['dynamic_vectors_decoder']["data"], "decoder_vectors_labels_mask": dataset.dicts[patient_idx]['dynamic_vectors_decoder']["mask"], "predictor_label": dataset.dicts[patient_idx]["predictor label"], "predictor_label_observability": dataset.dicts[patient_idx]["observability"], "static_encoder_input": dataset.dicts[patient_idx]["static_encoder"][self.data_kind_key], "static_encoder_data": dataset.dicts[patient_idx]["static_encoder"]["data"], "static_encoder_mask": dataset.dicts[patient_idx]["static_encoder"]["mask"], "static_raw": dataset.dicts[patient_idx]["raw"][self.data_kind_key]}
                if self.whether_use_RE_DATE:
                    input_dict["arr_RE_DATE"] = dataset.dicts[patient_idx]["RE_DATE"]
                if dataset.if_contains_imgs:
                    if dataset.if_separate_img:
                        assert(dataset.dicts[patient_idx]["sample_idx"] == patient_idx)
                        RNN_2D_dict= dataset.load_arr(sample_idx= patient_idx, input_= "RNN_2D", data_type = None)
                    else:
                        RNN_2D_dict= dataset.dicts[patient_idx]["RNN_2D"]
                    input_dict["images_RNN_input"] = RNN_2D_dict[self.data_kind_key]
                    input_dict["img_timesteps"] = np.array([RNN_2D_dict[self.data_kind_key].shape[1]])
                    if self.if_varying_shape_img:
                        input_dict["slice_idx_RNN_input"] = dataset.dicts[patient_idx]["RNN_2D"]["slice_idx"][self.data_kind_key]
                    if self.if_output_recons_imgs and not self.if_varying_shape_img:
                        input_dict["decoder_images_labels_data"] = RNN_2D_dict["data"]
                        input_dict["decoder_images_labels_mask"] = RNN_2D_dict["mask"]
                        if "decoder_input" in conv_encoder_decoder_structure_dict.keys():
                            if conv_encoder_decoder_structure_dict["decoder_input"]["input"] == "images_RNN_input_masked":
                                input_dict["images_RNN_input_masked"] = self.get_images_RNN_input_masked(images_RNN_input = RNN_2D_dict[self.data_kind_key])
                            else:
                                raise Exception(NotImplementedError)
                if static_conv_encoder_decoder_structure_dict is not None and dataset.if_contains_static_img:
                    if dataset.if_separate_img:
                        assert(dataset.dicts[patient_idx]["sample_idx"] == patient_idx)
                        static_2D_dict= dataset.load_arr(sample_idx= patient_idx, input_= "static_2D", data_type = None)
                    else:
                        static_2D_dict= dataset.dicts[patient_idx]["static_2D"]
                    input_dict["static_image_encoder_data"] = static_2D_dict[self.data_kind_key]
                    if False and self.if_output_recons_imgs and not self.if_varying_shape_img:
                        input_dict["decoder_images_labels_data"] = RNN_2D_dict["data"]
                        input_dict["decoder_images_labels_mask"] = RNN_2D_dict["mask"]
                        if "decoder_input" in conv_encoder_decoder_structure_dict.keys():
                            if conv_encoder_decoder_structure_dict["decoder_input"]["input"] == "images_RNN_input_masked":
                                input_dict["images_RNN_input_masked"] = self.get_images_RNN_input_masked(images_RNN_input = RNN_2D_dict[self.data_kind_key])
                            else:
                                raise Exception(NotImplementedError)
                if dataset.if_partial_label:
                    input_dict["label_tasks_mask"] = dataset.dicts[patient_idx]["label_tasks_mask"]
                else:
                    input_dict["label_tasks_mask"] = None

                metric_loss_dict_single_batch = self.model.train_on_batch(x = input_dict, y = {}, return_dict= True)
                for metric in metric_loss_dict_single_batch.keys():
                    if metric != "Prediction Error" or (metric == "Prediction Error" and dataset.dicts[patient_idx]["observability"][0][0] == 1.): ## For prediction loss, in training set, label is provided.
                        loss_dict[metric].append(metric_loss_dict_single_batch[metric])
            if self.verbose >= 1:
                print(metric_loss_dict_single_batch)
                # print(f"Decayed Learning Rate: {self.model.optimizer._decayed_lr('float32').numpy()}, Learning Rate: {float(self.model.optimizer.learning_rate)}")
        return loss_dict
    
    def predict(self, dataset, indices_of_patients, perturbation_info_for_feature_importance = None, swap_info_for_feature_importance = None, feature_importance_calculate_prob_dict = None, x_test = None, num_recons_images = 3):
        """Predict target label, reconstructed data.

        Parameters
        ----------
        dataset : Dataset
            Dataset object.
        perturbation_info_for_feature_importance : dict
            For example, perturbation_info_for_feature_importance = {"center": "mean" or 0., "proportion": 1.0}. If None, then do not plot feature importance by perturbation method.
        swap_info_for_feature_importance : dict
            For example, swap_info_for_feature_importance = {"proportion": 0.7 <= 1.0}.
        feature_importance_calculate_prob_dict : dict
            For example, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}.
        x_test : None
            Dummy variable, just leave it None, Do not use.
        
        Attributes
        ----------
        """

        ## Set default arguments
        feature_importance_calculate_prob_dict = merge_dictionaries([{"dynamic": 1.0, "static": 1.0}, feature_importance_calculate_prob_dict])

        indices_of_patients_loc = deepcopy(indices_of_patients)
        # indices_of_patients_for_recons_images = sample(indices_of_patients_loc, num_recons_images)
        indices_of_patients_for_recons_images = indices_of_patients_loc[:num_recons_images]
        indices_of_patients_for_recons_images_to_discard = []
        print(f"Autoencoder predicts on {len(indices_of_patients_loc)} samples.")
        dicts_input_to_keras_input_map = dict(RNN_vectors = "vectors_RNN_input", static_encoder= "static_encoder_input", raw= "static_raw")
        
        self.model.trainable = False

        ## Output: Set results containers
        enriched_vectors_stack = []
        predicted_labels_stack = []
        if self.if_output_recons_vecs: reconstructed_vectors_stack = []
        enriched_vectors_for_images_dict = dict()
        reconstructed_images_dict = dict()
        gradients_images_dict = dict()
        static_reconstructed_image_dict = dict()
        static_gradients_image_dict = dict()
        original_img = dict()

        ## Output: Feature importance
        feature_importance_dict = {}
        if perturbation_info_for_feature_importance is not None: feature_importance_dict["perturbation"] = {group_name: {feature_name: [] for feature_name in dataset.groups_of_features_info[group_name].keys()} for group_name in dataset.groups_of_features_info.keys()}
        if swap_info_for_feature_importance is not None: feature_importance_dict["swap"] = {group_name: {feature_name: [] for feature_name in dataset.groups_of_features_info[group_name].keys()} for group_name in dataset.groups_of_features_info.keys()}

        for patient_idx in tqdm(indices_of_patients_loc): ## The particiapant's sequence of predicted_labels_stack is same as the sequence of indices_of_patients_loc.
            # if dataset.dataset_kind.split("-")[0] == "chestxray":
            #     patientid = dataset.idx_key_map[patient_idx]

            original_inputs_dict = deepcopy({"vectors_RNN_input": dataset.dicts[patient_idx]["RNN_vectors"][self.data_kind_key], "decoder_vectors_labels_data": dataset.dicts[patient_idx]['dynamic_vectors_decoder']["data"], "decoder_vectors_labels_mask": dataset.dicts[patient_idx]['dynamic_vectors_decoder']["mask"], "predictor_label": dataset.dicts[patient_idx]["predictor label"], "predictor_label_observability": dataset.dicts[patient_idx]["observability"], "static_encoder_input": dataset.dicts[patient_idx]["static_encoder"][self.data_kind_key], "static_encoder_data": dataset.dicts[patient_idx]["static_encoder"]["data"], "static_encoder_mask": dataset.dicts[patient_idx]["static_encoder"]["mask"], "static_raw": dataset.dicts[patient_idx]["raw"][self.data_kind_key]})
            if self.whether_use_RE_DATE:
                original_inputs_dict["arr_RE_DATE"] = dataset.dicts[patient_idx]["RE_DATE"]
            if dataset.if_contains_imgs:
                if dataset.if_separate_img:
                    assert(dataset.dicts[patient_idx]["sample_idx"] == patient_idx)
                    RNN_2D_dict= dataset.load_arr(sample_idx= patient_idx, input_= "RNN_2D", data_type = None)
                else:
                    RNN_2D_dict= dataset.dicts[patient_idx]["RNN_2D"]
                data_height_RNN = RNN_2D_dict["data"].shape[2]
                original_inputs_dict["images_RNN_input"] = deepcopy(RNN_2D_dict[self.data_kind_key])
                original_inputs_dict["img_timesteps"] = np.array([RNN_2D_dict[self.data_kind_key].shape[1]])
                if self.if_varying_shape_img:
                    original_inputs_dict["slice_idx_RNN_input"] = dataset.dicts[patient_idx]["RNN_2D"]["slice_idx"][self.data_kind_key]
                if self.if_output_recons_imgs and not self.if_varying_shape_img:
                    original_inputs_dict["decoder_images_labels_data"] = deepcopy(RNN_2D_dict["data"])
                    original_inputs_dict["decoder_images_labels_mask"] = deepcopy(RNN_2D_dict["mask"])
                    if "decoder_input" in self.conv_encoder_decoder_structure_dict.keys():
                        if self.conv_encoder_decoder_structure_dict["decoder_input"]["input"] == "images_RNN_input_masked":
                            original_inputs_dict["images_RNN_input_masked"] = self.get_images_RNN_input_masked(images_RNN_input = RNN_2D_dict[self.data_kind_key])
                        else:
                            raise Exception(NotImplementedError)
            
            if dataset.if_partial_label:
                original_inputs_dict["label_tasks_mask"] = dataset.dicts[patient_idx]["label_tasks_mask"]
            else:
                original_inputs_dict["label_tasks_mask"] = None

            if dataset.if_contains_static_img:
                data_height_static = dataset.dicts[patient_idx]["static_2D"]["data"].shape[1]
                original_inputs_dict["static_image_encoder_data"] = deepcopy(dataset.dicts[patient_idx]["static_2D"][self.data_kind_key])

            ## Normal Prediction only for Prediction
            # enriched_vector, predicted_label, reconstructed_vectors = self.model.predict_on_batch(x = original_inputs_dict) ## outputs_list = [enriched_vector, predicted_label, reconstructed_vectors, reconstructed_images]
            outputs_list = self.model.predict_on_batch(x = original_inputs_dict)
            predicted_labels_stack.append(outputs_list[self.outputs_names_list.index("predictor")][0]) ## predicted_label[0] = [0.01, 0.98, ...]

            enriched_vectors_stack.append(outputs_list[self.outputs_names_list.index("vectors_last_hidden_state")][0])
            if self.if_output_recons_vecs:
                reconstructed_vectors_stack.append(outputs_list[self.outputs_names_list.index("dynamic_vectors_decoder")][0])
            
            if dataset.if_contains_imgs:
                if patient_idx in indices_of_patients_for_recons_images:
                    if self.if_output_recons_imgs and not self.if_varying_shape_img:
                        reconstructed_images_dict[patient_idx] = outputs_list[self.outputs_names_list.index("reconstructed_imgs")][0]
                    # original_images_dict[patient_idx] = deepcopy(dataset.dicts[patient_idx]["RNN_2D"]["data"])
                    enriched_vectors_for_images_dict[patient_idx] = outputs_list[self.outputs_names_list.index("images_last_hidden_state")][0]
                        
            if dataset.if_contains_static_img:
                if patient_idx in indices_of_patients_for_recons_images:
                    if self.if_output_recons_imgs:
                        static_reconstructed_image_dict[patient_idx] = outputs_list[self.outputs_names_list.index("static_image_decoder")][0]
                    # enriched_vectors_for_images_dict[patient_idx] = outputs_list[self.outputs_names_list.index("static_image_decoder")][0]
            
            if hasattr(self, "heatmap_model") and (False or patient_idx in indices_of_patients_for_recons_images):
                keys_input_for_gradient = []
                if dataset.if_contains_imgs: keys_input_for_gradient.append("images_RNN_input")
                if dataset.if_contains_static_img: keys_input_for_gradient.append("static_image_encoder_data")
                for key_input_for_gradient in keys_input_for_gradient:
                    with tf.GradientTape() as gtape:
                        input_for_gradient = tf.convert_to_tensor(original_inputs_dict[key_input_for_gradient])
                        gtape.watch(input_for_gradient)
                        tensor_original_inputs_dict = {}
                        for key in original_inputs_dict.keys():
                            if isinstance(original_inputs_dict[key], np.ndarray) and key != key_input_for_gradient: 
                                tensor_original_inputs_dict[key] = tf.convert_to_tensor(original_inputs_dict[key])
                            else: 
                                tensor_original_inputs_dict[key] = original_inputs_dict[key]
                        tensor_original_inputs_dict[key_input_for_gradient] = input_for_gradient
                        prediction_result = tf.cast(x= self.heatmap_model(tensor_original_inputs_dict), dtype=tf.float64)
                        prediction_loss = tf.convert_to_tensor(self.prediction_loss(predictor_label_predicted = prediction_result, predictor_label = tf.convert_to_tensor(dataset.dicts[patient_idx]["predictor label"]), predictor_label_observability = tf.cast(x= tf.convert_to_tensor([[1.0]]), dtype= tf.float64), factor = self.factors_dict_local["prediction"], kind= self.loss_kind_dict["prediction"], label_tasks_mask= original_inputs_dict["label_tasks_mask"]))
                        grads = gtape.gradient(prediction_loss, input_for_gradient)
                        if not grads is None:
                            if key_input_for_gradient == "images_RNN_input":
                                gradients_images_dict[patient_idx] = tf.math.abs(grads).numpy()[0, :, :data_height_RNN, :, :]
                            elif key_input_for_gradient == "static_image_encoder_data":
                                static_gradients_image_dict[patient_idx] =  tf.math.abs(grads).numpy()[0, :data_height_static, :, :]
                                image_shape = list(static_gradients_image_dict[patient_idx].shape)
                        else:
                            indices_of_patients_for_recons_images_to_discard.append(patient_idx)
                            print(f"WARNING: Gradients are None for patientid: {dataset.idx_key_map[patient_idx]}, maybe because of too small image in attention-reshaper.")
                        # print(grads.numpy())

            ### Calculate Feature Importance.
            whether_calculate_importance_static = random() < feature_importance_calculate_prob_dict["static"]
            whether_calculate_importance_dynamic = random() < feature_importance_calculate_prob_dict["dynamic"]
            for feature_group in dataset.groups_of_features_info.keys():
                for feature_name in dataset.groups_of_features_info[feature_group].keys():
                    ## Set variable of dict for convenience.
                    feature_info_dict = dataset.groups_of_features_info[feature_group][feature_name] ## {"input": self.feature_to_input_map[name], "idx": self.input_to_features_map[self.feature_to_input_map[name]].index(name), "observed_numbers": [], "mean": None, "std": None}
                    feature_idx = feature_info_dict["idx"]
                    keras_input = dicts_input_to_keras_input_map[feature_info_dict["input"]]
                    dicts_input = feature_info_dict["input"]

                    feature_importance_changed_input = {} ## Changed input: for "perturbation", "swap" each.
                    if feature_info_dict["input"] in ["static_encoder", "raw"] and whether_calculate_importance_static and dataset.dicts[patient_idx][feature_info_dict["input"]]["mask"][0][feature_idx] == 1.: ## For static input features, and change dataset.dicts[patient_idx]["static_features_data"], only in case observed.
                        ## Prepare the changed inputs for each method, as the input is the only difference.
                        if perturbation_info_for_feature_importance is not None: ## "perturbation" input
                            if perturbation_info_for_feature_importance["center"] == "mean": perturbation_loc = feature_info_dict["mean"]
                            else: perturbation_loc = perturbation_info_for_feature_importance["center"]
                            perturbation_scalar = perturbation_info_for_feature_importance["proportion"] * np.random.normal(loc = perturbation_loc, scale= feature_info_dict["std"], size = 1)[0] ## PERTURBATION: gaussian: scalar.
                            feature_importance_changed_input["perturbation"] = deepcopy(dataset.dicts[patient_idx][dicts_input][self.data_kind_key])
                            feature_importance_changed_input["perturbation"][0][feature_idx] += perturbation_scalar ## Input after perturbation.
                        if swap_info_for_feature_importance is not None: ## "swap" input
                            feature_importance_changed_input["swap"] = deepcopy(dataset.dicts[patient_idx][dicts_input][self.data_kind_key])
                            feature_importance_changed_input["swap"][0][feature_idx] = sample(feature_info_dict["observed_numbers"], k= 1)[0]
                        ## Calculate the changes in the prediction, with the changed input for each method.
                        for method in feature_importance_changed_input.keys():
                            outputs_changed_list = self.model.predict_on_batch(x = merge_dictionaries([original_inputs_dict, {keras_input: feature_importance_changed_input[method]}])) ## Output after input changes, merge_dictionaries uses shallow copy so does not change original_dict.
                            changes_on_prediction = np.sum(np.abs((outputs_list[1] - outputs_changed_list[1])[0])) ## [0] for batch dim.
                            feature_importance_dict[method][feature_group][feature_name].append(changes_on_prediction)

                    elif feature_info_dict["input"] in ["RNN_vectors"] and whether_calculate_importance_dynamic and np.sum(dataset.dicts[patient_idx][feature_info_dict["input"]]["mask"][0, :, feature_idx]) >= 1.: ## Find features in RNN input, and change dataset.dicts[patient_idx][self.key_name_LSTM_input], only in case observed.
                        num_timesteps = dataset.dicts[patient_idx][dicts_input]["data"].shape[1] ## self.key_name_LSTM_input == "LSTM inputs data and mask concatenated" or "LSTM inputs data"
                        ## Prepare the changed inputs for each method, as the input is the only difference.
                        if perturbation_info_for_feature_importance is not None: ## "perturbation" input
                            if perturbation_info_for_feature_importance["center"] == "mean": perturbation_loc = feature_info_dict["mean"]
                            else: perturbation_loc = perturbation_info_for_feature_importance["center"]
                            perturbation_time_steps = perturbation_info_for_feature_importance["proportion"] * np.random.normal(loc = perturbation_loc, scale= feature_info_dict["std"], size = (num_timesteps, )) * dataset.dicts[patient_idx][dicts_input]["mask"][0, :, feature_idx] ## PERTURBATION: gaussian: (time_steps, ) * mask: (time_steps, )
                            feature_importance_changed_input["perturbation"] = deepcopy(dataset.dicts[patient_idx][dicts_input][self.data_kind_key]) ## self.key_name_LSTM_input == "LSTM inputs data and mask concatenated" or "LSTM inputs data"
                            feature_importance_changed_input["perturbation"][0, :, feature_idx] += perturbation_time_steps ## Input after perturbation.
                        if swap_info_for_feature_importance is not None: ## "swap" input
                            mask_to_swap = mask_prob(shape= (num_timesteps, ), p= swap_info_for_feature_importance["proportion"]) * dataset.dicts[patient_idx][dicts_input]["mask"][0, :, feature_idx] ## vector in shape (time_steps, ), 1 if entry to swap, 0 if entry to keep the original.
                            observed_numbers_pool = feature_info_dict["observed_numbers"]
                            sample_indices = np.random.choice(len(observed_numbers_pool), (num_timesteps, ), replace=True) ## Randomly choose the indices.
                            sampled_record = np.array([observed_numbers_pool[sample_indices[t]] for t in range(num_timesteps)]) ## Fully populated sampled numbers.
                            feature_importance_changed_input["swap"] = deepcopy(dataset.dicts[patient_idx][dicts_input][self.data_kind_key]) ## self.key_name_LSTM_input == "LSTM inputs data and mask concatenated" or "LSTM inputs data"
                            feature_importance_changed_input["swap"][0, :, feature_idx] = sampled_record * mask_to_swap + feature_importance_changed_input["swap"][0, :, feature_idx] * (1. - mask_to_swap)
                        ## Calculate the changes in the prediction, with the changed input for each method.
                        for method in feature_importance_changed_input.keys():
                            outputs_changed_list = self.model.predict_on_batch(x = merge_dictionaries([original_inputs_dict, {keras_input: feature_importance_changed_input[method]}])) ## Output after input changes, merge_dictionaries uses shallow copy so does not change original_dict.
                            changes_on_prediction = np.sum(np.abs((outputs_list[1] - outputs_changed_list[1])[0])) / np.sum(dataset.dicts[patient_idx][dicts_input]["mask"][0, :, feature_idx])
                            feature_importance_dict[method][feature_group][feature_name].append(changes_on_prediction)

        self.enriched_vectors_stack = np.array(enriched_vectors_stack)
        self.predicted_labels_stack = np.array(predicted_labels_stack)
        self.predicted_labels_stack = predicted_labels_stack
        predictions_dict = {"enriched_vectors_stack": self.enriched_vectors_stack, "predicted_labels_stack": self.predicted_labels_stack, "feature_importance_dict": feature_importance_dict}

        if self.if_output_recons_vecs:
            self.reconstructed_vectors_stack = reconstructed_vectors_stack ## cannot stack to numpy array, because the arrays have different shapes due to the different time steps.
            predictions_dict["reconstructed_vectors_stack"] = reconstructed_vectors_stack

        if dataset.if_contains_imgs or dataset.if_contains_static_img:
            predictions_dict["indices_of_patients_for_recons_images"] = [idx for idx in indices_of_patients_for_recons_images if not idx in indices_of_patients_for_recons_images_to_discard]

        if dataset.if_contains_imgs:
            predictions_dict.update({"gradients_images_dict": gradients_images_dict})
            # indices_of_patients_for_recons_images = filter(lambda elem: elem not in indices_of_patients_for_recons_images_to_discard, indices_of_patients_for_recons_images)
            if self.if_output_recons_imgs and not self.if_varying_shape_img:
                predictions_dict.update({"reconstructed_images_dict": reconstructed_images_dict})
                if True:
                    predictions_dict.update({"original_img": original_img})
        if dataset.if_contains_static_img:
            predictions_dict.update({"static_gradients_image_dict": static_gradients_image_dict})
            if self.if_output_recons_imgs:
                predictions_dict.update({"static_reconstructed_image_dict": static_reconstructed_image_dict})
        return predictions_dict
    
    def get_images_RNN_input_masked(self, images_RNN_input):
        if False:
            is_tensor = tf.is_tensor(images_RNN_input)
            if is_tensor:
                if False:
                    images_RNN_input_masked = tf.identity(images_RNN_input) ## shape = (batch_size = 1, time_steps = None, num_rows (multiplied twice if concat), num_cols, 1)
                    imgs_shape = images_RNN_input_masked.shape
                    imgs_shape = tf.shape(images_RNN_input_masked)

                    mask = mask_prob(shape = imgs_shape, p = self.misc_kwargs["keep_rate"], is_tensor= is_tensor)
                    mask_half_0, mask_half_1 = tf.split(mask, num_or_size_splits= 2, axis = 2)
                    input_half_0, input_half_1 = tf.split(images_RNN_input_masked, num_or_size_splits= 2, axis = 2)
                    return tf.concat([input_half_0 * mask_half_0, input_half_1 * mask_half_0], axis= 2)
                images_RNN_input_masked = tf.identity(images_RNN_input)
                images_RNN_input_masked[0] = tf.map_fn(fn = lambda img: mask_single_img_tf(img, self.misc_kwargs["keep_rate"], if_concat = self.data_kind_key == "concat"), elems = images_RNN_input_masked[0])
                return images_RNN_input_masked
            else:
                images_RNN_input_masked = np.copy(images_RNN_input) ## shape = (batch_size = 1, time_steps = None, num_rows (multiplied twice if concat), num_cols, 1)
                imgs_shape = deepcopy(list(images_RNN_input_masked.shape))

                if self.data_kind_key == "data":
                    mask = mask_prob(shape = imgs_shape, p = self.misc_kwargs["keep_rate"], is_tensor= is_tensor)
                    images_RNN_input_masked = mask * images_RNN_input_masked
                elif self.data_kind_key == "concat":
                    imgs_shape[2] = imgs_shape[2] // 2
                    mask = mask_prob(shape = imgs_shape, p = self.misc_kwargs["keep_rate"], is_tensor= is_tensor)
                    images_RNN_input_masked[:, :, :imgs_shape[2], :, :] = images_RNN_input_masked[:, :, :imgs_shape[2], :, :] * mask
                    images_RNN_input_masked[:, :, imgs_shape[2]:, :, :] = images_RNN_input_masked[:, :, imgs_shape[2]:, :, :] * mask
                    return images_RNN_input_masked
                else:
                    raise Exception(NotImplementedError)
        else:
            if self.if_varying_shape_img:
                raise Exception(NotImplementedError)
                assert(isinstance(images_RNN_input, list))
                images_RNN_input_masked = []
                for ts in range(len(images_RNN_input[0])):
                    images_RNN_input_masked.append(self.get_masked_arr(nparr= images_RNN_input[0][ts], height_axis = 0))
                images_RNN_input_masked = [images_RNN_input_masked]
                images_RNN_input_masked = tf.ragged.constant(images_RNN_input_masked)
                return tf.expand_dims(images_RNN_input_masked, axis= 0) ## dummy batch dim.
            else:
                return self.get_masked_arr(nparr= images_RNN_input, height_axis = 2)
                # images_RNN_input_masked = np.copy(images_RNN_input) ## shape = (batch_size = 1, time_steps = None, num_rows (multiplied twice if concat), num_cols, 1)
                # imgs_shape = deepcopy(list(images_RNN_input_masked.shape))
                # if self.data_kind_key == "data":
                #     mask = mask_prob(shape = imgs_shape, p = self.misc_kwargs["keep_rate"], is_tensor= False)
                #     images_RNN_input_masked = mask * images_RNN_input_masked
                # elif self.data_kind_key == "concat":
                #     imgs_shape[2] = imgs_shape[2] // 2
                #     mask = mask_prob(shape = imgs_shape, p = self.misc_kwargs["keep_rate"], is_tensor= False)
                #     images_RNN_input_masked[:, :, :imgs_shape[2], :, :] = images_RNN_input_masked[:, :, :imgs_shape[2], :, :] * mask
                #     images_RNN_input_masked[:, :, imgs_shape[2]:, :, :] = images_RNN_input_masked[:, :, imgs_shape[2]:, :, :] * mask
                #     return images_RNN_input_masked

    def get_masked_arr(self, nparr, height_axis= 2):
        arr_shape = deepcopy(list(nparr.shape))
        if self.data_kind_key == "data":
            mask = mask_prob(shape = arr_shape, p = self.misc_kwargs["keep_rate"], is_tensor= False)
            masked_arr = mask * nparr
        elif self.data_kind_key == "concat":
            masked_arr = np.copy(nparr)
            arr_shape[height_axis] = arr_shape[height_axis] // 2
            mask = mask_prob(shape = arr_shape, p = self.misc_kwargs["keep_rate"], is_tensor= False)
            if height_axis == 2:
                masked_arr[:, :, :arr_shape[height_axis], :, :] = masked_arr[:, :, :arr_shape[height_axis], :, :] * mask
                masked_arr[:, :, arr_shape[height_axis]:, :, :] = masked_arr[:, :, arr_shape[height_axis]:, :, :] * mask
            elif height_axis == 0:
                masked_arr[:arr_shape[height_axis], :, :] = masked_arr[:arr_shape[height_axis], :, :] * mask
                masked_arr[arr_shape[height_axis]:, :, :] = masked_arr[arr_shape[height_axis]:, :, :] * mask
            else:
                raise Exception(NotImplementedError)
        return masked_arr

    def clear_model(self):
        """Delete self.model to save it. Without deleting model, strange error occurs when saving because of RepeatVector layer."""

        K.clear_session()
        del self.model
    
    def dynamic_vectors_reconstruction_loss(self, decoder_vectors_labels_data, decoder_vectors_labels_mask, decoder_vectors_labels_data_predicted, factor = 1.0, kind = 0.5):
        """
        
        Parameters
        ----------
        decoder_vectors_labels_data_predicted : tensor
            It's shape is (1, time_steps, num_features_decoder_labels).
        """

        if isinstance(kind, (int, float, complex)) and not isinstance(kind, bool):
            p = kind
            loss = K.sum(decoder_vectors_labels_mask * K.abs(decoder_vectors_labels_data - decoder_vectors_labels_data_predicted) ** p) / (K.sum(decoder_vectors_labels_mask) + self.small_delta)
        else:
            raise Exception(NotImplementedError)

        return factor * loss
    
    def dynamic_images_reconstruction_loss(self, decoder_images_labels_data, decoder_images_labels_mask, decoder_images_labels_data_predicted, factor = 1.0, kind = 0.5):
        """
        
        Parameters
        ----------
        decoder_images_labels_data_predicted : tensor
            It's shape is (1, time_steps, rows, columns, channels).
        """

        if isinstance(kind, (int, float, complex)) and not isinstance(kind, bool):
            p = kind
            loss = K.sum(decoder_images_labels_mask * K.abs(decoder_images_labels_data - decoder_images_labels_data_predicted) ** p) / (K.sum(decoder_images_labels_mask) + self.small_delta)
        else:
            raise Exception(NotImplementedError)

        return factor * loss
    
    def static_reconstruction_loss(self, origina_static_data, origina_static_mask, predicted_static_data, factor = 1.0, kind = 0.5):
        """
        
        Parameters
        ----------
        origina_static_data, origina_static_mask, predicted_static_data : tensor
            It's shape is (1, num_features_static).
        """

        if isinstance(kind, (int, float, complex)) and not isinstance(kind, bool):
            p = kind
            if origina_static_mask is None:
                loss = K.sum(K.abs(origina_static_data - predicted_static_data) ** p) / (origina_static_data.shape[1] * origina_static_data.shape[2])
            else:
                loss = K.sum(origina_static_mask * K.abs(origina_static_data - predicted_static_data) ** p) / (K.sum(origina_static_mask) + self.small_delta)
        else:
            raise Exception(NotImplementedError)

        return factor * loss

    def prediction_loss(self, predictor_label_predicted, predictor_label, predictor_label_observability, factor = 1.0, kind = "binary cross entropy", loss_wrapper_funct = None, loss_factor_funct = None, label_tasks_mask = None):
        """
        
        Parameters
        ----------
        predictor_label_observability : Tensor in shape (1, 1)
        """
        if label_tasks_mask is not None and kind == "SquaredHinge":
            raise Exception(NotImplementedError)

        if loss_wrapper_funct is None: loss_wrapper_funct = lambda x: x
        if loss_factor_funct is None: loss_factor_funct = lambda **kwargs: 1.
        if label_tasks_mask is None: label_tasks_mask = K.ones_like(predictor_label)

        if kind == "binary cross entropy":
            loss = - 1.0 * K.sum(loss_wrapper_funct((predictor_label * K.log(predictor_label_predicted + self.small_delta) + (1. - predictor_label) * K.log(1 - predictor_label_predicted + self.small_delta)) * label_tasks_mask)) ## log(smaller than 1) is negative, log(larger than 1) is positive, because of the self.small_delta log(something) can be positive or negative both.
        elif kind == "SquaredHinge":
            # loss = square(maximum(1 - y_true * y_pred, 0))
            loss = SquaredHinge()(predictor_label * 2. - 1., predictor_label_predicted * 2. - 1.)
        elif isinstance(kind, (int, float, complex)) and not isinstance(kind, bool): ## p is number, p-norm.
            p = kind
            loss = K.sum(loss_wrapper_funct((K.abs(predictor_label - predictor_label_predicted) * label_tasks_mask) ** p))
        else:
            raise Exception(NotImplementedError)
        
        loss = loss_factor_funct(predictor_label_predicted = predictor_label_predicted, predictor_label = predictor_label, predictor_label_observability = predictor_label_observability, factor = factor, kind = kind) * loss

        return factor * loss * predictor_label_observability[0][0]
    
def mask_single_img_tf(img, p, if_concat = False):
    """
    img : tensorflow tensor
        3D array of (height, width, and channel (1)).
    """
    # img_masked = tf.identity(img)
    img_masked = img ## already copied

    height = img.shape[0]
    width = img[0].shape[0]

    if if_concat:
        half_idx = height // 2
        mask = tf.random.uniform(shape = (half_idx, width, 1), minval= 0.0, maxval= 1.0)
        mask = mask < p
        img_masked[:half_idx] = img_masked[:half_idx] * mask
        img_masked[half_idx:] = img_masked[half_idx:] * mask
    else:
        mask = tf.random.uniform(shape = (height, width, 1), minval= 0.0, maxval= 1.0)
        mask = mask < p
        img_masked = img_masked * mask
    return img_masked

def schatten_p_norm_regularizer(weights):
    p = 1.0
    s, u, v = tf.linalg.svd(weights)
    return tf.reduce_sum(u ** p)

    # tf_a_approx and np_a_approx should be numerically close.            