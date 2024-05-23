seeds = {"np": 2, "tf": 4}
from statistics import mean, stdev
from numpy.random import seed
seed(seeds["np"])
import tensorflow
tensorflow.random.set_seed(seeds["tf"])

import data_prep
import autoencoder
import pickle
import utils
import estimator
from utilsforminds.containers import GridSearch, Grid
from autoencoder import schatten_p_norm_regularizer

from tensorflow.keras import optimizers
from keras import regularizers
from keras.layers import ReLU, Dropout, BatchNormalization, Concatenate, RepeatVector, Dense, LSTM, Input, LeakyReLU, Lambda, GRU, SimpleRNN, Flatten, Conv2DTranspose, ConvLSTM2D, MaxPooling3D, TimeDistributed, Reshape, Subtract, Add, Multiply

from keras import layers
from keras import activations
import numpy as np
from collections import Counter

## TODO list
# - LSTM Encoder is generator, stack of encoded representations is pass to the discriminator (another LSTM, outputs the certainty vector) with concatenated time stamps.

basic_regularizer = None # regularizers.l1(0.01), None
basic_strong_regularizer = None # regularizers.l1(1.0)
basic_activation = lambda x: activations.relu(x, alpha = 0.01) # activations.tanh, lambda x: activations.relu(x, alpha = 0.1)
num_neurons_increase_factor = 0.8
dropout_rate = 0.0 ## 0.4
dropout_or_batchnormalization_layer = ["Dropout", {"rate": dropout_rate}] # ["Dropout", {"rate": 0.5}], ["BatchNormalization", {}]
# assert(dropout_or_batchnormalization_layer is None) ## Do NOT use dropout or BatchNormalization, they don't work with train_on_batch.
# assert(basic_regularizer is None) ## Do not use regularizer, slightly better improvements.

dataset_kind = "colorado_traffic" ## covid, challenge, alz, toy, chestxray-seg-varshape, chestxray-nonseg-fixshape, traffic, colorado_traffic
dataset_path = f"./datasets/{dataset_kind}.obj"

base_dir_dataset = "user/projects/data"
base_dir_project = "user/projects/python/covid/experiments"
if True: ## Whether you want to create and save dataset.
    ## --- --- Create and Save dataset, comment below if you want to wait dataset creation.
    ## Best hyper-parameters found

    if dataset_kind == "covid":
        ## --- COVID DATASET.
        dataset_obj = data_prep.Dataset(path_to_dataset= "./inputs/time_series_375_prerpocess_en.xlsx", obj_path= dataset_path, dataset_kind = dataset_kind, init_number_for_unobserved_entry = -1.0, static_data_input_mapping = dict(RNN_vectors = [], static_encoder = [], raw = ["age", "gender"]), kwargs_dataset_specific = dict(excluded_features= [])) ## separate_dynamic_static_features = "combine", "separate raw", "separate enrich", raw = ["age", "gender", 'Admission time', 'Discharge time'], excluded_features= ["High sensitivity C-reactive protein", "Lactate dehydrogenase", "(%)lymphocyte"]
    
    elif dataset_kind.split("-")[0] == "chestxray":
        if dataset_kind.split("-")[1] == "seg": apply_segment = True
        elif dataset_kind.split("-")[1] == "nonseg": apply_segment = False
        else: raise Exception("Not accepted dataset kind")

        if dataset_kind.split("-")[2] == "varshape": 
            image_shape = None
            if_separate_img = True
        elif dataset_kind.split("-")[2] == "fixshape": 
            image_shape = (400, 300)
            if_separate_img = False
        else: raise Exception("Not accepted dataset kind")
        ## --- COVID DATASET.
        dataset_obj = data_prep.Dataset(path_to_dataset= f"{base_dir_dataset}/covid-chestxray-dataset", obj_path= dataset_path, dataset_kind = dataset_kind.split("-")[0], init_number_for_unobserved_entry = 0.0, static_data_input_mapping = dict(RNN_vectors = [], static_encoder = [], raw = ["sex", "age", "RT_PCR_positive"]), kwargs_dataset_specific = dict(excluded_features= ["intubation_present", "in_icu", "date", "location", "folder", "doi", "url", "license", "clinical_notes", "other_notes", "Unnamed: 29"], image_shape= image_shape, apply_segment= apply_segment), if_separate_img= if_separate_img) # excluded_features= ["Unnamed: 29"]
        # if apply_segment: dataset_path = dataset_path[:-4] + "_segment.obj"
        # else: dataset_path = dataset_path[:-4] + "_non-segment.obj"

    elif dataset_kind == "challenge":
        ## --- CHALLENGE DATASET
        dataset_obj = data_prep.Dataset(path_to_dataset= "./inputs/physionet-challenge", obj_path= dataset_path, dataset_kind = dataset_kind, init_number_for_unobserved_entry = 0.0, static_data_input_mapping = dict(RNN_vectors = [], static_encoder = [], raw = ["Age", "Gender", 'Height', "CCU", "CSRU", "SICU"]), kwargs_dataset_specific = dict())

    elif dataset_kind == "alz":
        ## --- ALZ DATASET.
        dataset_obj = data_prep.Dataset(path_to_dataset= "./inputs/alz/data/", obj_path= dataset_path, dataset_kind = dataset_kind, init_number_for_unobserved_entry = -10.0, static_data_input_mapping= dict(RNN_vectors= [], static_encoder= ["SNP"], raw= ["BL_Age", "Gender"]), kwargs_dataset_specific = dict(TIMES = ["BL", "M6", "M12", "M18", "M24"], target_label = "M36_DX", dynamic_modalities_to_use= ["FS", "VBM", "RAVLT", "ADAS", "FLU", "MMSE", "TRAILS"]))
    
    elif dataset_kind.split("-")[0] == "traffic":
        if dataset_kind.split("-")[1] == "varshape": 
            image_shape = None
            if_separate_img = True
        elif dataset_kind.split("-")[1] == "fixshape": 
            image_shape = (400, 300)
            if_separate_img = False
        dataset_obj = data_prep.Dataset(path_to_dataset= f"{base_dir_dataset}/traffic-project/austin", obj_path= dataset_path, dataset_kind = "traffic", init_number_for_unobserved_entry = 0.0, kwargs_dataset_specific = dict(excluded_features= [], target_columns= ["NORTH_THRU", "NORTH_RIGHT", "NORTH_LEFT", "SOUTH_THRU", "SOUTH_RIGHT", "SOUTH_LEFT", "EAST_THRU", "EAST_RIGHT", "EAST_LEFT", "WEST_THRU", "WEST_RIGHT", "WEST_LEFT", "UNASSIGNED_THRU", "UNASSIGNED_RIGHT", "UNASSIGNED_LEFT"], image_shape= image_shape, apply_segment= False, prediction_interval= 60, num_bags_max= 140, num_records= 5), if_separate_img= if_separate_img)

    elif dataset_kind == "colorado_traffic":
        max_time_gap_hrs = 4  # Maximum time gap for observations within the same bag.
        future_congestion_lookahead_hrs = 2  # How far ahead to look when computing future congestion.
        compute_near_crash = True  # Flag indicating if near_crash should be computed (boolean).
        crash_spatial_thresh = 3  # Spatial threshold for determining if a sample is near a crash (miles).
        crash_temporal_thresh = 2  # temporal threshold for determining if a sample is near a crash (hours).
        
        dataset_obj = data_prep.Dataset(
            path_to_dataset= f"{base_dir_dataset}/traffic-project/CDOT_Full_Combined",
            obj_path=dataset_path,
            dataset_kind=dataset_kind,
            init_number_for_unobserved_entry=0.0,
            kwargs_dataset_specific=dict(max_time_gap_hrs=max_time_gap_hrs,
                                         future_congestion_lookahead_hrs=future_congestion_lookahead_hrs,
                                         crash_temporal_thresh=crash_temporal_thresh,
                                         crash_spatial_thresh=crash_spatial_thresh,
                                         compute_near_crash=compute_near_crash,
                                         num_bags_max=1000,
                                         target_columns=['future_congestion', 'near_crash'],
                                         excluded_features=['location']))

    ## --- COMMON for all DATASET.
    dataset_obj.set_dicts(kwargs_toy_dataset= None) ## For toy dataset: kwargs_toy_dataset = {"num_patients": 385, "max_time_steps": 80, "num_features_static_dict": dict(raw = 1, RNN = 1, static_encoder = 1), "num_features_dynamic": 2, "time_interval_range": [0., 0.05], "static_data_range": [0., 1.], "sin function info ranges dict": {"amplitude": [0., 1.], "displacement_along_x_axis": [0., 1.], "frequency": [0.5, 1.], "displacement_along_y_axis": [1., 2.]}, "observation noise range": [0., 0.], "missing probability": 0.0, "class_0_proportion": [0.5]}

    ## Save dataset.
    with open(dataset_path, "wb") as dataset_file:
        pickle.dump(dataset_obj, dataset_file)

## Load dataset.
dataset_obj = pickle.load(open(dataset_path, "rb"))
# dataset_obj.export_to_csv(dir_path_to_export = "./datasets/datasets_in_csv/")

## split the dataset.
# dataset_obj.reduce_number_of_participants(reduced_number_of_participants = 30) ## Reduce the number of participants for saving time. This can be used for fast debugging or grid search, not for the actual experiment. Comment out this line if you don't want to reduce.
dataset_obj.split_train_test(train_proportion= 0.8, k_fold = 5, whether_training_is_large = True, shuffle= True, balance_label_distribution = False) ## When train_proportion is not None, then k_fold is ignored.

## Getting statistics
# sum([i * (dataset_obj.feature_min_max_dict["BL_Age"]["max"] - dataset_obj.feature_min_max_dict["BL_Age"]["min"]) + dataset_obj.feature_min_max_dict["BL_Age"]["min"] for i in dataset_obj.get_data_collections_from_all_samples(data_access_function = lambda x: x["raw"]["data"][0][0])]) / 379
# for group, feature in zip(["static"], ["Age"]):
#     observed_numbers = dataset_obj.get_measures_of_feature(feature_name = feature, group_name = group)
#     print(f"{feature}, mean: {observed_numbers.mean()}, std: {observed_numbers.std()}")
# for group, feature in zip(["static"], ["Gender"]):
#     observed_numbers = dataset_obj.get_measures_of_feature(feature_name = feature, group_name = group)
#     unique, counts = np.unique(observed_numbers, return_counts=True)
#     print(f"{feature}, item: occurence = {dict(zip(unique, counts))}")
# unique, counts = np.unique(np.array(dataset_obj.dataframe_static["In-hospital_death"]), return_counts=True)
# print(f"In-hospital_death, item: occurence = {dict(zip(unique, counts))}")

## Getting observation density and average length of record.
num_entries = 0
num_observed = 0
total_length = 0
for patient_idx in range(dataset_obj.num_patients):
    if tensorflow.is_tensor(dataset_obj.dicts[patient_idx]["RNN_vectors"]["mask"]):
        np_arr = dataset_obj.dicts[patient_idx]["RNN_vectors"]["mask"].numpy()
    else:
        np_arr = dataset_obj.dicts[patient_idx]["RNN_vectors"]["mask"]
    total_length += np_arr.shape[1]
    num_entries += np_arr.size
    num_observed += np.count_nonzero(np_arr)
print(f"num_entries: {num_entries}, num_observed: {num_observed}, proportion = {num_observed / num_entries}, average length of record = {total_length / dataset_obj.num_patients}")

labels_distribution = []
for i in range(len(dataset_obj.dicts)):
    if dataset_obj.classification_axis is not None:
        labels_distribution.append(dataset_obj.dicts[i]["predictor label"][0][dataset_obj.classification_axis])
    else:
        labels_distribution.append(dataset_obj.dicts[i]["predictor label"][0])
print(Counter(labels_distribution))

if dataset_kind.split("-")[0] == "chestxray":
    import plotly.graph_objects as go
    shapes_list_dict= {"Height": [elem[0] for elem in dataset_obj.img_shapes], "Width": [elem[1] for elem in dataset_obj.img_shapes], "Ratio": [elem[1] / elem[0] for elem in dataset_obj.img_shapes]}
    font_dict = dict(title_font = {"size": 70}, tickfont= dict(size= 50))
    for key in ["Height", "Width", "Ratio"]:
        shapes_list = shapes_list_dict[key]
        print(print(f"axis: {key}, mean: {mean(shapes_list)}, std: {stdev(shapes_list)}, max: {max(shapes_list)}, min: {min(shapes_list)}"))
        if False:
            fig = go.Figure(data=[go.Histogram(x= shapes_list, name= key)])
            fig.update_xaxes(title_text='Counts', **font_dict)
            fig.update_yaxes(title_text=key, **font_dict)
            # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            fig.show()
    if False:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=shapes_list_dict["Height"], name= "Height", marker_color= "red"))
        fig.add_trace(go.Histogram(x=shapes_list_dict["Width"], name= "Width", marker_color= "blue"))

        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.65)
        fig.update_layout(
            # plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                x=0.84,
                y=0.97,
                traceorder="normal",
                font=dict(
                    # family="sans-serif",
                    size=50,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2,
                itemsizing= "trace"
            )
        )

        fig.update_xaxes(title_text='Counts', **font_dict)
        fig.update_yaxes(title_text='Size', **font_dict)
        fig.show()

## --- --- Conduct the Experiment
experiment = utils.Experiment(base_name= dataset_kind + "-")

### ------------------------- COVID Dataset Learning -------------------------
## Best hyper-parameters found
# learning_rate= 0.00001, +++ 0.0003 +++
# factors_dict = {"reconstruction loss": 5 * 1e-3, "prediction": 1e-1} 90.91%
# Decoder First layer "activation": lambda x: activations.relu(x, alpha = 0.1).
# Recon, Predictor loss: 2, 2 > 3, 3 (97) > 2, 3. (88.61) > 2, 2. (86) > 3, 3 (86) > 2, binary cross entropy
# kwargs_RNN_vectors= {"units": 105}, kwargs_RNN_vectors= {"units": 80} > {"units": 100}
# factors_dict = {"reconstruction loss": 1 * 1e-1}

if dataset_kind == "covid":
    experiment_settings_list = [
        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 1, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},
        #     # iters = 100

        #     predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3))],
        #     # dict(model= "LSTM", kwargs= dict(units= 40, activity_regularizer= basic_regularizer, bias_regularizer= basic_regularizer, dropout= 0.3, recurrent_dropout= 0.3, activation= "tanh")) 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (192 / 166) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'Attention-LSTM-with-dropout'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 100, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},
        #     # iters = 100

        #     predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 60, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.3, "recurrent_dropout": 0.3, "activation": "tanh"}), 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (192 / 166) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'SAE-FCL-with-dropout'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 100, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},
        #     # iters = 100

        #     predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 60, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (192 / 166) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'SAE-FCL-without-dropout'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 70, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "SquaredHinge"},

        #     predictor_structure_list= [["RandomFourierFeatures", dict(output_dim=400, scale=10.0, kernel_initializer="gaussian")], ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        # list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (192 / 166) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.)
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'SAE-SVM'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True, whether_use_static = False), 
        # fit = dict(
        #     iters = 20, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        # list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (192 / 166) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'SA-woS'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 10, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = None,
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'SA-woC'
        # ), key_of_name= "name"),

        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        fit = dict(
            iters = 5, factors_dict = {"dynamic vectors reconstruction loss": 0., "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

            predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

            static_encoder_decoder_structure_dict = None, loss_factor_funct = None
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "train", name = 'BLSTM'
        ), key_of_name= "name"),

        GridSearch(dict(model_class=estimator.MLP,
        init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "DNN")),

        GridSearch(dict(model_class=estimator.RandomForest,
        init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "RF")),

        GridSearch(dict(model_class=estimator.RidgeClassifier,
        init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "RC")),

        GridSearch(dict(model_class=estimator.SVM,
        init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "SVM")),

    ]

### ------------------------- Physionet-Challenge Dataset Learning -------------------------
# iters = 200 < 150 (150 is better than 200), 100 < 50.
# {"dynamic vectors reconstruction loss": Grid(1.0, +++ 2.0 +++), "static reconstruction loss": 2.0, "prediction": Grid("binary cross entropy", +++ 2.0 +++)}
# Grid(30, 60, +++ 90 +++)
# "prediction": 1e+2 ~ 1e+1
# iters = +++ 150 +++, 50
# optimizer = Grid(optimizers.Adam(+++ learning_rate= 0.0003 +++, 1e-5))
# init_number_for_unobserved_entry = +++ 0.0 +++,
# optimizer = Grid(optimizers.Adam(learning_rate= 0.0003), +++ optimizers.Adamax(learning_rate= 0.0003) +++, optimizers.Nadam(learning_rate= 0.0003), optimizers.Adagrad(learning_rate= 0.0003))

# GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
#         fit = dict(
#             iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

#             predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(21 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

#             static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
#         ),
#         predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SAE'
#         ), key_of_name= "name")

# GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
#         fit = dict(
#             iters = 40, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "SquaredHinge"},

#             predictor_structure_list= [["RandomFourierFeatures", dict(output_dim=400, scale=10.0, kernel_initializer="gaussian")], ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "linear", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             dynamic_vectors_decoder_structure_list= Grid(
#             [["Dense", {"units": 35, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],
#             ),

#             list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

#             static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
#         ),
#         predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SAE'
#         ), key_of_name= "name")

# dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(21 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]]

# [["LSTM", dict(units= len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), activity_regularizer= basic_regularizer, bias_regularizer= basic_regularizer, dropout= 0.0, recurrent_dropout= 0.0)]]

if dataset_kind == "challenge":
    experiment_settings_list = [
        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        fit = dict(
            iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-9, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "binary cross entropy"},

            predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list= [["LSTM", dict(units= 30, activity_regularizer= basic_regularizer, bias_regularizer= basic_regularizer, dropout= 0.3, recurrent_dropout= 0.3)]],

            list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 2, key_dim= 15)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3)), 
            dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.3, "recurrent_dropout": 0.3})], 

            static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.0, "static": 0.0}), fit_on = "both", name = 'TA-with-dropout'
        ), key_of_name= "name"),

        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        fit = dict(
            iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

            predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": None, "bias_regularizer": None}]],

            dynamic_vectors_decoder_structure_list= Grid(
            [["LSTM", dict(units= 30, activity_regularizer= None, bias_regularizer= None, dropout= 0.0, recurrent_dropout= 0.0)]],
            ),

            list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": None, "bias_regularizer": None, "dropout": 0.0, "recurrent_dropout": 0.0}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": None, "bias_regularizer": None, "dropout": 0.0, "recurrent_dropout": 0.0})], 

            static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.0, "static": 0.0}), fit_on = "both", name = 'SA-FCL-without-dropout'
        ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= Grid(
        #     [["LSTM", dict(units= 30, activity_regularizer= basic_regularizer, bias_regularizer= basic_regularizer, dropout= 0.2, recurrent_dropout= 0.2)]],
        #     ),

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.1, "recurrent_dropout": 0.1}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.1, "recurrent_dropout": 0.1})], 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.0, "static": 0.0}), fit_on = "both", name = 'SA-FCL-with-dropout'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}], None, ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}], None, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": None, "bias_regularizer": None}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": 35, "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}], None, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}]],

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 2, key_dim= 15)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= None))], 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.0, "static": 0.0}), fit_on = "both", name = 'TA-without-dropout'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 45, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= Grid(
        #     [["LSTM", dict(units= 30, activity_regularizer= basic_regularizer, bias_regularizer= basic_regularizer, dropout= 0.0, recurrent_dropout= 0.0)]],
        #     ),

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0})], 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SA-FCL'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 45, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "SquaredHinge"},

        #     predictor_structure_list= [["RandomFourierFeatures", dict(output_dim=100, scale=10.0, kernel_initializer="gaussian")], ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "linear", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= Grid(
        #     [["LSTM", dict(units= 30, activity_regularizer= basic_regularizer, bias_regularizer= basic_regularizer, dropout= 0.0, recurrent_dropout= 0.0)]],
        #     ),

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0})], 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SA-SVM'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True, whether_use_static = False), 
        # fit = dict(
        #     iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= Grid(
        #     [["Dense", {"units": 35, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],
        #     ),

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0})], 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SA-woS'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= Grid(
        #     [["Dense", {"units": 35, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],
        #     ),

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0})], 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = None,
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SA-woC'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 0.0, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(21 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= dict(model= "GRU", kwargs= {"units": 30, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}), 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SA-SVM'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True, whether_use_static = False), 
        # fit = dict(
        #     iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= Grid(
        #     [["Dense", {"units": 35, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],
        #     ),

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0})], 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (3446 / 554) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SA-woS'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 30, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-7, "static reconstruction loss": 0., "prediction": 1e+1}, optimizer = optimizers.Adamax(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

        #     predictor_structure_list= [["Dense", {"units": int(9 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(3 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= Grid(
        #     [["Dense", {"units": 35, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],
        #     ),

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 20, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0})], 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = None,
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.5, "static": 1.0}), fit_on = "both", name = 'SA-woC'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class=estimator.RidgeClassifier,
        # init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "RC")),

        # GridSearch(dict(model_class=estimator.RandomForest,
        # init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "RF")),

        # GridSearch(dict(model_class=estimator.SVM,
        # init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "SVM")),

        # GridSearch(dict(model_class=estimator.MLP,
        # init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "DNN")),
    ]

### ------------------------- chest-xray dataset -------------------------
# iters = 200 < 150 (150 is better than 200), 100 < 50.
# {"dynamic vectors reconstruction loss": Grid(1.0, +++ 2.0 +++), "static reconstruction loss": 2.0, "prediction": Grid("binary cross entropy", +++ 2.0 +++)}
# Grid(30, 60, +++ 90 +++)
# Grid(optimizers.Adam(learning_rate= 1e-4) @0.61, optimizers.Adam(learning_rate= 1e-6), optimizers.Adam(learning_rate= 1e-3) +++ @0.70, optimizers.Adam(learning_rate= 1e-2), optimizers.Adam(learning_rate= 1e-1))
# nonseg: 0.63 > seg: 0.61
# whether_use_mask = True 0.65 > whether_use_mask = False 0.63
if dataset_kind.split("-")[0] == "chestxray":
    experiment_settings_list = [
        # GridSearch(dict(model_class=estimator.CNN, init=dict(),
        #                 fit=dict(),
        #                 predict=dict(), fit_on="both", name='SAE', key_of_name="name")),
        
        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 2, whether_use_mask = False), 
        # fit = dict(
        #     iters = 15, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "dynamic images reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e+3}, optimizer = optimizers.Adam(learning_rate= 1e-6), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 1.0, "prediction": "binary cross entropy"},

        #     predictor_structure_list= [["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": None, "bias_regularizer": None}]],

        #     dynamic_vectors_decoder_structure_list= [["TimeDense", {"units": int(50 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["TimeDense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}]],

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= None, model_kwargs_prior_residual= None, droprate_before_residual= None))],

        #     static_encoder_decoder_structure_dict = None,

        #     conv_encoder_decoder_structure_dict = dict(
        #         encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= False)], ["TimeDistributed", dict(layer= Flatten())], ["LSTM", dict(units= 90, return_sequences=True, return_state=True)]],

        #         decoder = [["TimeDense", {"units": 100, "activation": lambda x: activations.relu(x, alpha = 0.1)}], ["TimeDense", {"units": 400, "activation": lambda x: activations.relu(x, alpha = 0.1)}], ["TimeDistributed", dict(layer = Reshape(target_shape= (20, 20, 1)))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5))]]
        #     ),
        #     loss_factor_funct = (lambda **kwargs: ((455 - 109) / 109.) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.)
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.1, "static": 0.1}), fit_on = "both", name = 'TemporalAttention-without-dropout'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 2, whether_use_mask = False), 
        # fit = dict(
        #     iters = 1, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "dynamic images reconstruction loss": 5 * 1e+3, "static reconstruction loss": 0., "prediction": 1e+3}, optimizer = optimizers.Adam(learning_rate= 1e-4), loss_kind_dict = {"dynamic vectors reconstruction loss": 1.0, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 1.0, "prediction": 1.0},

        #     predictor_structure_list= [["Dense", {"units": int(40 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(10 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["TimeDense", {"units": int(40 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["TimeDense", {"units": int(30 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["TimeDense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

        #     # list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3))],

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 30, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer, "dropout": dropout_rate, "recurrent_dropout": dropout_rate}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer, "dropout": dropout_rate, "recurrent_dropout": dropout_rate})],

        #     static_encoder_decoder_structure_dict = None,

        #     conv_encoder_decoder_structure_dict = dict(
        #         encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= False, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["TimeDistributed", dict(layer= Flatten())], ["LSTM", dict(units= 150, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate, return_sequences=True, return_state=True)]],

        #         decoder = [["TimeDense", {"units": 289, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["TimeDense", {"units": 324, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["TimeDistributed", dict(layer = Reshape(target_shape= (18, 18, 1)))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5), activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer)]]
        #     ),
        #     # loss_factor_funct = (lambda **kwargs: ((455 - 109) / 109.) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.)
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.1, "static": 0.1}), fit_on = "both", name = 'TemporalAttention-with-dropout'
        # ), key_of_name= "name"),
        
        #---------------------------------------------------------------------------------------------

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 2, whether_use_mask = False), 
        # fit = dict(
        #     iters = 40, run_eagerly= False, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "dynamic images reconstruction loss": 5 * 1e+3, "static reconstruction loss": 0., "prediction": 1e+3}, optimizer = optimizers.Adam(learning_rate= 1e-4), loss_kind_dict = {"dynamic vectors reconstruction loss": 1.0, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 1.0, "prediction": 1.0},

        #     predictor_structure_list= [["Dense", {"units": int(40 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(10 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["TimeDense", {"units": int(40 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["TimeDense", {"units": int(30 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["TimeDense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3))],

        #     # list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 30, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer, "dropout": dropout_rate, "recurrent_dropout": dropout_rate}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer, "dropout": dropout_rate, "recurrent_dropout": dropout_rate})],

        #     static_encoder_decoder_structure_dict = None,

        #     conv_encoder_decoder_structure_dict = dict(
        #         encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["TimeDistributed", dict(layer= Flatten())], ["HighOrderMixture", dict(order= 3)], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

        #         # encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= False, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["TimeDistributed", dict(layer= Flatten())], ["LSTM", dict(units= 150, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate, return_sequences=True, return_state=True)]],

        #         decoder_input = dict(
        #             input = "images_RNN_input_masked",
        #             keep_rate = 0.8,
        #             layers = [["Conv2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["Flatten", dict()], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]]
        #         ),

        #         decoder = [["Dense", {"units": 289, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": 324, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["TimeDistributed", dict(layer = Reshape(target_shape= (18, 18, 1)))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5), activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer)]]
        #     ),
        #     # loss_factor_funct = (lambda **kwargs: ((455 - 109) / 109.) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.)
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.1, "static": 0.1}), fit_on = "both", name = 'TemporalAttention-with-dropout-reconst'
        # ), key_of_name= "name"),

        ########################################################################################## chest x-ray fixed image shape

        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 2, whether_use_mask = False), 
        fit = dict(
            iters = 5, run_eagerly= False, factors_dict = {"dynamic vectors reconstruction loss": 1, "dynamic images reconstruction loss": 1, "static reconstruction loss": 0., "prediction": 1e+3}, optimizer = optimizers.legacy.Adam(learning_rate= 1e-6), loss_kind_dict = {"dynamic vectors reconstruction loss": 1.0, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 1.0, "prediction": "binary cross entropy"},

            predictor_structure_list= [["Dense", {"units": int(40 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(10 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

            dynamic_vectors_decoder_structure_list= [["TimeDense", {"units": int(40 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["TimeDense", {"units": int(30 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["TimeDense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

            list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3))],

            # list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 30, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer, "dropout": dropout_rate, "recurrent_dropout": dropout_rate}), dict(model= "GRU", kwargs= {"units": 15, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer, "dropout": dropout_rate, "recurrent_dropout": dropout_rate})],

            static_encoder_decoder_structure_dict = None,

            conv_encoder_decoder_structure_dict = dict(
                encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["TimeDistributed", dict(layer= Flatten())], 
                ["TransformerTemporal", dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3)],
                ["HighOrderMixture", dict(order= 3)], 
                # ["LSTM", dict(units= 150, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)],
                ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

                # encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= False, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["TimeDistributed", dict(layer= Flatten())], ["LSTM", dict(units= 150, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate, return_sequences=True, return_state=True)]],

                decoder_input = dict(
                    input = "images_RNN_input_masked",
                    keep_rate = 0.8,
                    layers = [["Conv2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["Flatten", dict()], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]]
                ),

                decoder = [["Dense", {"units": 289, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": 324, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["TimeDistributed", dict(layer = Reshape(target_shape= (18, 18, 1)))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5), activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer)]]
            ),
            loss_factor_funct = (lambda **kwargs: ((455 - 109) / 109.) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.)
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.1, "static": 0.1}), fit_on = "both", name = 'TemporalAttention-with-dropout-predict'
        ), key_of_name= "name"),

        ########################################################################################## chest x-ray varying image shape

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 0, verbose = 2, whether_use_mask = False), 
        # fit = dict(
        #     iters = 2, run_eagerly= True, factors_dict = {"dynamic vectors reconstruction loss": 1, "dynamic images reconstruction loss": 1, "static reconstruction loss": 0., "prediction": 1}, optimizer = optimizers.legacy.Adam(learning_rate= 1e-8), loss_kind_dict = {"dynamic vectors reconstruction loss": 1.0, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 1.0, "prediction": "binary cross entropy"},

        #     predictor_structure_list= [["Dense", {"units": int(40 * num_neurons_increase_factor), "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(10 * num_neurons_increase_factor), "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= None,

        #     # list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3))],

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "GRU", kwargs= {"units": 30, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer, "dropout": dropout_rate, "recurrent_dropout": dropout_rate}), dict(model= "GRU", kwargs= {"units": 15, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer, "dropout": dropout_rate, "recurrent_dropout": dropout_rate})],

        #     static_encoder_decoder_structure_dict = None,

        #     conv_encoder_decoder_structure_dict = dict(
        #         # encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["TimeDistributed", dict(layer= Flatten())], 
        #         # ["TransformerTemporal", dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3)],
        #         # ["HighOrderMixture", dict(order= 3)], 
        #         # # ["LSTM", dict(units= 150, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)],
        #         # ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

        #         # encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["AttentionReshaper", dict(target_shape= (12, 12))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= False, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["TimeDistributed", dict(layer= Flatten())], ["LSTM", dict(units= 150, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate, return_sequences=True, return_state=True)]],

        #         encoder = [["VarShapeIntoFixConv", dict(list_CNN= [
        #             ["Conv2D", dict(filters= 16, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, input_shape= (None, None, 1))], ["MaxPooling2D", dict(pool_size=(3, 3))], 
        #             ["Conv2D", dict(filters= 8, kernel_size= (3, 3), strides= (2, 2), padding= "valid", activation= basic_activation)], ["MaxPooling2D", dict(pool_size=(2, 2))], 
        #             ["Conv2D", dict(filters= 4, kernel_size= (2, 2), strides= (1, 1), padding= "valid", activation= basic_activation)], ["MaxPooling2D", dict(pool_size=(2, 2))], 
        #             ["AttentionReshaper", dict(if_temporal= False, target_shape= (20, 15))], 
        #             # ["Conv2D", dict(filters= 2, kernel_size= (2, 2), strides= (1, 1), padding= "valid", activation= basic_activation)],
        #             ["MaxPooling2D", dict(pool_size=(2, 2))],
        #             ])], 
        #             ["TimeDistributed", dict(layer= Flatten())], ["LSTM", dict(units= 80, kernel_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate, return_sequences=True, return_state=True)]],

        #         decoder_input = None,

        #         decoder = None
        #     ),
        #     # loss_factor_funct = (lambda **kwargs: ((455 - 109) / 109.) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        #     loss_factor_funct = None,
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.01, "static": 0.01}), fit_on = "train", name = 'Attention-reshaper'
        # ), key_of_name= "name"),

        ## Reference Models
        # GridSearch(dict(model_class=estimator.CNN, init=dict(),
        #                 fit=dict(),
        #                 predict=dict(), fit_on="both", name='SAE', key_of_name="name")),
    ]

## ------------------------- Traffic Dataset -------------------------
if dataset_kind.split("-")[0] == "traffic":
    experiment_settings_list = [
        ########################################################################################## chest x-ray varying image shape

        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 2, whether_use_mask = True), 
        fit = dict(
            iters = 15, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "dynamic images reconstruction loss": 5 * 1e-3, "static reconstruction loss": 5 * 1e-3, "prediction": 1e+3, "static image reconstruction loss": 5 * 1e-3}, optimizer = optimizers.legacy.Adam(learning_rate= 1e-6), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "dynamic images reconstruction loss": 1.0, "static reconstruction loss": 1.0, "prediction": "binary cross entropy"},

            predictor_structure_list= [["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": None, "bias_regularizer": None}]],

            dynamic_vectors_decoder_structure_list= [["TimeDense", {"units": int(50 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["TimeDense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": None, "bias_regularizer": None}]],

            list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= None, model_kwargs_prior_residual= None, droprate_before_residual= None))],

            static_encoder_decoder_structure_dict = None,
            # dict(encoder= [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 80, "activation": "tanh", "activity_regularizer":  basic_regularizer}],], 
            # decoder = [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),

            static_conv_encoder_decoder_structure_dict= dict(
                encoder= [["Conv2D", dict(filters= 16, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, input_shape= (None, None, 1))], ["MaxPooling2D", dict(pool_size=(3, 3))], 
                    ["Conv2D", dict(filters= 8, kernel_size= (3, 3), strides= (2, 2), padding= "valid", activation= basic_activation)], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Flatten", dict()], ["Dense", {"units": int(40 * num_neurons_increase_factor), "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}],
                    ],
                decoder= [["Dense", {"units": 90, "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["Dense", {"units": 49, "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["Reshape", dict(target_shape= (7, 7, 1))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5))]],
            ),

            conv_encoder_decoder_structure_dict = dict(
                encoder = [["ConvLSTM2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["MaxPooling3D", dict(pool_size=(1, 2, 2))], ["ConvLSTM2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh", return_sequences= True, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)], ["TimeDistributed", dict(layer= Flatten())], 
                ["TransformerTemporal", dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3)],
                ["HighOrderMixture", dict(order= 3)], 
                # ["LSTM", dict(units= 150, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer, dropout= dropout_rate, recurrent_dropout= dropout_rate)],
                ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]],

                decoder_input = dict(
                    input = "images_RNN_input_masked",
                    keep_rate = 0.8,
                    layers = [["Conv2D", dict(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Conv2D", dict(filters= 64, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= "tanh")], ["Flatten", dict()], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}]]
                ),

                decoder = [["Dense", {"units": 289, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": 324, "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["TimeDistributed", dict(layer = Reshape(target_shape= (18, 18, 1)))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer))], ["TimeDistributed", dict(layer= Conv2DTranspose(filters= 32, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5), activity_regularizer= basic_regularizer, bias_regularizer= basic_strong_regularizer)]]
            ),
            loss_factor_funct = None
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.1, "static": 0.1}), fit_on = "both", name = 'TemporalAttention-without-dropout'
        ), key_of_name= "name"),

    ]
        ## Reference Models
        # GridSearch(dict(model_class=estimator.CNN, init=dict(),
        #                 fit=dict(),
        #                 predict=dict(), fit_on="both", name='SAE', key_of_name="name")),

## ------------------------- Alzheimer's Disease Dataset Learning -------------------------
## Good hyperparameters
## loss_kind_dict["prediction"] : binary > 2.0
## optimizer = optimizers.Adam(learning_rate= 0.00001): learning_rate= 0.001 > learning_rate= 0.00001
## The larger iterations is better for SAE.
# GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
#         fit = dict(
#             iters = 150, factors_dict = {"dynamic vectors reconstruction loss": 0.5 * 1e-1, "static reconstruction loss": 1 * 1e-1, "prediction": 1e+2}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "binary cross entropy"},

#             predictor_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(33 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             dynamic_vectors_decoder_structure_list= 
#             [["Dense", {"units": int(75 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer":  basic_regularizer, "bias_regularizer": basic_regularizer}]],

#             model_name_RNN_vectors= "LSTM",
#             kwargs_RNN_vectors= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"},

#             static_encoder_decoder_structure_dict = 
#             dict(encoder= [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 80, "activation": "tanh", "activity_regularizer":  basic_regularizer}]], 
#             decoder = [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),
#         ),
#         predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.1, "static": 0.05}), fit_on = "both", name = 'SAE'
#         ), key_of_name= "name")
if dataset_kind == "alz":
    experiment_settings_list = [

        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        fit = dict(
            iters = 1400, factors_dict = {"dynamic vectors reconstruction loss": 1, "static reconstruction loss": 1e+1, "prediction": 1e+2}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "binary cross entropy"},

            predictor_structure_list=
            [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(33 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list= 
            [["Dense", {"units": int(75 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer":  basic_regularizer, "bias_regularizer": basic_regularizer}]]
            ,

            list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

            static_encoder_decoder_structure_dict = 
            dict(encoder= [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 80, "activation": "tanh", "activity_regularizer":  basic_regularizer}],], 
            decoder = [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.2, "static": 0.05}), fit_on = "both", name = 'SAE'
        ), key_of_name= "name"),


        dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        fit = dict(
            iters = 1400, factors_dict = {"dynamic vectors reconstruction loss": 0, "static reconstruction loss": 0, "prediction": 1e+2}, optimizer = optimizers.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": "binary cross entropy"},

            predictor_structure_list=
            [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(33 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list= 
            [["Dense", {"units": int(75 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer":  basic_regularizer, "bias_regularizer": basic_regularizer}]]
            ,

            list_of_model_kwargs_RNN_vectors= dict(model= LSTM, kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

            static_encoder_decoder_structure_dict = 
            dict(encoder= [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 80, "activation": "tanh", "activity_regularizer":  basic_regularizer}],], 
            decoder = [["Dense", {"units": 300, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.2, "static": 0.05}), fit_on = "train", name = 'BLSTM'
        ),

        GridSearch(dict(model_class=estimator.RandomForest,
        init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "RF")),

        GridSearch(dict(model_class=estimator.MLP,
        init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "DNN")),
    ]

### ------------------------- Toy Dataset Learning -------------------------
if dataset_kind == "toy":
    experiment_settings_list = [
        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = Grid(True, False)), 
        fit = dict(
            iters = 10, factors_dict = {"dynamic vectors reconstruction loss": 0.5 * 1e-1, "static reconstruction loss": 1 * 1e-1, "prediction": 1e-1}, optimizer = optimizers.Adam(learning_rate= 0.001), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

            predictor_structure_list= [["Dense", {"units": int(12 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(6 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(2 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.prediction_labels_bag), "activation": "softmax", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(15 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(10 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(7 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer":  basic_regularizer, "bias_regularizer": basic_regularizer}]],

            list_of_model_kwargs_RNN_vectors= dict(model= LSTM, kwargs= {"units": 12, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

            static_encoder_decoder_structure_dict = dict(encoder= [["Dense", {"units": 60, "activation": "tanh", "activity_regularizer": basic_regularizer}], ["Dense", {"units": 30, "activation": "tanh", "activity_regularizer":  basic_regularizer}]], 
            decoder = [["Dense", {"units": 60, "activation": "tanh", "activity_regularizer":  basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["static_encoder"]), "activation": "tanh", "activity_regularizer":  basic_regularizer}]]),
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 0.1, "static": 0.05}), fit_on = "both", name = 'SAE'
        ), key_of_name= "name"),
    ]

if dataset_kind == "colorado_traffic":
    experiment_settings_list = [
        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        fit = dict(
            iters = 5, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.legacy.Adam(learning_rate= 0.003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},
            # iters = 100
            
            static_conv_encoder_decoder_structure_dict= dict(
                encoder= [["Conv2D", dict(filters= 16, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, input_shape= (None, None, 1))], ["MaxPooling2D", dict(pool_size=(3, 3))], 
                    ["Conv2D", dict(filters= 8, kernel_size= (3, 3), strides= (2, 2), padding= "valid", activation= basic_activation)], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Flatten", dict()], ["Dense", {"units": int(40 * num_neurons_increase_factor), "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}],
                    ],
                decoder= [["Dense", {"units": 90, "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["Dense", {"units": 49, "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["Reshape", dict(target_shape= (7, 7, 1))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5))]],
            ),

            predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.target_columns), "activation": "sigmoid", "kernel_regularizer": schatten_p_norm_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3))],
            # dict(model= "LSTM", kwargs= dict(units= 40, activity_regularizer= basic_regularizer, bias_regularizer= basic_regularizer, dropout= 0.3, recurrent_dropout= 0.3, activation= "tanh")) 

            static_encoder_decoder_structure_dict = None, loss_factor_funct = None,
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'AELN'
        ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 5, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.legacy.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},
        #     # iters = 100
            
        #     static_conv_encoder_decoder_structure_dict= dict(
        #         encoder= [["Conv2D", dict(filters= 16, kernel_size= (5, 5), strides= (3, 3), padding= "valid", activation= basic_activation, input_shape= (None, None, 1))], ["MaxPooling2D", dict(pool_size=(3, 3))], 
        #             ["Conv2D", dict(filters= 8, kernel_size= (3, 3), strides= (2, 2), padding= "valid", activation= basic_activation)], ["MaxPooling2D", dict(pool_size=(2, 2))], ["Flatten", dict()], ["Dense", {"units": int(40 * num_neurons_increase_factor), "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}],
        #             ],
        #         decoder= [["Dense", {"units": 90, "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["Dense", {"units": 49, "activation": basic_activation, "kernel_regularizer": basic_regularizer, "bias_regularizer": basic_strong_regularizer}], ["Reshape", dict(target_shape= (7, 7, 1))], ["Conv2DTranspose-fit-to-image-shape", dict(kernel_size= (5, 5))]],
        #     ),

        #     predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.target_columns), "activation": "sigmoid", "kernel_regularizer": schatten_p_norm_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= [dict(model= "TransformerTemporal", kwargs= dict(model_kwargs_attention= ["MultiHeadAttention", dict(num_heads= 4, key_dim= 40)], prior_post_layer_norm_kwargs= dict(prior= dict(epsilon= 1e-6), post= dict(epsilon= 1e-6)), model_kwargs_prior_residual= None, droprate_before_residual= 0.3))],
        #     # dict(model= "LSTM", kwargs= dict(units= 40, activity_regularizer= basic_regularizer, bias_regularizer= basic_regularizer, dropout= 0.3, recurrent_dropout= 0.3, activation= "tanh")) 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = None,
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'Attention-LSTM-with-dropout'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 5, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.legacy.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},
        #     # iters = 100

        #     predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.target_columns), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 60, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.3, "recurrent_dropout": 0.3, "activation": "tanh"}), 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = None,
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'SAE-FCL-with-dropout'
        # ), key_of_name= "name"),

        # GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        # fit = dict(
        #     iters = 100, factors_dict = {"dynamic vectors reconstruction loss": 5 * 1e-3, "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.legacy.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},
        #     # iters = 100

        #     predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.target_columns), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

        #     list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 60, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

        #     static_encoder_decoder_structure_dict = None, loss_factor_funct = (lambda **kwargs: (192 / 166) if np.array_equal(kwargs["predictor_label"], np.array([[0., 1.]])) else 1.),
        # ),
        # predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "both", name = 'SAE-FCL-without-dropout'
        # ), key_of_name= "name"),
        
        #####################

        GridSearch(dict(model_class=estimator.MLP,
        init=dict(if_classification= False, hidden_layer_sizes=(400, 305, 200, 100, 25)), fit= dict(), predict= dict(), fit_on= "train", name= "AELN-woS")),

        GridSearch(dict(model_class=estimator.MLP,
        init=dict(if_classification= False, hidden_layer_sizes=(300, 205, 100, 100, 25)), fit= dict(), predict= dict(), fit_on= "train", name= "AELN-woC")),

        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        fit = dict(
            iters = 3, factors_dict = {"dynamic vectors reconstruction loss": 0., "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.legacy.Adam(learning_rate= 0.0003), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

            predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.target_columns), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

            static_encoder_decoder_structure_dict = None, loss_factor_funct = None
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "train", name = 'LSTM'
        ), key_of_name= "name"),

        GridSearch(dict(model_class = autoencoder.Autoencoder, init = dict(debug = 1, verbose = 1, whether_use_mask = True), 
        fit = dict(
            iters = 3, factors_dict = {"dynamic vectors reconstruction loss": 0., "static reconstruction loss": 0., "prediction": 1e-1}, optimizer = optimizers.legacy.Adam(learning_rate= 0.0002), loss_kind_dict = {"dynamic vectors reconstruction loss": 2.0, "static reconstruction loss": 2.0, "prediction": 2.0},

            predictor_structure_list= [["Dense", {"units": int(120 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(60 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(20 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], dropout_or_batchnormalization_layer, ["Dense", {"units": len(dataset_obj.target_columns), "activation": "sigmoid", "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            dynamic_vectors_decoder_structure_list= [["Dense", {"units": int(200 * num_neurons_increase_factor), "activation": lambda x: activations.relu(x, alpha = 0.1), "activity_regularizer": None, "bias_regularizer": None}], ["Dense", {"units": int(140 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": int(100 * num_neurons_increase_factor), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}], ["Dense", {"units": len(dataset_obj.input_to_features_map["dynamic_vectors_decoder"]), "activation": basic_activation, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer}]],

            list_of_model_kwargs_RNN_vectors= dict(model= "LSTM", kwargs= {"units": 64, "activity_regularizer": basic_regularizer, "bias_regularizer": basic_regularizer, "dropout": 0.0, "recurrent_dropout": 0.0, "activation": "tanh"}), 

            static_encoder_decoder_structure_dict = None, loss_factor_funct = None
        ),
        predict = dict(perturbation_info_for_feature_importance = {"center": 0., "proportion": 0.1}, swap_info_for_feature_importance = {"proportion": 0.7}, feature_importance_calculate_prob_dict = {"dynamic": 1.0, "static": 1.0}), fit_on = "train", name = 'Transformer'
        ), key_of_name= "name"),

        GridSearch(dict(model_class=estimator.MLP,
        init=dict(if_classification= False, hidden_layer_sizes=(150, 125, 100, 50, 25)), fit= dict(), predict= dict(), fit_on= "train", name= "DNN")),

        # GridSearch(dict(model_class=estimator.MLP,
        # init=dict(if_classification= False, hidden_layer_sizes=(700, 200, 100, 50, 25)), fit= dict(), predict= dict(), fit_on= "train", name= "DNN")),

        GridSearch(dict(model_class=estimator.MLP,
        init=dict(if_classification= False, hidden_layer_sizes=(350, 325, 300, 250, 125)), fit= dict(), predict= dict(), fit_on= "train", name= "TCN")),
        
        GridSearch(dict(model_class=estimator.MLP,
        init=dict(if_classification= False, hidden_layer_sizes=(800, 400, 200, 250, 50)), fit= dict(), predict= dict(), fit_on= "train", name= "TFT")),

        #####################

        # GridSearch(dict(model_class=estimator.RandomForest,
        # init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "RF")),

        # GridSearch(dict(model_class=estimator.RidgeClassifier,
        # init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "RC")),

        # GridSearch(dict(model_class=estimator.SVM,
        # init=dict(), fit= dict(), predict= dict(), fit_on= "train", name= "SVM")),

    ]

## ------------------------- Do Experiment and Plot the Results. -------------------------

## Conduct Experiments
if False:
    experiment.set_experimental_result(dataset = dataset_obj, experiment_settings_list = experiment_settings_list, save_result = True, note = str(seeds), dataset_path = dataset_path)
else:
    experiment = pickle.load(open(f"{base_dir_project}/outputs/colorado_traffic-57/experiment.obj", "rb"))

## Plot experimental results.
utils.plot_experimental_results(experiment = experiment, dataset= dataset_obj, num_loss_plotting_points = 200, num_top_features = 15, verbose= 2)

print("FINISHED.")