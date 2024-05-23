import pickle
import utilsforminds
import os
from shutil import rmtree
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from utilsforminds.math import mean, std
from utilsforminds.containers import GridSearch
from utilsforminds.containers import merge_dictionaries as merge
from datetime import datetime
import heapq
from operator import itemgetter

import plotly.express as px
from io import BytesIO
from PIL import Image

from random import sample, random

from outer_sources.clean_labels import VBMDataCleaner
from outer_sources.roi_map import VBMRegionOfInterestMap
from outer_sources.roi_map import FSRegionOfInterestMap

class Experiment():
    """Experiment object to predict something from dataset object and estimator object."""

    def __init__(self, base_name):
        """
        
        Attributes
        ----------
        base_name : str
            Directory path to save the experimental result.
        """

        ## Set the parent directory to save the experimental results.
        self.base_name = base_name
        parent_dir_name = utilsforminds.helpers.getNewDirectoryName("outputs/", base_name)
        self.parent_path = "./outputs/" + parent_dir_name
        os.mkdir(Path(self.parent_path + "/"))

    def set_experimental_result(self, experiment_settings_list, save_result = True, note: str = None, dataset = None, dataset_path: str = None):
        """Conduct the experiment and save the experimental results.
        
        Attributes
        ----------
        experiment_settings_list : list of (dict or GirdSearch)
            Each dictionary contains each model represented by it's keyword arguments or GridSearch object, for example,
            dict(model_class = autoencoder.Autoencoder,
            init = dict(debug = 1, verbose = 1, use_mask_in_LSTM_input = True),
            fit = dict(...) (optional),
            predict = dict(...),
            fit_on = 'train',
            name = 'semi-supervised AE' (optional))

        use_mask_in_LSTM_input : bool
            Whether to concatenate mask (observabilities on the features) to input.
        """

        ## Load dataset
        if dataset is None:
            assert(dataset_path is not None)
            dataset = pickle.load(open(dataset_path, "rb"))
        self.dataset_path = dataset_path
        self.dataset_ID = dataset.ID
        with open(self.parent_path + "/parameters.txt", "a") as txt_file: ## Save the dataset parameters.
            if note is not None: txt_file.write(f"\tNote: \n{note}\n")
            txt_file.write(f"\t=================== Dataset ===================\n")
            txt_file.write(utilsforminds.strings.container_to_str(dataset.__dict__, recursive_prints= False, whether_print_object= False, limit_number_of_prints_in_container= 20) + "\n")
        
        ## For debugging: The indices of patients should not be changed.
        indices_of_patients_copy = deepcopy(dataset.indices_patients_shuffled)

        ## Expand the GridSearch object if it exists.
        experiment_settings_list_expanded = []
        list_of_paths_to_grids = []
        for experiment_setting in experiment_settings_list:
            if isinstance(experiment_setting, GridSearch):
                experiment_settings = experiment_setting.get_list_of_grids()
                for idx in range(len(experiment_settings)):
                    experiment_settings_list_expanded.append(experiment_settings[idx])
                    list_of_paths_to_grids.append(deepcopy(experiment_setting.list_of_paths_to_grids))
            else:
                experiment_settings_list_expanded.append(experiment_setting)
                list_of_paths_to_grids.append(None)

        ## Conduct experiments.
        self.experiment_results = {} ## {"model_1": [result for each split], ...}
        for experiment_setting, experiment_idx in zip(experiment_settings_list_expanded, range(len(experiment_settings_list_expanded))): ## For each model.
            ## Set model name.
            if "name" in experiment_setting.keys(): experiment_name = experiment_setting["name"]
            else: experiment_name = experiment_setting["model_class"].name
            self.experiment_results[experiment_name] = [] ## [dict_for_split_1, dict_for_split_2, ...]

            ## Save the model parameters.
            with open(self.parent_path + "/parameters.txt", "a") as txt_file:
                txt_file.write(f"\t============================{experiment_name}============================\n")
                txt_file.write(utilsforminds.strings.container_to_str(experiment_setting, file_format = "txt", paths_to_emphasize= list_of_paths_to_grids[experiment_idx]) + "\n")

            ## Set indices of patients to fit and predict
            for split_train_test, split_train_test_idx in zip(dataset.splits_train_test, range(len(dataset.splits_train_test))): ## split_train_test = {"train": [...], "test": [...]}.
                experiment_result_on_each_split = {} ## Result for this split.

                ## Set the indices of participants to learn.
                if experiment_setting["fit_on"] == "train":
                    experiment_result_on_each_split["fit_on"] = deepcopy(split_train_test["train"])
                elif experiment_setting["fit_on"] == "test":
                    experiment_result_on_each_split["fit_on"] = deepcopy(split_train_test["test"])
                elif experiment_setting["fit_on"] == "both":
                    experiment_result_on_each_split["fit_on"] = deepcopy(split_train_test["train"] + split_train_test["test"])

                if experiment_setting["model_class"].name == "Autoencoder": ## Set observabilities only for Autoencoder model.
                    dataset.set_observabilities(indices_patients_train = split_train_test["train"], indices_patients_test = split_train_test["test"])
                    ## Sanity check.
                    for pateint_idx in split_train_test["test"]:
                        assert(dataset.dicts[pateint_idx]["observability"][0][0] == 0.)
                    for pateint_idx in split_train_test["train"]:
                        assert(dataset.dicts[pateint_idx]["observability"][0][0] == 1.)
            
                ### Train and Predict.
                model_instance = experiment_setting["model_class"](**experiment_setting["init"])

                ## Discard the training samples with incomplete target labels
                if dataset.if_partial_label and not experiment_setting["model_class"].name == "Autoencoder":
                    indices_of_patients_fit_on = []
                    for idx in experiment_result_on_each_split["fit_on"]:
                        if np.sum(dataset.dicts[idx]["label_tasks_mask"]) == dataset.dicts[idx]["predictor label"].shape[1]:
                            indices_of_patients_fit_on.append(idx)
                else:
                    indices_of_patients_fit_on = experiment_result_on_each_split["fit_on"]

                ## Train if trainable.
                if "fit" in experiment_setting.keys(): experiment_result_on_each_split["loss_dict"] = model_instance.fit(dataset= dataset, indices_of_patients = indices_of_patients_fit_on, x_train = dataset.most_recent_record_dict["x_train"][split_train_test_idx], y_train = dataset.most_recent_record_dict["y_train"][split_train_test_idx], **experiment_setting["fit"])

                ## Predict.
                experiment_result_on_each_split["predictions_dict"] = model_instance.predict(dataset= dataset, indices_of_patients = split_train_test["test"], x_test = dataset.most_recent_record_dict["x_test"][split_train_test_idx], **experiment_setting["predict"]) ## possible keys: "enriched_vectors_stack", "predicted_labels_stack", "reconstructed_vectors_stack", "feature_importance_dict".
                assert("predicted_labels_stack" in experiment_result_on_each_split["predictions_dict"].keys()) ## experiment_result_on_each_split["predictions_dict"] is dictionary which should contain at least key "predicted_labels_stack".

                self.experiment_results[experiment_name].append(experiment_result_on_each_split) ## [dict_for_split_1, dict_for_split_2, ...]

        ## For debugging: The indices of patients should not be changed.
        for idx in range(dataset.num_patients):
            assert(indices_of_patients_copy[idx] == dataset.indices_patients_shuffled[idx])

        if save_result:
            with open(self.parent_path + "/experiment.obj", "wb") as experiment_file:
                # self.model_instance.clear_model()
                pickle.dump(self, experiment_file)


def plot_experimental_results(experiment, dataset = None, num_loss_plotting_points = 200, num_top_features = None, verbose = 1, output_plot_dict= None, plot_output_input_pair_map= None):
    """Plot the experimental results from the experiment object. We separate the visualization from the experiment.
    
    Parameters
    ----------
    experiment : Experiment
        The experiment object containing the results.
    dataset : Dataset
        The same dataset used in the experiment object.
    num_loss_plotting_points : int
        The number of plotting points for losses of model.
    num_top_features : int
        The number of most important features to plot.
    """

    ## Load dataset.
    if dataset is None: dataset = pickle.load(open(experiment.dataset_path, "rb"))
    assert(dataset.ID == experiment.dataset_ID) ## To check whether the dataset used in the experiment is same as this dataset.

    def recons_vectors_formatter(data, patient_idx):
        mask = dataset.dicts[patient_idx]["dynamic_vectors_decoder"]["mask"][0]
        return data * np.where(mask == 0, np.nan, 1)
    output_plot_dict_local= dict(enriched_vectors_stack= dict(shape= (5, 10), formatter= lambda data, patient_idx: data), reconstructed_vectors_stack= dict(shape= (5, 10), formatter= recons_vectors_formatter), enriched_vectors_for_images_dict= dict(shape= (0, None), formatter= lambda data, patient_idx: data))
    if output_plot_dict is not None:
        for key in output_plot_dict.keys():
            output_plot_dict_local[key] = output_plot_dict[key]
    plot_output_input_pair_map_local = merge([dict(reconstructed_vectors_stack= dict(name= "dynamic_vectors_decoder", formatter= lambda x: x["data"][0] * np.where(x["mask"][0] == 0, np.nan, x["mask"][0]))), plot_output_input_pair_map])

    # visualizations_patent_path = experiment.parent_path + "/visualizations"
    visualizations_patent_path = experiment.parent_path + "/" + utilsforminds.helpers.getNewDirectoryName(experiment.parent_path + "/", "visualizations_", root_dir= "")

    # for sub_path in ["/", "/losses/", "/feature_importance/", "/ROC_curve/", "/outputs_plot/"]:
    #     if os.path.exists(visualizations_patent_path + sub_path):
    #         rmtree(visualizations_patent_path + sub_path)
    #     else:
    #         os.mkdir(visualizations_patent_path + sub_path)
    os.mkdir(visualizations_patent_path + "/")
    os.mkdir(visualizations_patent_path + "/losses/")
    os.mkdir(visualizations_patent_path + "/feature_importance/")
    os.mkdir(visualizations_patent_path + "/ROC_curve/")
    os.mkdir(visualizations_patent_path + "/outputs_plot/")

    ### Calculate the prediction accuracy.
    ## For ROC plot
    y_true_classification_all_splits = []
    if not dataset.classification_axis is None:
        y_true_regression_all_splits = []
        regression_axes = [axis for axis in range(len(dataset.target_columns)) if axis != dataset.classification_axis]

        regression_min = np.array([dataset.feature_min_max_dict[dataset.target_columns[axis]]["min"] for axis in regression_axes])
        regression_max_minus_min = np.array([dataset.feature_min_max_dict[dataset.target_columns[axis]]["max"] - dataset.feature_min_max_dict[dataset.target_columns[axis]]["min"] for axis in regression_axes])
        regression_inverse_scale_funct = lambda arr: arr * regression_max_minus_min + regression_min
    for split_train_test in dataset.splits_train_test:
        for patient_idx in split_train_test["test"]:
            assert(dataset.dicts[patient_idx]["sample_idx"] == patient_idx)
            if dataset.classification_axis is None:
                y_true_classification_all_splits.append(dataset.dicts[patient_idx]["predictor label"][0])
            else:
                y_true_classification_all_splits.append([dataset.dicts[patient_idx]["predictor label"][0][dataset.classification_axis]])
                y_true_regression_all_splits.append([dataset.dicts[patient_idx]["predictor label"][0][regression_axes]])
    y_true_classification_all_splits = np.array(y_true_classification_all_splits)
    list_of_y_pred_classification_all_splits = []
    list_of_y_pred_regression_all_splits = []
    list_of_model_names = []

    if dataset.dataset_kind == "colorado_traffic": 
        station_id_to_results = {}
        datetime_counts = {}
        os.mkdir(visualizations_patent_path + "/OSM/")

    ## Plot the results.
    with open(experiment.parent_path + "/detail_scores.txt", "a") as txt_file_detail, open(experiment.parent_path + "/short_scores.txt", "a") as txt_file_short:
        for experiment_name in experiment.experiment_results.keys(): ## For each model,
            txt_file_detail.write(f"\t============================{experiment_name}============================\n")
            txt_file_short.write(f"\t============================{experiment_name}============================\n")
            if verbose >= 1: print(f"\t============================{experiment_name}============================")
            scores_dict_short = {"accuracy": [], "precision": {prediction_label: [] for prediction_label in dataset.prediction_labels_bag}, "recall": {prediction_label: [] for prediction_label in dataset.prediction_labels_bag}, "F1 score": {prediction_label: [] for prediction_label in dataset.prediction_labels_bag}, "RMSE": []} ## "all" for all k splits.
            feature_importance_dict_merged_across_splits = None
            list_of_model_names.append(experiment_name)
            y_pred_classification_this_model = []
            y_pred_regression_this_model = []
            # y_true_regression = []

            outputs_to_plot = dict() ## outputs_to_plot[plot_kind][img_idx]
            for key in output_plot_dict_local.keys():
                outputs_to_plot[key] = [list() for i in range(output_plot_dict_local[key]["shape"][0])]
                if key in plot_output_input_pair_map_local.keys():
                    outputs_to_plot[plot_output_input_pair_map_local[key]["name"]] = [list() for i in range(output_plot_dict_local[key]["shape"][0])]
            # patients_idc_to_plot = deepcopy(outputs_to_plot)
            plot_key_imshow_kwargs = {plot_key: dict() for plot_key in outputs_to_plot.keys()}
            # plot_key_imshow_kwargs["reconstructed_vectors_stack"] ## , zmin= 0, zmax= 1

            for split_train_test, split_train_test_idx in zip(dataset.splits_train_test, range(len(dataset.splits_train_test))): ## split_train_test = {"train": [...], "test": [...]}.
                txt_file_detail.write(f"\t==============split: {split_train_test_idx}==============\n")
                if verbose >= 2: print(f"\t==============split: {split_train_test_idx}==============")

                ## Set the result for this split.
                list_patient_idx = split_train_test["test"]
                predictions_dict = experiment.experiment_results[experiment_name][split_train_test_idx]["predictions_dict"]

                ## [vec, vec, ...] -> [scalar, scalar, ...], one-hot encoded vectors to scalar labels.
                if dataset.classification_axis is None:
                    if type(predictions_dict["predicted_labels_stack"]) == type([]): predicted_labels_classification_this_split = [dataset.prediction_label_mapping(prediction, if_many_to_one = True) for prediction in predictions_dict["predicted_labels_stack"]] ## When list, Note that 'if_many_to_one = True'.
                    else: predicted_labels_classification_this_split = [dataset.prediction_label_mapping(predictions_dict["predicted_labels_stack"][i], if_many_to_one = True) for i in range(predictions_dict["predicted_labels_stack"].shape[0])] ## When array.
                else:
                    if type(predictions_dict["predicted_labels_stack"]) == type([]):
                        assert(len(list_patient_idx) == len(predictions_dict["predicted_labels_stack"]))
                        predicted_labels_classification_this_split = [dataset.prediction_label_mapping(prediction, if_many_to_one = True)[dataset.classification_axis] for prediction in predictions_dict["predicted_labels_stack"]] ## When list, Note that 'if_many_to_one = True'.
                        predicted_labels_regression_this_split = [np.array([dataset.prediction_label_mapping(prediction, if_many_to_one = True)[axis] for axis in regression_axes]) for prediction in predictions_dict["predicted_labels_stack"]]
                    else:
                        assert(len(list_patient_idx) == predictions_dict["predicted_labels_stack"].shape[0])
                        predicted_labels_classification_this_split = [dataset.prediction_label_mapping(predictions_dict["predicted_labels_stack"][i], if_many_to_one = True)[dataset.classification_axis] for i in range(predictions_dict["predicted_labels_stack"].shape[0])] ## When array.
                        predicted_labels_regression_this_split = [np.array([dataset.prediction_label_mapping(predictions_dict["predicted_labels_stack"][i], if_many_to_one = True)[axis] for axis in regression_axes]) for i in range(predictions_dict["predicted_labels_stack"].shape[0])]
                    true_labels_regression_this_split = [dataset.dicts[patient_idx]["predictor label"][regression_axes] for patient_idx in list_patient_idx]
                    y_pred_regression_this_model += predicted_labels_regression_this_split


                ## Stack outputs (enriched vectors or reconstructed inputs..) to plot
                for plot_key in output_plot_dict_local.keys():
                    if plot_key in predictions_dict.keys():
                        if plot_key == "enriched_vectors_for_images_dict":
                            assert(output_plot_dict_local[plot_key]["shape"][0] == 0 and output_plot_dict_local[plot_key]["shape"][1] == None)
                            num_to_stack = len(predictions_dict["enriched_vectors_for_images_dict"].keys())
                            for key_img in ["reconstructed_images_dict", "gradients_images_dict"]:
                                assert(len(predictions_dict[key_img].keys()) == num_to_stack)
                            samples_idx_data = [[idx, predictions_dict[plot_key][idx]] for idx in predictions_dict["reconstructed_images_dict"].keys()]
                            sampled_output_data = [output_plot_dict_local[plot_key]["formatter"](idx_data[1], split_train_test["test"][idx_data[0]]) for idx_data in samples_idx_data]
                            outputs_to_plot[plot_key].append(sampled_output_data)
                            # patients_idc_to_plot[plot_key].append([idx_data[0] for idx_data in samples_idx_data])
                        else:
                            num_to_stack = min(max(output_plot_dict_local[plot_key]["shape"][1] // len(dataset.splits_train_test), 1), len(predictions_dict[plot_key]))
                            for i in range(output_plot_dict_local[plot_key]["shape"][0]):
                                samples_idx_data= sample(list(enumerate(predictions_dict[plot_key])), num_to_stack)
                                sampled_output_data = [output_plot_dict_local[plot_key]["formatter"](idx_data[1], split_train_test["test"][idx_data[0]]) for idx_data in samples_idx_data]
                                if plot_key in plot_output_input_pair_map_local.keys():
                                    sampled_patient_idx = [split_train_test["test"][idx_data[0]] for idx_data in samples_idx_data]
                                    for patient_idx in sampled_patient_idx:
                                        outputs_to_plot[plot_output_input_pair_map_local[plot_key]["name"]][i].append(plot_output_input_pair_map_local[plot_key]["formatter"](dataset.dicts[patient_idx][plot_output_input_pair_map_local[plot_key]["name"]]))
                                outputs_to_plot[plot_key][i] = outputs_to_plot[plot_key][i] + sampled_output_data
                                # patients_idc_to_plot[plot_key][i] + [split_train_test["test"][idx_data[0]] for idx_data in samples_idx_data]
                
                ## result dict only for this split.
                true_pred_counts = {} ## {'true label 1': {'pred label 1': 4, 'pred label 2': 7, ...}}
                for label_in_bag in dataset.prediction_labels_bag: ## true
                    true_pred_counts[label_in_bag] = {label_in_bag_: 0 for label_in_bag_ in dataset.prediction_labels_bag} ## pred in true.
                
                target_label_availability = {"classification": [], "regression": []}
                ## Counts the match/mismatch cases.
                for i, patient_idx in zip(range(len(list_patient_idx)), list_patient_idx):
                    if dataset.classification_axis is None or "label_tasks_mask" not in dataset.dicts[patient_idx].keys():
                        target_label_availability["classification"].append(1.0)
                    else:
                        target_label_availability["classification"].append(dataset.dicts[patient_idx]["label_tasks_mask"][0][dataset.classification_axis])
                        target_label_availability["regression"].append(dataset.dicts[patient_idx]["label_tasks_mask"][0][regression_axes])
                    
                    if target_label_availability["classification"][-1] == 1.0:
                        true_label = round(dataset.dicts[patient_idx]["outcome"])
                        predicted_label = predicted_labels_classification_this_split[i]
                        true_pred_counts[true_label][predicted_label] += 1
                    if dataset.classification_axis is None:
                        y_pred_classification_this_model.append(predictions_dict["predicted_labels_stack"][i])
                    else:
                        y_pred_classification_this_model.append([predictions_dict["predicted_labels_stack"][i][dataset.classification_axis]])
                        # y_pred_regression.append(predicted_labels_regression)
                        # y_true_regression.append(dataset.dicts[patient_idx]["predictor label"][regression_axes])
                            
                    if dataset.dataset_kind == "colorado_traffic" and target_label_availability["classification"][-1] == 1.0 and np.sum(target_label_availability["regression"][-1]) == len(regression_axes): 
                        count_station_id = int(dataset.dicts[patient_idx]['count_station_id'])
                        datetime_last = dataset.dicts[patient_idx]["datetime"][-1].strftime('%Y-%m') # datetime_last = dataset.dicts[patient_idx]["datetime"][-1].strftime('%Y-%m-%d')
                        if datetime_last not in datetime_counts.keys():
                            datetime_counts[datetime_last] = 1
                        else:
                            datetime_counts[datetime_last] += 1
                        if count_station_id not in station_id_to_results.keys():
                            station_id_to_results[count_station_id] = {"y_pred": {datetime_last: predictions_dict["predicted_labels_stack"][i]}, "y_true": {datetime_last: dataset.dicts[patient_idx]["predictor label"][regression_axes]}, "lat": dataset.dicts[patient_idx]["lat_raw"], "lon": dataset.dicts[patient_idx]["lon_raw"]}
                        else:
                            station_id_to_results[count_station_id]["y_pred"][datetime_last] = predictions_dict["predicted_labels_stack"][i]
                            station_id_to_results[count_station_id]["y_true"][datetime_last] = dataset.dicts[patient_idx]["predictor label"][regression_axes]
                
                ### Calculate the metrics only for this split.
                accuracy_of_this_split = sum([true_pred_counts[label_in_bag][label_in_bag] for label_in_bag in dataset.prediction_labels_bag]) / np.sum(target_label_availability["classification"]) ## Calculate Accuracy
                precision_of_this_split = {}
                recall_of_this_split = {}
                for label_in_bag_i in dataset.prediction_labels_bag: ## label_in_bag is integer; 0 or 1 for COVID; 1, 2, 3 for Alz.
                    ## Calculate Precision
                    pred_i_sum = sum([true_pred_counts[label_in_bag_j][label_in_bag_i] for label_in_bag_j in dataset.prediction_labels_bag])
                    if pred_i_sum > 0 : precision_of_this_split[label_in_bag_i] = true_pred_counts[label_in_bag_i][label_in_bag_i] / pred_i_sum
                    else: precision_of_this_split[label_in_bag_i] = np.nan
                    ## Calculate Recall
                    true_i_sum = sum([true_pred_counts[label_in_bag_i][label_in_bag_j] for label_in_bag_j in dataset.prediction_labels_bag])
                    if true_i_sum > 0 : recall_of_this_split[label_in_bag_i] = true_pred_counts[label_in_bag_i][label_in_bag_i] / true_i_sum
                    else: recall_of_this_split[label_in_bag_i] = np.nan
                result_of_this_split_str = f"\tAccuracy: {accuracy_of_this_split},\n\tPrecision: {precision_of_this_split},\n\tRecall: {recall_of_this_split}"
                if dataset.classification_axis is not None:
                    RMSE_arr = (np.sum([np.square(regression_inverse_scale_funct(predicted_labels_regression_this_split[i] - true_labels_regression_this_split[i]) * target_label_availability["regression"][i]) for i in range(len(list_patient_idx))]) / np.sum([np.sum(mask) for mask in target_label_availability["regression"]])) ** 0.5

                    result_of_this_split_str += f", \n\tRMSE: {RMSE_arr} for {[dataset.target_columns[axis] for axis in regression_axes]}"
                    scores_dict_short["RMSE"].append(np.mean(RMSE_arr))

                if verbose >= 2: print(result_of_this_split_str)
                txt_file_detail.write(f"{result_of_this_split_str}\n{true_pred_counts}\n")
                scores_dict_short["accuracy"].append(accuracy_of_this_split)
                for label_in_bag in dataset.prediction_labels_bag: ## Precision, Recall for each label.
                    scores_dict_short["precision"][label_in_bag].append(precision_of_this_split[label_in_bag])
                    scores_dict_short["recall"][label_in_bag].append(recall_of_this_split[label_in_bag])
                    scores_dict_short["F1 score"][label_in_bag].append(2 * precision_of_this_split[label_in_bag] * recall_of_this_split[label_in_bag] / max(1e-8, precision_of_this_split[label_in_bag] + recall_of_this_split[label_in_bag]))

                ## Collect and merge the feature importance, for this split.
                if num_top_features is not None and "feature_importance_dict" in predictions_dict.keys():
                    if feature_importance_dict_merged_across_splits is None: feature_importance_dict_merged_across_splits = deepcopy(predictions_dict["feature_importance_dict"])
                    else:
                        for method in predictions_dict["feature_importance_dict"].keys():
                            for feature_group in predictions_dict["feature_importance_dict"][method].keys():
                                for feature_name in predictions_dict["feature_importance_dict"][method][feature_group].keys():
                                    feature_importance_dict_merged_across_splits[method][feature_group][feature_name] = feature_importance_dict_merged_across_splits[method][feature_group][feature_name] + deepcopy(predictions_dict["feature_importance_dict"][method][feature_group][feature_name])

                ## Plot loss graph to see convergency, if exists.
                if "loss_dict" in experiment.experiment_results[experiment_name][split_train_test_idx].keys() and experiment.experiment_results[experiment_name][split_train_test_idx]["loss_dict"] is not None and num_loss_plotting_points >= 0:
                    loss_dict = experiment.experiment_results[experiment_name][split_train_test_idx]["loss_dict"]
                    dir_to_save_loss = visualizations_patent_path + f"/losses/{experiment_name}_{split_train_test_idx}/"
                    os.mkdir(dir_to_save_loss)
                    for loss_name, loss_list in loss_dict.items():
                        squeezed_list = utilsforminds.containers.squeeze_list_of_numbers_with_average_of_each_range(list_of_numbers = loss_list, num_points_in_list_out= num_loss_plotting_points)
                        utilsforminds.visualization.plot_xy_lines(range(len(squeezed_list)), [{"label": loss_name, "ydata": squeezed_list}], f'{dir_to_save_loss}{loss_name}.eps', save_tikz = False)

            list_of_y_pred_classification_all_splits.append(np.array(y_pred_classification_this_model)) ## For ROC curve plot, for this model.
            if dataset.classification_axis is not None: list_of_y_pred_regression_all_splits.append(np.array(y_pred_regression_this_model))
            
            ## Calculates the statistics for all the splits and for this model. scores_dict_short = {"accuracy": [...], "precision": {1: [...], 2: [...]}, "recall": {1: [...], 2: [...]}}
            metrics = ["accuracy"]
            if dataset.classification_axis is not None: metrics += ["RMSE"]
            for metric in metrics:
                formatted_str = f"\t{metric}. mean: {round(mean(scores_dict_short[metric]), 4)}, std: {round(std(scores_dict_short[metric]), 6)}, scores: {scores_dict_short[metric]}\n"
                if verbose >= 1: print(formatted_str)
                txt_file_detail.write(formatted_str)
                txt_file_short.write(formatted_str)
            for metric in ["precision", "recall", "F1 score"]:
                for label_in_bag in dataset.prediction_labels_bag:
                    formatted_str = f"\t{metric}_{label_in_bag}. mean: {round(mean(scores_dict_short[metric][label_in_bag]), 4)}, std: {round(std(scores_dict_short[metric][label_in_bag]), 6)}, scores: {scores_dict_short[metric][label_in_bag]}\n"
                    if verbose >= 1: print(formatted_str)
                    txt_file_detail.write(formatted_str)
                    txt_file_short.write(formatted_str)
            
            ## --- Plot Images
            if dataset.if_contains_imgs or dataset.if_contains_static_img:
                keys_for_images = []
                for key in ["reconstructed_images_dict", "gradients_images_dict", "original_img", "static_reconstructed_image_dict", "static_gradients_image_dict"]:
                    if key in predictions_dict.keys():
                        keys_for_images.append(key)
                
                imshow_params_map = dict(reconstructed_images_dict= dict(color_continuous_scale= "gray", zmin= 0, zmax= 1), static_reconstructed_image_dict= dict(color_continuous_scale= "gray", zmin= 0, zmax= 1), original_img= dict(color_continuous_scale= "gray", zmin= 0, zmax= 1), gradients_images_dict= dict(color_continuous_scale= "viridis"), static_gradients_image_dict= dict(color_continuous_scale= "viridis"), )
                indices_of_patients_for_recons_images = predictions_dict["indices_of_patients_for_recons_images"] if "indices_of_patients_for_recons_images" in predictions_dict.keys() else []
                for key_img in keys_for_images:
                    if not key_img.startswith("static_"):
                        key_data = "RNN_2D"
                    else:
                        key_data = "static_2D"
                    for patient_idx in list(set(indices_of_patients_for_recons_images) & set(predictions_dict[key_img].keys())):
                        if key_img.endswith("original_img"):
                            if not dataset.if_separate_img:
                                img_stacks = dataset.dicts[patient_idx][key_data]["data"][0]
                            else:
                                img_stacks = dataset.load_arr(sample_idx= patient_idx, input_= key_data, data_type= "data")[0]
                        else:
                            img_stacks = predictions_dict[key_img][patient_idx]
                        if not key_img.startswith("static_"):
                            for img_idx in range(img_stacks.shape[0]):
                                file_name = dataset.dicts[patient_idx][key_data]["file_name"][img_idx].split(".")[-2].split("/")[-1] if "file_name" in dataset.dicts[patient_idx][key_data].keys() else img_idx
                                img = img_stacks[img_idx, :, :, 0].T
                                fig = px.imshow(img, **imshow_params_map[key_img])
                                fig.write_html(f"{visualizations_patent_path}/outputs_plot/{experiment_name}_{key_img}_{patient_idx}_{dataset.idx_key_map[patient_idx]}_{file_name}.html")
                        else:
                            img = img_stacks[:, :, 0].T
                            fig = px.imshow(img, **imshow_params_map[key_img])
                            fig.write_html(f"{visualizations_patent_path}/outputs_plot/{experiment_name}_{key_img}_{patient_idx}_{dataset.idx_key_map[patient_idx]}.html")

            ## --- --- Plot Feature Importance.
            if feature_importance_dict_merged_across_splits is not None: ## feature_importance_dict_merged_across_splits[method][feature_group][feature_name] = [changes_on_prediction, changes_on_prediction, ...]
                ## --- Common plotting for all dataset.
                # if dataset.dataset_kind == "covid" or dataset.dataset_kind == "alz":
                for method in feature_importance_dict_merged_across_splits.keys():
                    for feature_group in feature_importance_dict_merged_across_splits[method].keys():
                        original_name_to_num_dummy = {}
                        feature_importance_dict_this_group = feature_importance_dict_merged_across_splits[method][feature_group]
                        for dummy_name in dataset.dummy_name_to_original_name:
                            original_name = dataset.dummy_name_to_original_name[dummy_name]
                            if dummy_name in feature_importance_dict_this_group.keys():
                                if original_name in original_name_to_num_dummy:
                                    original_name_to_num_dummy[original_name] += 1
                                else:
                                    original_name_to_num_dummy[original_name] = 1
                                if original_name in feature_importance_dict_this_group.keys():
                                    feature_importance_dict_this_group[original_name] = [feature_importance_dict_this_group[original_name][i] + feature_importance_dict_this_group[dummy_name][i] for i in range(len(feature_importance_dict_this_group[dummy_name]))]
                                else:
                                    feature_importance_dict_this_group[original_name] = deepcopy(feature_importance_dict_this_group[dummy_name])
                                if dummy_name != original_name:
                                    del feature_importance_dict_this_group[dummy_name]
                        for original_name in original_name_to_num_dummy.keys():
                            for i in range(len(feature_importance_dict_this_group[original_name])):
                                feature_importance_dict_this_group[original_name][i] /= original_name_to_num_dummy[original_name]
                        if len(next(iter(feature_importance_dict_this_group.values()))) >= 1: ## Plot if at least one importance exists.
                            ## Get name, mean, std of each feature importance.
                            num_top_features_local = min(num_top_features, len(feature_importance_dict_this_group))
                            name_mean_std_list = [[feature_name, mean(feature_importance_dict_this_group[feature_name], default= 0.), std(feature_importance_dict_this_group[feature_name], default= 0.)] for feature_name in feature_importance_dict_this_group.keys()] ## [[name, mean, std], [name, mean, std], ...]
                            ## Cut top num_top_features features.
                            name_mean_std_list.sort(key = lambda x: x[1], reverse= True) ## sort by mean
                            feature_importance_names = [name_mean_std_list[i][0] for i in range(num_top_features_local)]
                            feature_importance_means = [name_mean_std_list[i][1] for i in range(num_top_features_local)]
                            feature_importance_stds = [name_mean_std_list[i][2] for i in range(num_top_features_local)] ## Standard Deviation across the patients in test set.

                            utilsforminds.visualization.plot_bar_charts(path_to_save = f"{visualizations_patent_path}/feature_importance/{experiment_name}_{feature_group}_{method}_with_std.eps", name_numbers = {"Mean": feature_importance_means}, xlabels = feature_importance_names, xlabels_for_names = None, xtitle = None, ytitle = "Importance", bar_width = 'auto', alpha = 0.8, colors_dict = None, format = 'eps', diagonal_xtickers = False, name_errors = {"Mean": feature_importance_stds}, name_to_show_percentage = None, name_not_to_show_percentage_legend = None, fontsize = 12, title = None, figsize = None, ylim = None, fix_legend = True, plot_legend = False, save_tikz = False, horizontal_bars= True)

                ## Plot outputs
                if len(outputs_to_plot) > 0:
                    for plot_key in outputs_to_plot.keys():
                        if len(outputs_to_plot[plot_key]) > 0 and len(outputs_to_plot[plot_key][0]) > 0:
                            for img_idx in range(len(outputs_to_plot[plot_key])):
                                utilsforminds.visualization.plot_multiple_matrices(container_of_matrices= outputs_to_plot[plot_key][img_idx], path_to_save= f"{visualizations_patent_path}/outputs_plot/{experiment_name}_{plot_key}_{img_idx}.html", imshow_kwargs= plot_key_imshow_kwargs[plot_key])

                ## --- Additional plotting dedicated to the specific dataset.
                if dataset.dataset_kind == "alz":
                    for method in feature_importance_dict_merged_across_splits.keys():
                        ## --- --- SNP plot.
                        feature_importance_dict_snp = feature_importance_dict_merged_across_splits[method]["SNP"]
                        names_SNPs = list(dataset.df_dict["snp"].columns)[1:]
                        if len(next(iter(feature_importance_dict_snp.values()))) >= 1: ## Plot if at least one importance exists.
                            importance_of_snp_at_idc_chr_not_X_list = [] ## plottable SNPs < 1224 total SNPs.
                            for snp_idx in dataset.idc_chr_not_X_list: ## For each snp feature.
                                importance_of_snp_at_idc_chr_not_X_list.append(mean(feature_importance_dict_snp[names_SNPs[snp_idx]]))
                            weights_appended_SNPs_info_df = dataset.reordered_SNPs_info_df.copy()
                            weights_appended_SNPs_info_df["Importance"] = importance_of_snp_at_idc_chr_not_X_list

                            ## --- Actual plots.
                            ## Scatter plots.
                            utilsforminds.visualization.plot_group_scatter(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/chromosome_{experiment_name}_{method}.eps", xlabel= 'Chromosome', group_column = "chr", y_column = "Importance", color_column = "chr_colors", group_sequence = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 17, 19, 20, 21])

                            utilsforminds.visualization.plot_group_scatter(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/alzgene_{experiment_name}_{method}.eps", xlabel= 'AlzGene', group_column = "AlzGene", y_column = "Importance", color_column = None, rotation_xtickers = 45, group_sequence= ['MTHFR', 'ECE1', 'CR1', 'LDLR', 'IL1B', 'BIN1', 'TF', 'NEDD9', 'LOC651924', 'TFAM', 'CLU', 'IL33', 'DAPK1', 'SORCS1', 'GAB2', 'PICALM', 'SORL1', 'ADAM10', 'ACE', 'PRNP'])

                            utilsforminds.visualization.plot_group_scatter(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/location_{experiment_name}_{method}.eps", xlabel= 'location', group_column = "location", y_column = "Importance", color_column = None, rotation_xtickers = 45)

                            utilsforminds.visualization.plot_group_scatter(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/identified_group_{experiment_name}_{method}.eps", xlabel= 'Identified Group', group_column = "identified_group", y_column = "Importance", color_column = None, rotation_xtickers = 45, num_truncate_small_groups= 140)

                            ## Bar charts plots for individual SNPs.
                            for group_column in ['chr', 'AlzGene', 'location', 'identified_group']:
                                utilsforminds.visualization.plot_top_bars_with_rows(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/topbars_SNP_{group_column}_{experiment_name}_{method}.eps", colors_rotation = None, color_column = group_column + "_colors", order_by = "Importance", x_column = "SNP", show_group_size = False, xlabel = "SNP", ylabel = "Importance", num_bars = 10, num_rows = 2, re_range_max_min_proportion = None, rotation_xtickers = 45, save_tikz = False)

                            utilsforminds.visualization.plot_top_bars_with_rows(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/topbars_SNP_{experiment_name}_{method}.eps", colors_rotation = ["blue"], order_by = "Importance", x_column = "SNP", show_group_size = False, xlabel = "SNP", ylabel = "Importance", num_bars = 10, num_rows = 3, re_range_max_min_proportion = None, rotation_xtickers = 45)

                            ## Bar charts plots for each group.
                            for color_column, group_column, xlabel, rotation_xtickers in zip(['chr_colors', 'AlzGene_colors', 'location_colors', 'identified_group_colors'], ["chr", "AlzGene", "location", "identified_group"], ["Chromosome", "AlzGene", "Location", "Identified Group"], [45, 45, 45, 0]):
                                utilsforminds.visualization.plot_top_bars_with_rows(weights_appended_SNPs_info_df, f"{visualizations_patent_path}/feature_importance/topbars_{xlabel}_{experiment_name}_{method}.eps", color_column = color_column, order_by = "Importance", group_column = group_column, xlabel = xlabel, ylabel = "Importance", num_bars = 10, num_rows = 2, re_range_max_min_proportion = None, rotation_xtickers = rotation_xtickers)
                        
                        ### MRI Region Of Interest plot.
                        labels_path_dict = {"FS": f"{dataset.path_to_dataset}/fs_atlas_labels.csv", "VBM": f"{dataset.path_to_dataset}/longitudinal imaging measures_VBM_mod_final.xlsx"}
                        for dynamic_modality in ["FS", "VBM"]:
                            feature_importance_dict_dynamic_modality = feature_importance_dict_merged_across_splits[method][dynamic_modality]
                            if len(next(iter(feature_importance_dict_dynamic_modality.values()))) >= 1: ## Plot if at least one importance exists.
                                feature_importance_weights = [mean(feature_importance_dict_dynamic_modality[feature_name]) for feature_name in dataset.dynamic_feature_names_for_each_modality[dynamic_modality]] ## For each dynamic feature.
                                draw_brains(weights_of_rois = feature_importance_weights, path_to_save = f"{visualizations_patent_path}/feature_importance/{dynamic_modality}_{experiment_name}_{method}.png", path_to_labels = labels_path_dict[dynamic_modality], modality = dynamic_modality)
        
            if dataset.dataset_kind == "colorado_traffic":
                time_points = heapq.nlargest(3, datetime_counts.items(), key=itemgetter(1))
                time_points = [e[0] for e in time_points]
                os.mkdir(visualizations_patent_path + f"/OSM/{experiment_name}/")
                plot_on_OSM(station_id_to_results= station_id_to_results, dir_path_to_save = visualizations_patent_path + f"/OSM/{experiment_name}/", target_columns= dataset.target_columns, time_points= time_points, zoom = 5)

    if dataset.dataset_kind == "colorado_traffic":
        color_models = ["red", "orange", "green", "blue", "magenta", "black", "brown", "pink", "yellow"]
        line_names = []
        dash = []
        color = []
        xs, ys = [], []
        for model_idx in range(len(list_of_y_pred_regression_all_splits)):
            xs_curr = [list_of_y_pred_regression_all_splits[model_idx][pred_idx][0] for pred_idx in range(list_of_y_pred_regression_all_splits[model_idx].shape[0])]
            ys_curr = [max(min(list_of_y_pred_classification_all_splits[model_idx][pred_idx][0], 1.0), 0.) for pred_idx in range(list_of_y_pred_classification_all_splits[model_idx].shape[0])]
            xs_curr_arr, ys_curr_arr = np.array(xs_curr), np.array(ys_curr)
            m, c = np.linalg.lstsq(np.vstack([xs_curr_arr, np.ones(len(xs_curr_arr))]).T, ys_curr_arr, rcond=None)[0]
            xs.append(xs_curr)
            ys.append([m * xs_curr[i] + c for i in range(len(xs_curr))])
            print(f"model: {list_of_model_names[model_idx]}, m: {m}")
            # model_name = list_of_model_names[model_idx]
            line_names.append(f"{model_name}")
            dash.append(None)
            color.append(color_models[model_idx])
        utilsforminds.visualization.plot_xs_ys_lines(xs= xs, ys= ys, pair_names= line_names, path_to_save= visualizations_patent_path + f"/OSM/correlation", title= "Congestion Score v.s. Accidents Prob", xaxis_title= "Congestion Score", yaxis_title = "Accidents Prob", num_points_smooth= (-4, +4), dash= dash)
            
    ## Plot the ROC curve
    utilsforminds.visualization.plot_ROC(path_to_save= visualizations_patent_path + "/ROC_curve/roc_curve", y_true= y_true_classification_all_splits, list_of_y_pred= list_of_y_pred_classification_all_splits, list_of_model_names = list_of_model_names, list_of_class_names = dataset.class_names, title = None, xlabel = 'False Positive Rate', ylabel = 'True Positive Rate', colors = None, linewidth = 1, extension = "eps", fontsize_ratio= 0.7)
                    
    print(f"Experimental results are saved in {experiment.parent_path}")

def vector_with_one_hot_encoding(vector):
    """Find the index of maximum element, and set it to 1, and set others to 0
    
    Examples
    --------
    print(vector_with_one_hot_encoding([0.3, -0.1, 0., 0.5, 0.2]))
        [0, 0, 0, 1, 0]
    print(vector_with_one_hot_encoding([1.5, -0.4, 1.5, 0.5, 0.4]))
        [1, 0, 0, 0, 0]
    """

    if type(vector) == type([]) or type(vector) == type(()): ## list or tuple
        num_elements = len(vector)
    else: ## Array
        num_elements = vector.shape[0]
    encoded = [0 for i in range(num_elements)]
    encoded[np.argmax(vector)] = 1
    return encoded

def draw_brains(weights_of_rois, path_to_save, path_to_labels, modality = "FS"):
    """Plot the Region Of Interests from FS or VBM rois in ADNI dataset.
    
    Attributes
    ----------
    path_to_labels : str
        Possibly one of "./inputs/alz/data/fs_atlas_labels.csv" or "./inputs/alz/data/longitudinal imaging measures_VBM_mod_final.xlsx"
    weights_of_rois : list
        List of weights on 90 rois of VBM or FS.
    """
    if any(pd.isna(weights_of_rois)):
        print(f"WARNING: This function draw_brains encounters invalid numbers such as nan/inf, so will be passed and do nothing.")
        return None
    
    if modality == "FS":
        fs_roi_map = FSRegionOfInterestMap()
        fs_label_df = pd.read_csv(path_to_labels)
        for index, row in fs_label_df.iterrows():
            atlas = row["Atlas"]
            rois = row[atlas].split("+")
            for roi in rois:
                fs_roi_map.add_roi(roi, weights_of_rois[index], atlas)
        fs_roi_map.build_map(smoothed=True)
        fs_roi_map.save(path_to_save, "FS") ## label at upper left box.
    elif modality == "VBM":
        ## Load vbm labels.
        vbm_cleaner = VBMDataCleaner()
        vbm_cleaner.load_data(path_to_labels)
        vbm_labels = vbm_cleaner.clean()
        
        ## plot VBM
        vbm_roi_map = VBMRegionOfInterestMap()
        for label, weight in zip(vbm_labels, weights_of_rois):
            vbm_roi_map.add_roi(label, weight)

        vbm_roi_map.build_map(smoothed=True)
        #vbm_roi_map.plot(time)
        vbm_roi_map.save(path_to_save, "VBM") ## label at upper left box.
    else:
        raise Exception(NotImplementedError)

def convert_time_to_minutes(time_str):
    if time_str == "no image":
        return time_str
    time_str = time_str[1:]
    hours = int(time_str.split("h")[0])
    minutes = int((time_str.split("h")[1]).split("m")[0])
    date = (time_str.split("m")[1]).split("_")[0]
    date_format = "%Y-%m-%d"
    days = (datetime.strptime(date, date_format) - datetime.strptime('1022-11-21', date_format)).days

    return minutes + 60 * hours + 60 * 24 * days

def get_strides_to_target_output_shape(output_shape, input_shape, kernel_size):
    strides= [0, 0]
    for axis in [0, 1]:
        # kernel_size[axis] = dataset.shape_2D_records[axis] % (output_dict["dynamic_images_decoder"].shape[1 + axis] - 1)
        # strides[axis] = dataset.shape_2D_records[axis] // (output_dict["dynamic_images_decoder"].shape[1 + axis] - 1)
        strides[axis] = (output_shape[axis] - kernel_size[axis]) // (input_shape[axis] - 1)
        if (output_shape[axis] - kernel_size[axis]) % (input_shape[axis] - 1) != 0: strides[axis] += 1
    return strides

def get_OSM_arr(latitude, longitude, zoom = 18):
    assert(zoom <= 18)
    #creating OSM figure: https://stackoverflow.com/questions/74118490/get-open-street-map-image-of-an-area-by-longitude-and-latitude-with-highlight-o
    # df = pd.DataFrame({'lat':[random.uniform(39.86, 40.08) for i in range(100)],
    #     'lon':[random.uniform(-83.1, -82.86) for i in range(100)],
    #     'category':[random.choice(['a', 'b', 'c', 'd']) for i in range(100)]})

    fig = px.scatter_mapbox(
                            # df, 
                            # lat='lat',
                            # lon='lon',
                            # color='category',
                            center={'lat':latitude,
                                    'lon':longitude},
                            zoom=zoom)

    fig.update_layout(mapbox_style='open-street-map')

    # fig.show()

    #convert Plotly fig to  an array: https://stackoverflow.com/questions/62039490/convert-plotly-image-byte-object-to-numpy-array
    fig_bytes = fig.to_image(format="png")
    buf = BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

def plot_on_OSM(station_id_to_results, dir_path_to_save, target_columns, time_points, zoom = 15):
    # time_points = sample(list(station_id_to_results["y_pred"].keys()), k= num_time_points)
    # target_columns=['future_congestion', 'near_crash']
    assert(len(target_columns) == 2)
    # for target_idx in range(len(target_columns)):
    for result_type in ["y_pred", "y_true"]:
        for time_point in time_points:
            lat = []
            lon = []
            color = []
            size = []      
            for station_id in station_id_to_results.keys():
                if time_point in station_id_to_results[station_id][result_type].keys():
                    lat.append(station_id_to_results[station_id]["lat"])
                    lon.append(station_id_to_results[station_id]["lon"])
                    if result_type == "y_pred":
                        color.append(station_id_to_results[station_id][result_type][time_point][1])
                        size.append(station_id_to_results[station_id][result_type][time_point][0])
                    elif result_type == "y_true":
                        color.append(station_id_to_results[station_id][result_type][time_point][0][1])
                        size.append(station_id_to_results[station_id][result_type][time_point][0][0])
            fig = px.scatter_mapbox(
                    lat= lat, lon= lon,
                    color= color, color_continuous_scale= 'Bluered',
                    size= size, 
                    size_max= 15,
                    range_color= [0., 1.],
                    center={'lat': mean(lat), 'lon': mean(lon)},
                        zoom= zoom)
            # fig.update_traces() # marker_sizemin=3, marker_cmin = 0., marker_cmax = 1.0,
            fig.update_layout(mapbox_style='open-street-map')
            fig.write_html(f"{dir_path_to_save}/both-targets_{result_type}_{str(time_point).replace(' ', '_').replace(':', '-')}.html")

if __name__ == "__main__":
    # arr = get_OSM_arr(longitude= -105.077087, latitude= 40.57289)
    # print(arr.shape)
    pass