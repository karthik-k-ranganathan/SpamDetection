import sys
import os
import yaml
import json
from termcolor import colored

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # to train the model
from sklearn.feature_extraction.text import TfidfVectorizer  # to tokenize input data
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # to find the accuracy score
from sklearn.pipeline import (
    Pipeline,
)  # pipeline is used to execute more than one item in the pipeline
from sklearn.linear_model import LogisticRegression  # trying to build different models
from sklearn.naive_bayes import MultinomialNB  # trying to do build different models

sys.path.append(
    os.path.abspath("../configuration")
)  # this is needed if the file is in a different path
from model_info import ModelInfo


class SpamDetector:
    def __load_json_configurations(self):
        self.__configs = {
            "enable_trace": False,
            "log_level": "DEBUG",
            "log_path": ".",
        }  # initializing the values

        config_path = os.path.join(self.__script_dir, "../configuration/config.json")
        # print(f"The config path : {config_path}")

        with open(config_path, "r") as file:
            config = json.load(file)

        self.__configs["enable_trace"] = config["Settings"]["enable_trace"]
        self.__configs["log_level"] = config["Settings"]["log_level"]
        self.__configs["log_path"] = config["Settings"]["log_path"]
        # self.__configs["host"] = config["Database"]["host"]

    # TODO: KR: Remove method. This is for testing purpose
    def get_configs(self):
        return self.__configs

    # Not Implemented
    def __load_yaml_configurations(self):
        with open("../configuration/config.yaml") as f:
            config = yaml.safe_load(f)

        self.__configs["enable_trace"] = config["Settings"]["enable_trace"]
        self.__configs["log_level"] = config["Settings"]["log_level"]
        self.__configs["log_path"] = config["Settings"]["log_path"]
        # self.__configs["host"] = config["Database"]["host"]

    # Private method: Loading the raw data to a data frame
    def __load_data(self):
        raw_data_path = os.path.join(self.__script_dir, "../data/raw/mail_data.csv")
        if self.__configs["enable_trace"]:
            print(f"The raw data directory: {raw_data_path}")

        df = pd.read_csv(raw_data_path)
        return df

    # private method: transforming the data
    def __explorative_data_analysis_and_transformation(self, df):
        # Checking for null
        empty_record_total = df.isna().sum()
        if self.__configs["enable_trace"]:
            print(f"empty_record_total:{empty_record_total}")

        # Add a category which is numerical
        df["IsSpam"] = df["Category"].map(
            {"spam": 1, "ham": 0}
        )  # an alternate way to set the values instead of using lambda

        return df

    # Defining all the models that i want to try into a dictionary
    # TODO: KR: Read these information from a config file and write a function to load this as a pipe
    # TODO: KR: Make the algorithms dynamic so that we can build a class to add new models without touching the code
    def __get_training_models(self):
        self.__models_dict = {
            "Logistic Regression": LogisticRegression(),
            "Naive Bayes Multinomia": MultinomialNB(),
        }
        return self.__models_dict

    def __select_best_model(self, df):
        # Loop through the models and for each model you will have to train and test and get the accuracy
        # Since we canot loop and reference via index, we are converting the dictionary to a list
        models_dict = self.__get_training_models()
        models_list = list(models_dict)
        models_values_list = list(models_dict.values())
        # temp_average_score = 0.0
        models_info = list()
        selected_model = ModelInfo()
        recomended_model_count = len(models_list)
        trace_enabled = self.__configs["enable_trace"]

        if trace_enabled:
            print(f"Recomended Models Count: {recomended_model_count}")

        for l in range(len(models_list)):
            print(
                colored(
                    f"\nCurrent Model: {models_list[l]} [{models_values_list[l]}]",
                    "yellow",
                )
            )
            model = models_values_list[l]
            # vectorize the x_train data and x_test data
            vectorizer = TfidfVectorizer(
                min_df=1, stop_words="english", lowercase=True
            )  # object to tokenize the messages

            # Calculating the mean of the accuracy so that way we know if this would work for various test data
            meanAccuracy_train_data = 0.0
            repeatCount = 20  # Ensure we get the accuracy for different train data
            for i in range(0, repeatCount, 1):
                x_train, x_test, y_train, y_test = train_test_split(
                    df["Message"], df["IsSpam"], test_size=0.20
                )

                x_train_vect = vectorizer.fit_transform(x_train)
                x_test_vect = vectorizer.transform(x_test)

                # fit the vector data into the model
                model.fit(x_train_vect, y_train)

                y_pred = model.predict(x_test_vect)  # test with vectorized test data

                # calculate the accuracy score for the model
                accuracyScore = accuracy_score(y_test, y_pred)
                if trace_enabled:
                    print(
                        f"\t The accuracy score for the model {model} iteration {i} is {accuracyScore}"
                    )
                meanAccuracy_train_data += accuracyScore

                # Predicting
                # y_train_pred = pipe.predict

            meanAccuracy_train_data = meanAccuracy_train_data / repeatCount
            print(
                colored(
                    f"Mean Accuracy Score [{repeatCount} Iterations]: {meanAccuracy_train_data}",
                    "magenta",
                )
            )

            if trace_enabled:
                print(
                    f"\tcurrent_model.AccuracyScore: {meanAccuracy_train_data}, \n\tcurrent_model.ModelName: {models_list[l]}, \n\tcurrent_model.Model: {model}, \n\tcurrent_model.Tokenizer: {vectorizer}"
                )

            current_model = ModelInfo()
            current_model.AccuracyScore = meanAccuracy_train_data
            current_model.ModelName = models_list[l]
            current_model.Model = model
            current_model.Tokenizer = vectorizer
            models_info.append(current_model)

            if trace_enabled:
                print(
                    f"\tmodels_info_count: {len(models_info)}, models_info_type: {type(models_info)}"
                )
                print(f"\t index:{l}, {models_info[l].Model}")
            sel_model_accuracy = np.nan_to_num(selected_model.AccuracyScore)

            # print(f"Selected Model's Accuracy:{sel_model_accuracy}, Current Model's Accuracy:{current_model.AccuracyScore}")
            if sel_model_accuracy < current_model.AccuracyScore:
                selected_model = current_model

        # if the accuracy score is mre than the other ones, than we need to take the best algorithm to predict.
        print(f"\nSumary:")
        for m in models_info:
            print(
                colored(
                    f"Model Name:{m.ModelName} with accuracy {m.AccuracyScore}", "cyan"
                )
            )
        print(f"\n")

        return selected_model

    def IsSpam(self, message):
        input_data_features = self.selected_model.Tokenizer.transform([message])

        output = self.selected_model.Model.predict(input_data_features)
        result_code = output[0]
        if 1 == result_code:
            result_text = "Spam"
        else:
            result_text = "Not a Spam"
        print(
            f"The prediction by the algorithm {x.selected_model.Model} is {result_text} [{result_code}]"
        )

        return result_code, result_text
        # response_data = {
        #     "code": result_code,
        #     "description": result_text
        # }

        # return json.dump(response_data)

    def __init__(self) -> None:
        self.__script_dir = os.path.dirname(
            __file__
        )  # Getting the current file's directory path

        # load configuration files
        self.__load_json_configurations()
        # read raw data
        self.__df = self.__load_data()
        # transform the data
        self.__df = self.__explorative_data_analysis_and_transformation(self.__df)
        # Selecting the best model based on the data that was given
        self.selected_model = self.__select_best_model(self.__df)
        # print(f"\n The selected Model's Name: {self.selected_model.ModelName}")


x = SpamDetector()
print("selected model:", x.selected_model.ModelName, "\n")
