import pickle

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.utils import resample, shuffle
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import pickle

import lightgbm as lgb
from xgboost import XGBClassifier


class evaluation:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    # performing a full prediction and evaluation
    def feval(self):
        """
        The feval function analyses fits and trains models prints out the confusion matirces of the models created, a
        table of the model used and their hpyerparameters, a table of the models with the recall, precision, f1, and
        AUC scores and a plot of the ROC curve.
        """
        models_used = pd.DataFrame()
        comp_df = pd.DataFrame()
        models = {"Logistic Regression_ypred": LogisticRegression(solver='liblinear', C=10.0, random_state=0),
                  "Decision Tree_ypred": DecisionTreeClassifier(criterion="entropy", max_depth=4),
                  "Random Forest_ypred": RandomForestClassifier(random_state=42),
                  "Support Vector Machines_ypred": svm.SVC(kernel='rbf', probability=True),
                  "Light Gradient Boost_ypred": lgb.LGBMClassifier(),
                  "XGBoost_ypred": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')}

        for i, model in models.items():
            model_name = i.split("_")[0]

            # fit the model
            model.fit(self.X_train, self.y_train)

            # predicting using test and test set
            pred_test = model.predict(self.X_test)
            pred_train = model.predict(self.X_train)

            # predicting probabilities that will be used for calculating the auc and roc curve
            prob_test = model.predict_proba(self.X_test)
            prob_train = model.predict_proba(self.X_train)

            # keep probabilities for the positive outcome only
            prob_test = prob_test[:, 1]
            prob_train = prob_train[:, 1]

            # calculate scores
            model_auc_test = roc_auc_score(self.y_test, prob_test)
            model_auc_train = roc_auc_score(self.y_train, prob_train)
            # summarize scores

            # print('Logistic: ROC AUC=%.3f' % (lr_auc))
            # calculate roc curves

            model_fpr, model_tpr, _ = roc_curve(self.y_test, prob_test)
            # plot the roc curve for the model

            # creating the confusion matrix
            matrix_test = confusion_matrix(self.y_test, pred_test)
            matrix_train = confusion_matrix(self.y_train, pred_train)

            # calculating various evaluation metrics
            precision_test, recall_test, fscore_test, support_test = score(self.y_test, pred_test, average='macro')
            precision_train, recall_train, fscore_train, support_train = score(self.y_train, pred_train,
                                                                               average='macro')

            i = {"Model": model_name, "Model_name": model_name, "model_": model,
                 "Precision_test": precision_test, "Precision_train": precision_train,
                 "Recall_test": recall_test, "Recall_train": recall_train,
                 "F-Score_test": fscore_test, "F-Score_train": fscore_train,
                 "AUC_test": model_auc_test, "AUC_train": model_auc_train}
            msaved = {"model_name": model_name, "model": model}

            # printing confusion matrix
            models_used = models_used.append(msaved, ignore_index=True)
            comp_df = comp_df.append(i, ignore_index=True)
            print()
            print(model_name + " test set confusion matrix")
            print(matrix_test)
            print(model_name + " train set confusion matrix")
            print(matrix_train)

            # creating the plots
            pyplot.plot(model_fpr, model_tpr, marker='.', label=model_name)
            pyplot.xlabel('False Positive Rate')
            pyplot.ylabel('True Positive Rate')
            # show the legend
            pyplot.legend()

        # printing models used, and train and test dataset
        print()
        print("Models used in prediction:\n {}".format(models_used))
        print()
        print(comp_df[["Model", "Precision_test", "Precision_train", "Recall_test", "Recall_train",
                       "F-Score_test", "F-Score_train", "AUC_test", "AUC_train"]])
        # show the plot
        pyplot.show()
        return comp_df, models_used

    def choice(self, df, auc_thresh, crit, filename):  # , crit, auc_thresh
        """
        fidn the modl with the highes value of criteria
        use it to find the model
        save the model with the right name

        check and make sure the acu test and train is not greater than 0.3
        2.selected metrics should not have a difference greater than 0.3
        1.pick all models whose auc test is above the threshold


        check if the selected model has a criteria difference greater than 0.3
        """
        nex = crit.replace("test", "train")

        useful = df[df["AUC_test"] > auc_thresh]

        try:

            assert len(useful) > 0, "All models are not reliable "
            usefuler = useful.loc[abs(useful["AUC_test"] - useful["AUC_train"]) <= 0.3]
            assert len(usefuler) > 0, "AUCs  overfitting "
            usefulerr = usefuler.loc[abs(useful[crit] - useful[nex]) <= 0.3]
            assert len(usefulerr) > 0, "Model criteria shows overfitting"
            modelusedd = usefulerr[usefulerr[crit] == usefulerr[crit].max()]["model_"]
            print(usefulerr)
            print(modelusedd)
            pickle.dump(modelusedd.values[0], open(filename, 'wb'))
        except AssertionError as e:
            print(e.args[0])

#         print(crit)
#         print(modelusedd)
#         print(crit)
#         print(auc_thresh)

#         pickle.dump(model, open(filename, 'wb'))


#         crit = input("Model selection criteria (precision, recall, f-score: \n")
#         data = ["Model", "Precision_test", "Precision_train", "Recall_test", "Recall_train",
#                                 "F-Score_test", "F-Score_train", "Support_test", "Support_train"]

#         if crit == "precision":
#             comp_df= comp_df[comp_df["Precision_test"] == comp_df["Precision_test"].max()]
#             print(comp_df[data])
#         elif crit == "recall":
#             comp_df= comp_df[comp_df["Recall_test"] == comp_df["Recall_test"].max()]
#             print(comp_df[data])
#         elif crit == "f-score":
#             comp_df= comp_df[comp_df["Recall_test"] == comp_df["Recall_test"].max()]
#             print(comp_df[data])
#         else:
#             print("Invalid criteria")
