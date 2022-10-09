import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn import metrics
from sklearn.metrics import auc
import random 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

def preprocess():
    data= pd.read_csv('/Users/ryan/Desktop/ResearchRepo/HSCRPmlResearch/preprocessed_data.csv').drop(columns=['Unnamed: 0','Waist_C', 'Carb_diet'])
    y = data['HSCRP_cat'].replace({'low': 0, 'high': 1})
    data['Gender'] = data['Gender'].replace({1:0,2:1})
    x = data.drop(columns=['HSCRP_cat', 'HSCRP'])
    return y, x

class instantiate_model:
    def __init__(self):
        super().__init__()
    def LogisticRegression(self, C=0, penalty='none', solver='liblinear',max_iter: int= 1000, class_weight=None, l1_ratio=None):
        return LogisticRegression(C=C, penalty=penalty,solver=solver,max_iter=max_iter, class_weight=class_weight, l1_ratio=l1_ratio)
    def KNN(self, n_neighbors: int= 5):
        return KNeighborsClassifier(n_neighbors)
    def DecisionTree(self, max_depth=10, min_samples_split=2,max_leaf_nodes=None, 
                        min_impurity_decrease=0.0):
        return DecisionTreeClassifier(max_depth= max_depth, min_samples_split=min_samples_split,
                                    max_leaf_nodes=max_leaf_nodes, min_impurity_decrease =min_impurity_decrease)
    def NaiveBayes(self):
        return GaussianNB()
    def RandomForest(self, class_weight=None):
        return RandomForestClassifier(class_weight=class_weight)
    def GradientBoost(self,loss='deviance'):
        return GradientBoostingClassifier(loss=loss)
    def tfLogisticRegression(self, activation_function = 'sigmoid' ):
        tf_log_reg = tf.keras.Sequential()
        tf_log_reg.add(tf.keras.layers.Dense(1, activation= activation_function))
        return tf_log_reg

class model_metrics:
    def classification_metrics(self, y_actual: list, y_pred: list, y_prob: list) -> list:
        accuracy = accuracy_score(y_actual,y_pred)
        ROC_AUC = roc_auc_score(y_actual,y_prob)
        precision, recall, thresholds = metrics.precision_recall_curve(y_actual, y_prob)
        PR_AUC = auc(recall, precision)
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(y_actual, y_pred).ravel()
        recall = true_positives /(true_positives + false_negatives)
        precision = true_positives/(true_positives+false_positives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        brier_score = sklearn.metrics.brier_score_loss(y_actual, y_prob)
    
        return accuracy, ROC_AUC, PR_AUC, brier_score, f1_score, recall, precision, int(true_negatives), int(false_negatives), int(true_positives), int(false_positives)

    def plot_lr_tf_metrics(self, data):
        plt.figure(figsize = (13,8))
        loss= pd.DataFrame({"Loss":data.history['loss'], "Val Loss": data.history['val_loss']})
        sns.set_style("darkgrid")
        ax = sns.lineplot(data= loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title( "Training vs. Validation Loss") 
        plt.show(ax)

    def plot_roc_auc(self, X,y, models, model_name):
        for model in models:
            if "sklearn" in "{}".format(type(model)):
                y_pred = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_pred)
                auc_sklearn= auc(fpr, tpr)
                plt.plot(fpr, tpr, label='{} (AUC = {:.3f})'.format(model_name, auc_sklearn))     

            elif  "keras" in "{}".format(type(model)):
                y_pred = model.predict(X).ravel()
                fpr, tpr, _= roc_curve(y, y_pred)
                auc_keras = auc(fpr,tpr)
                plt.plot(fpr, tpr, label='{} (AUC = {:.3f})'.format(model_name, auc_keras))
        plt.show()

class fit_model:
    def __init__(self):
        self.model_inst = instantiate_model()
        self.model_metrics__ = model_metrics()
    
    def get_weights(self, model):
        return model.get_weights()

    def set_weights(self, model, model_weights: list):
        model.set_weights(model_weights)
        return model

    def check_data(self, data):
        return [False if (np.isnan(data).sum() > 0).any() else True][0]
    
    def train_model(self, model, data_dict, class_weights="balanced", optimizer=None, learning_rate= None, loss_function=None):
        '''Returns trained model'''

        assert self.check_data(data_dict['x_train'])
        assert self.check_data(data_dict['y_train'])
        assert self.check_data(data_dict['x_val'])
        assert self.check_data(data_dict['y_val'])

        if "sklearn" in "{}".format(type(model)):
            trained_model = model.fit(data_dict['x_train'],data_dict['y_train'])

        elif "keras" in "{}".format(type(model)):
            model.build(input_shape=data_dict['x_train'].shape)
            try:
                model_for_set= self.model_inst.tfLogisticRegression()
                model_for_set.build(input_shape = data_dict['x_train'].shape)
                self.set_weights(model,model_weights =self.get_weights(model_for_set ))
                trained_model = model.fit(x=data_dict['x_train'], y=data_dict['y_train'], batch_size=128,
                            validation_data=(data_dict['x_val'],data_dict['y_val']), epochs=300, shuffle=True, 
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)],verbose=0)
              
            except Exception as err:
                print(f"Model training for {model} has failed\n")
                raise err
            else:
                print(f"Model training for {model} was successful\n")
        else:
            print("Model passed in is neither sklearn, tensorflow, or keras")
        return trained_model

    def test_model(self, trained_model, x_test):

        if "sklearn" in "{}".format(type(trained_model)):
            predicted = (trained_model.predict(x_test)).astype("int32")
            predicted_prob = (trained_model.predict_proba(x_test)[:,1]).astype("float")
        elif "keras" in "{}".format(type(trained_model)):
            predicted = (trained_model.predict(x_test) > 0.5).astype("int32")
            predicted_prob = (trained_model.predict(x_test).flatten()).astype("float")

        assert self.check_data(predicted)
        assert self.check_data(predicted_prob)

        return predicted, predicted_prob

class model_selection:
    def __init__(self):
        self.model_inst = instantiate_model()
        self.model_fit = fit_model()
        self.model_metrics__ = model_metrics()

    def model_selection_pipeline(self, x_train, y_train, model_dict: dict ,num_splits: int= 8, **kwargs):
        
        metric_cols = ['Model','PR AUC','Brier Score','F1 Score','Recall','Precision']
        metric_list= []
    
        skf = StratifiedKFold(n_splits= num_splits, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(x_train, y_train):

            X_train_skf, X_test_skf= x_train.iloc[train_index],x_train.iloc[test_index]
            y_train_skf, y_test_skf = y_train.iloc[train_index],y_train.iloc[test_index]
            data_dict= {'x_train': X_train_skf, 'x_val':X_test_skf, 'y_train': y_train_skf, 'y_val':y_test_skf }

            for model_name, model in model_dict.items():

                '''train each model but for tf model, weights need to be reset before configure during each iteration of the split'''

                trained_model = self.model_fit.train_model(model, data_dict)

                model_to_pass= [trained_model if "sklearn" in "{}".format(type(trained_model)) else model][0]
                
                predicted,predicted_prob = self.model_fit.test_model(model_to_pass,X_test_skf)

                metric_list.append(
                    {
                        'Model': model_name,
                        'PR AUC': self.model_metrics__.classification_metrics(y_test_skf,predicted, predicted_prob)[2],
                        'Brier Score':self.model_metrics__.classification_metrics(y_test_skf,predicted, predicted_prob)[3],
                        'F1 Score': self.model_metrics__.classification_metrics(y_test_skf,predicted, predicted_prob)[4],
                        'Recall': self.model_metrics__.classification_metrics(y_test_skf,predicted,predicted_prob)[5],
                        'Precision': self.model_metrics__.classification_metrics(y_test_skf,predicted, predicted_prob)[6]
                    }
                )

        averages_df = pd.DataFrame(metric_list, columns = metric_cols).groupby('Model').mean()
        stdevs_df = pd.DataFrame(metric_list, columns=metric_cols).groupby('Model').std()
        return averages_df, stdevs_df


    def generate_focal_models(self, num_models:int=1, model_dict:dict = None):
        if model_dict is None: model_dict = dict()

        for i in range(0,num_models):
            m = self.model_inst.tfLogisticRegression()
            opt,alpha,gamma, configured_model = self.generate_focal_hyperparams(m)
            model_dict[f'Focal Loss Logistic Regression {i}'] = configured_model
            print(f"Model: {configured_model} \n Optimzer: {opt} \n Learning Rate: {opt.learning_rate} \n Alpha: {alpha} \n Gamma: {gamma}")
        return model_dict

    def generate_focal_hyperparams(self, model):
        learning_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
        optimizers = [keras.optimizers.Adam(random.choice(learning_rates)), keras.optimizers.Nadam(random.choice(learning_rates))]
        optimizer = random.choice(optimizers)
        alpha = random.choice([0.6, 0.65, 0.675, 0.7])
        gamma = random.choice([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
        model.compile(optimizer=optimizer, loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma))
        return optimizer, alpha,gamma, model

def train_test_split_func(x, y, test_size: int =0.15):
    X_train_tf, X_test_tf, y_train_tf, y_test_tf  = train_test_split(x, y,test_size=test_size, random_state=42)
    X_train_tf, X_val_tf, y_train_tf, y_val_tf  = train_test_split(X_train_tf, y_train_tf, test_size=0.1766, random_state=42)
    
    return X_train_tf, X_val_tf, X_test_tf, y_train_tf, y_val_tf, y_test_tf

def iterative_impute(X_train, X_test, X_val=None):
    iterative_imputer = IterativeImputer(estimator = BayesianRidge(),random_state=42,max_iter=50)
    imputed_train = pd.DataFrame(iterative_imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    imputed_test = pd.DataFrame(iterative_imputer.transform(X_test),columns=X_test.columns, index=X_test.index)

    if X_val is not None:
        imputed_val= pd.DataFrame(iterative_imputer.transform(X_val),columns=X_val.columns, index=X_val.index)
        return imputed_train, imputed_test, imputed_val
    return imputed_train, imputed_test

def scaler(train, train_index, test, test_index, val=None, val_index=None):
    scaler_tf = StandardScaler()
    scaled_train = pd.DataFrame(scaler_tf.fit_transform(train), index=train_index) 
    scaled_test = pd.DataFrame(scaler_tf.transform(test), index=test_index)

    joblib.dump(scaler_tf, '/Users/ryan/Desktop/ResearchRepo/HSCRPmlResearch/venv/scaler.save')

    if (val is not None) and (val_index is not None):
        scaled_val = pd.DataFrame(scaler_tf.transform(val), index=val_index)
        return scaled_train, scaled_test, scaled_val

    return scaled_train, scaled_test

def split_impute_scale_pipe(x, y, test_size:int =0.15):
    x_train, x_val, x_test, y_train, y_val, y_test = train_test_split_func(x, y, test_size)
    x_train_imp, x_test_imp, x_val_imp = iterative_impute(x_train, x_test, x_val)
    x_train_scaled, x_test_scaled, x_val_scaled = scaler(x_train_imp, y_train.index, x_test_imp, y_test.index, x_val_imp, y_val.index)
    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test
