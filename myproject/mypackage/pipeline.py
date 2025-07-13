import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, LassoCV,Ridge, LogisticRegression 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer # import SimpleImputer for missing value treatment
from sklearn.pipeline import Pipeline # importing pipeline class. The Pipeline class is used to create a sequence of data processing steps.
from sklearn.compose import ColumnTransformer # importing ColumnTransformer class to apply different preprocessing steps to different subsets of features in your dataset.
from scipy.stats import zscore
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor, export_graphviz
from sklearn.svm import SVC

prob_models = ['log_reg','nb','lin_reg','knn','tree'] #models based on probabilites
closed_form_models = ['nb','lin_reg','knn','tree','reg_tree'] #models that can't be iterated on

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

# Dictionary of all of the available scalers
scalers = {
    'std_scaler': StandardScaler(), 
    'min_max_scaler': MinMaxScaler()
}

models = {
    'ridge': Ridge(),
    'lasso': Lasso(),
    'log_reg': LogisticRegression(),
    'nb': GaussianNB(),
    'lin_reg': LinearRegression(),
    'knn': KNeighborsClassifier(),
    'tree': DecisionTreeClassifier(),
    'reg_tree': DecisionTreeRegressor(),
    'svc': SVC(probability=True)
}

def find_outliers_z(df, column, threshold=3):
    # cols = df.columns
    # for col in cols:
    z_scores = zscore(df[column].dropna())
    mask = np.abs(z_scores) > threshold
    median_val = df[column].median()
    return z_scores, mask, median_val

def validateParams(params, scalers, models):
        
        valid_strats = ['median']
        valid_scalers = list(scalers.keys())
        valid_models = list(models.keys())
        valid_scoring = ['neg_mean_absolute_error']

        
        
        ### Refactor list checks as a loop ###
        
        # Validate scaler name
        if params['scaler_name'] is not None and params['scaler_name'] not in valid_scalers:
            raise ValueError(f"Unknown scaler name: '{params['scaler_name']}'. Please use one of the following values: {valid_scalers}")
            
        # Validate model name
        if params['model_name'] not in valid_models:
            raise ValueError(f"Unknown scaler name: '{params['model_name']}'. Please use one of the following values: {valid_models}")
            
        # Validate scoring
        if params['scoring'] not in valid_scoring:
            raise ValueError(f"Unknown scaler name: '{params['scoring']}'. Please use one of the following values: {valid_scoring}")
            
        # Validate impute strategy
        if params['strategy'] not in valid_strats:
            raise ValueError(f"Unknown strategy name: '{params['strategy']}'. Please use one of the following values: {valid_strats}")
            
        
        
        ### Refactor int checks as a loop ###
        
        # Validate random state
        if params['random_state'] != None and not isinstance(params['random_state'],int):
            raise ValueError(f"The 'random_state' parameter '{params['random_state']}' must be an int. Got {params['random_state']} as a {type(params['random_state'])} instead.")
            
        # Validate cv
        if params['cv'] != None and not isinstance(params['cv'],int):
            raise ValueError(f"The 'cv' parameter '{params['cv']}' must be an int. Got {params['cv']} as a {type(params['cv'])} instead.")
            
        # Validate n_jobs
        if params['n_jobs'] != None and not isinstance(params['n_jobs'],int):
            raise ValueError(f"The 'n_jobs' parameter '{params['n_jobs']}' must be an int. Got {params['n_jobs']} as a {type(params['n_jobs'])} instead.")
           
        ### -------------- ###
        
        
        # Validate test size
        if isinstance(params['test_size'], float):
                if not 0.0 < params['test_size'] < 1.0:
                    raise ValueError(f"The 'test_size' parameter for 'runPipeline()' must be a float in the range (0.0, 1.0), an int in the range [1, inf) or None. Got {params['test_size']} as a {type(params['test_size'])} instead.")
        
        elif isinstance(params['test_size'], int):
                if params['test_size'] <= 1:
                    raise ValueError(f"The 'test_size' parameter for 'runPipeline()' must be a float in the range (0.0, 1.0), an int in the range [1, inf) or None. Got {params['test_size']} as a {type(params['test_size'])} instead.")
        
        elif isinstance(params['test_size'], str):
            raise ValueError(f"The 'test_size' parameter for 'runPipeline()' must be a float in the range (0.0, 1.0), an int in the range [1, inf) or None. Got {params['test_size']} as a {type(params['test_size'])} instead.")

def modelDictionary(key=None):
    valid_keys = ['models','scalers','prob_models','closed_form_models']

    if key in valid_keys:
        if key == 'models':
            return models.keys().tolist()
        elif key == 'scalers':
            return scalers.keys().tolist()
        elif key == 'prob_models':
            return prob_models
        elif key == 'closed_form_models':
            return closed_form_models
    elif key == None:
        return {
            'models': models.keys(),
            'scalers': scalers.keys(),
            'prob_models': prob_models,
            'closed_form_models': closed_form_models
        }
    else:
        raise ValueError(f"Unknown key: '{key}'. Please use one of the following values: {valid_keys}")

def runPipeline(X, y, showChart=True, parameters=None):
    
    # Remove all rows from X and y where y is null
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    multiclass = False # For charting purposes
    
    bundle = dict() # These will all of the return values
    
    default_params = {
            'test_size': 0.30, 
            'random_state': 2024,
            'max_iter': 1000,
            'strategy': 'median', 
            'scaler_name': None,
            'model_name': 'ridge',
            'grid': {
                'start': 0.1,
                'stop': 2.1,
                'step': 0.1
            },
            'scoring': 'neg_mean_absolute_error',
            'cv': 5,
            'n_jobs': -1,
            'drop_cols': None,
            'n_neighbors': 3,
            'ccp_alpha': 0,
            'class_weight': None,
            'max_depth': None,
            'min_samples_split': None
        }
    
    
     # Ensure all user given parameters are assigned to params
    if parameters == None:
        params = default_params
    else:
        params = {**default_params, **parameters} # Overrides all default params with user given params
        
    params['pipeline_results'] = {}
    

    if params['ccp_alpha'] == None:
        params.pop('ccp_alpha')
    
          
    validateParams(params, scalers, models)

    if params['scaler_name'] is not None:
        params['scaler'] = scalers[params['scaler_name']]
    
    params['model'] = models[params['model_name']]
    
    if params['drop_cols'] != None:
        X.drop(columns=params['drop_cols'], inplace=True)
    
    
    # Encode binary categorical values 
    label_encoder = LabelEncoder()

    
    if y.nunique() >= 2:
        if y.nunique() > 2: multiclass = True

        y = pd.Series(label_encoder.fit_transform(y), name=y.name)
        params['y_label_encoder'] = label_encoder
        y_cat_labels = label_encoder.classes_
    
    # Remove unnamed columns from the dataframe
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=params['random_state'])
    
    # Set up preprocessing steps for numeric and categorical data
    col_cat = X_train.select_dtypes(include=['object','category']).columns
    col_num = X_train.select_dtypes(include=[np.number]).columns
    
    
    # Numeric variables pipeline
    if params['scaler_name'] is not None:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=params['strategy'])),
            ('std_scaler', params['scaler'])
        ])
    else:
            num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=params['strategy'])) 
        ])
    
    # Unified preprocessing for both numeric and categorical data
    preprocessing = ColumnTransformer([
        ('num', num_pipeline, col_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), col_cat)
    ])
    
    # Build a Ridge regression model within a complete pipeline

    final_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        (f"{params['model_name']}", params['model'])
    ])
    
    # Define the grid of hyperparameters to search

    # Note: You can set parameters for any step by using its name followed by a double underscore(__) and the parameter name.
    
    grid = dict()

    if params['model_name'] not in ['knn','nb']:
        grid[f"{params['model_name']}__random_state"] = [params['random_state']]

    if params['max_depth'] is not None:
        grid[f"{params['model_name']}__max_depth"] = params['max_depth']
        
    if params['min_samples_split'] is not None:
        grid[f"{params['model_name']}__min_samples_split"] = params['min_samples_split']

    if  params['n_neighbors'] != None and params['model_name'] == 'knn':
        grid[f"{params['model_name']}__n_neighbors"] = [params['n_neighbors']]

    if params['class_weight'] != None:
        grid[f"{params['model_name']}__class_weight"] = [params['class_weight']]
    
    if params['model_name'] not in closed_form_models:
        grid[f"{params['model_name']}__max_iter"] = [params['max_iter']]

    
    # if params['model_name'] not in prob_models:
    if params['model_name'] in closed_form_models and params['model_name'] not in ['knn','nb']:
    
        start = params['grid']['start']
        stop = params['grid']['stop']
        step = params['grid']['step']

        if params['model_name'] in ['tree','reg_tree']:
            grid[f"{params['model_name']}__ccp_alpha"] = np.arange(start,stop,step)
        else:
            grid[f"{params['model_name']}__alpha"] = np.arange(start,stop,step)
    
    search = GridSearchCV(estimator = final_pipeline, param_grid = grid, scoring = params['scoring'],cv = params['cv'], n_jobs = params['n_jobs'])
    # Fit the GridSearchCV object to the training data

    # Fit GridSearchCV to the training data
    results = search.fit(X_train, y_train)
    print('MAE: %.3f' % -results.best_score_)
    print('Config: %s' % results.best_params_)
    
    # Get best model (pipeline with best hyperparameters)
    best_model = search.best_estimator_

    # R² score on training set
    train_score = best_model.score(X_train, y_train)

    # R² score on test set
    test_score = best_model.score(X_test, y_test)
    

    print(f"Train R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    # Predict on the test set using the trained model
    y_pred = search.predict(X)
    y_pred_test = search.predict(X_test)
    y_pred_train = search.predict(X_train)
    mae = mean_absolute_error(y_test, y_pred_test)
    pred_df = X.copy()

    if params['y_label_encoder'] is not None:
        y_pred_test = label_encoder.inverse_transform(y_pred_test)
        y_pred_train = label_encoder.inverse_transform(y_pred_train)
        y_pred = label_encoder.inverse_transform(y_pred)
        y_test = label_encoder.inverse_transform(y_test)
        y_train = label_encoder.inverse_transform(y_train)

    pred_df[f"pred_{y.name}"] = y_pred
    print("Mean Absolute Error:", mae)
    print("---------------------------")
    
    if params['model_name'] in prob_models:
        
        y_pred_test_prob = results.predict_proba(X_test)[:, 1]
        y_pred_train_prob = results.predict_proba(X_train)[:, 1]
        
        chart_params = {
            'y_test': y_test,
            'y_train': y_train,
            'y_pred_test_prob': y_pred_test_prob,
            'y_pred_train_prob': y_pred_train_prob,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'y_pred': y_pred,
            'y_cat_labels': y_cat_labels,
            'X_test': X_test,
            'X_train': X_train,
            'results': results,
            'model_name': params['model_name'],
            'model': best_model
        }

        if showChart:
            chartEvals(chart_params, multiclass=multiclass)
    
    params['pipeline_results']['mae'] = mae
    params['pipeline_results']['mae_best_score'] = results.best_score_
    params['pipeline_results']['mae_best_params'] = results.best_params_
    params['pipeline_results']['y_pred'] = y_pred_test
    params['pipeline_results']['pred_df'] = pred_df
    params['pipeline_results']['best_model'] = best_model
    params['pipeline_results']['test_score'] = test_score
    params['pipeline_results']['train_score'] = train_score
    params['data'] = {}
    params['data']['X_train'] = X_train
    params['data']['X_test'] = X_test
    params['data']['y_train'] = y_train
    params['data']['y_test'] = y_test
    

    return params

def evaluateModel(trainData,testData,modelObject):
    pass

def predict(params, threshold=None):
    
    X_train = params['X_train']
    X_test = params['X_test']
    y = params['y']
    y_pred_test = params['y_pred_test']
    y_pred_train = params['y_pred_train']
    model = params['results']
    y_probs = params['y_probs']
    
    
    y_pred_custom = (y_probs > threshold).astype(int)
    
    
    # Predict on the test set using the trained model
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    pred_df = X_test.copy()
    pred_df[f"pred_{y.name}"] = y_pred_test

def chartEvals(chartParams, multiclass=False):
    bundle = dict()
    y_train = chartParams['y_train']
    y_pred_train = chartParams['y_pred_train']
    y_test = chartParams['y_test']
    y_pred_test = chartParams['y_pred_test']
    y_pred_test_prob = chartParams['y_pred_test_prob']
    y_cat_labels = chartParams['y_cat_labels']
    model_name = chartParams['model_name']
    X_train = chartParams['X_train']
    model = chartParams['model']

    # Have to get the column names after preprocessing
    preprocessor = chartParams['model'].named_steps['preprocessing']
    onehot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out()

    feature_names = np.concatenate([
        chartParams['X_train'].select_dtypes(include=[np.number]).columns,
        onehot_cols
    ])


    
    # Predict probabilities on the testing set
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    bundle['pred_prob'] = pd.DataFrame({
        'y_pred': y_pred_test,
        'y_actual': y_test,
        'y_pred_test_prob': y_pred_test_prob
    })

    bundle['conf_matrix_test'] = conf_matrix
    bundle['conf_matrix_train'] = conf_matrix_train

    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels=y_cat_labels)
    cm_display.plot()
    plt.show()

    # Print classification report
    print("\t\tLogistic Regression Classification Report:")
    print("\t\t-----------------------------------------")
    print(classification_report(y_test, y_pred_test))

    if model_name == 'tree':
        
        tree_model = model.named_steps['tree']
        plt.figure(figsize=(14,20))
        export_graphviz(tree_model,"ship_model.dot",feature_names=feature_names, rounded=True, class_names=['No','Yes'], filled=True)
        plt.show()

    if not multiclass:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_prob)
        
        # Compute the Area Under the Curve (AUC) for the ROC curve
        roc_auc = auc(fpr, tpr)

        # Compute Youden's J statistic for each threshold
        youden_j = tpr - fpr
        optimal_threshold_index = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_threshold_index]

        print(f"Optimal Threshold: {optimal_threshold:.4f}")

        # Plot the ROC curve with the optimal threshold marked
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', marker='o', label=f'Optimal Threshold = {optimal_threshold:.4f}')
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    return conf_matrix

def boxplotGrid(df):
    number_cols = df.select_dtypes(exclude=['object']).dropna(axis=1)
    total_num_cols = number_cols.shape[1]
    plt_cols = 3
    plt_rows = (total_num_cols % plt_cols) + total_num_cols // plt_cols
    width = 10
    height = 8
    axes_list = []

    fig, axes = plt.subplots(plt_cols, plt_rows, figsize=(width, height))
    plt.tight_layout(pad=2.0)
    
    for i in range(plt_cols):
        for j in range(plt_rows):

            axes_list.append(axes[i, j])
            
    for i, ax in enumerate(axes_list[total_num_cols:]):
         fig.delaxes(ax)
    #axes_list = axes_list[:total_num_cols]
        # axes[0, 0].boxplot(data)
        # axes[0, 0].set_title(title) 
    for i, col in enumerate(number_cols):
        title = df[col].name
        data = df[col]
        axes_list[i].boxplot(data)
        axes_list[i].set_title(title)
    
    plt.show()