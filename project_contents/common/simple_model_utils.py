import streamlit as st
from streamlit_tree_select import tree_select

import numpy as np
import pandas as pd

import ppscore as pps
import lightgbm.sklearn as lgbm
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score

import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import kstest
from .chart_utils import get_feature_distribution, get_report_2d_on_target, REPORT_HEIGHT
from .process_utils import free_port, run_seperate_command
from .di_config import EXPLAINER_PORT, EXPLAINER_URL

import streamlit.components.v1 as components
from zeda2.describe import *
from zeda2.common_report.block import *

import shap
import re
import unidecode

from explainerdashboard import RegressionExplainer, ExplainerDashboard

STRONG_PREDICTOR_THRESHOLD = 0.4
STATISTICAL_ALPHA = 0.1

def beautify_model_summary(summary):
    def accepted_p_value(v):
        return 'background-color:green;color:white' if v < 0.1 else None
    p_value_column = [c for c in summary.tables[1].columns if 'P>' in c][0]
    summary_metrics = pd.concat([
        summary.tables[0].iloc[:, :2].rename(columns={0: 'Metric', 1: 'Value'}),
        summary.tables[0].iloc[:, 2:].rename(columns={2: 'Metric', 3: 'Value'})
    ], axis=0).set_index('Metric').T

    new_summary = [
        summary_metrics,
        summary.tables[1].style.applymap(accepted_p_value, subset=p_value_column)
    ]
    return new_summary

def detect_potential_keys(df):
    all_columns = df.columns.tolist()
    df_nodup = df.drop_duplicates()

    idx = 1
    while True:
        keys = all_columns[:idx]
        df_with_keys = df.drop_duplicates(subset=keys)
        if df_with_keys.shape[0] == df_nodup.shape[0]:
            return keys
        idx += 1

def select_features(df:pd.DataFrame, target_feature:str, nb_features:int=20, nb_categories:int=10):
    """Return a dataframe with target_feature and top 20 features most important"""
    missing_features = df.isnull().mean()
    missing_much_features = missing_features[missing_features > 0.8].index.tolist()
    df = df[[c for c in df.columns if c not in missing_much_features or c == target_feature]]

    ## transform category features
    object_features = [c for c in df.columns if df[c].dtypes.name == 'object' and c not in missing_much_features]
    count_categories = df[object_features].nunique()
    multiple_categories = count_categories[count_categories > nb_categories].index.tolist()
    for category in multiple_categories:
        category_count = df[category].value_counts().sort_values(ascending=False).index.tolist()
        replace_categories = category_count[nb_categories:]
        df.loc[:, category] = df[category].map(lambda x: x if x not in replace_categories else 'Others')

    predictors = pps.predictors(df, target_feature)
    nb_features = min(df.shape[1], nb_features)
    selected_features = list(predictors[predictors['ppscore'] <= STRONG_PREDICTOR_THRESHOLD].reset_index(drop=True).loc[:nb_features, 'x'].values)

    return {
        'strong_predictors': predictors[predictors['ppscore'] > STRONG_PREDICTOR_THRESHOLD]['x'].tolist(),
        'missing_features': missing_much_features,
        'selected_features': selected_features
    }

def variance_threshold_selector(df, threshold=0):
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    return df[df.columns[selector.get_support(indices=True)]]
def lgbm_model(df:pd.DataFrame, target_feature:str, is_explain:bool=True, max_sample_to_explain:int=1000, max_sample=10000, predictors:list=[], ignore_features:list=[], keys:list=[], nb_features:int=50, nb_categories:int=10):
    if target_feature in df.columns:
        p_label = target_feature
    else:
        raise ValueError(f"Your column name '{target_feature}' does not exist in the dataframe. Available columns are: {df.columns.tolist()}")

    if df.shape[1] == 1:
        raise ValueError(f"Cannot build model with the data has only 1 column.")
    if df.shape[0] < 2:
        raise ValueError(f'Cannot build model with the data that has less than 2 samples. Current rows: {df.shape[0]}.')

    predictors = list(set(predictors))
    ignore_features = list(set(ignore_features))

    if df.shape[0] > max_sample_to_explain:
        is_explain = False

    if df.shape[0] > max_sample:
        df = df.sample(n=max_sample, random_state=212)

    if len(keys) == 0:
        keys = detect_potential_keys(df)

    ignore_features += keys
    strong_predictors, missing_features, selected_features = select_features(df, target_feature=target_feature, nb_features=nb_features, nb_categories=nb_categories).values()

    ignore_features = list(set(ignore_features + strong_predictors + missing_features))
    if len(predictors) == 0:
        p_feature = [c for c in selected_features if c not in ignore_features + [p_label]]
    else:
        p_feature = [c for c in predictors.copy() if c not in [p_label] + keys]

    full_data = df.copy()
    USING_REGRESSION = full_data[p_label].dtypes.name != "object" and full_data[p_label].nunique() > 5
    print("IS USING REGRESSION:", USING_REGRESSION)
    full_data = full_data[full_data[p_label].notnull()][p_feature + [p_label]]
    print(full_data[[p_label]].describe())

    if not USING_REGRESSION:
        label_count = full_data[p_label].value_counts(normalize=True) * 100
        to_filter = label_count[label_count < 1].index.tolist()
        full_data = full_data[~full_data[p_label].isin(to_filter)]

    list_category = []
    list_numeric = []
    for x, y in zip(full_data.dtypes, full_data.columns):
        if x == "object" and y != p_label:
            list_category.append(y)
        else:
            list_numeric.append(y)

    for c in list_category:
        full_data[c] = full_data[c].fillna("NaN").astype("category")

    random_state = 212

    if not USING_REGRESSION:
        le = preprocessing.LabelEncoder()
        le.fit(full_data[p_label])
        full_data[p_label] = full_data.apply(lambda x: le.transform([x[p_label]])[0], axis=1)
        full_data[f'{p_label}_str'] = full_data[p_label].map(lambda x: le.classes_[x])

    if is_explain:
        # no need to split train test
        X_train = X_valid = full_data[p_feature]
        y_train = y_valid = full_data[p_label]

        print("EXPLAIN MODE")
        print(y_valid.nunique())

    else:
        if USING_REGRESSION:
            X_train, X_valid, y_train, y_valid = train_test_split(full_data[p_feature],
                                                                  full_data[p_label],
                                                                  test_size=0.2,
                                                                  random_state=random_state)
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(full_data[p_feature],
                                                                  full_data[p_label],
                                                                  test_size=0.2,
                                                                  random_state=random_state,
                                                                  stratify=full_data[p_label])

    # rename due to the error not support JSON characters in LGBM
    column_name_dict = {
        re.sub('[^A-Za-z0-9_]+', '', unidecode.unidecode(x.replace(" ", "_"))): x for x in X_train.columns
    }
    X_train = X_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', unidecode.unidecode(x.replace(" ", "_"))))
    X_valid = X_valid.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', unidecode.unidecode(x.replace(" ", "_"))))

    ## Setting params ##
    metric_score = {}
    if USING_REGRESSION:
        lgb_params = {
            'boosting': 'gbdt',
            'objective': 'regression',
            'metric': ['rmse'],
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': 20,  # -1
            'feature_fraction': 0.4,
            'bagging_fraction': 0.8,
            'bagging_freq': 8,
            'min_data_in_leaf': 5,
            'lambda_l1': 5.020328536795886e-05,
            'lambda_l2': 5.03166477561831e-06
        }
        lgb_model = lgbm.LGBMRegressor(**lgb_params)
        print("TRAINING DONE")
        lgb_model.fit(X_train, y_train, categorical_feature=list_category)
        y_pred_val = lgb_model.predict(X_valid)

        metric_score['MAE'] = mean_absolute_error(y_valid, y_pred_val)
        metric_score['RMSE'] = mean_squared_error(y_valid, y_pred_val, squared=False)

    else:
        lgb_params = {
            "boosting": "gbdt",
            "objective": "multiclass" if len(le.classes_) > 2 else "binary",
            "metric": ["auc_mu"] if len(le.classes_) > 2 else "auc",
            "num_leaves": 5,
            "learning_rate": 0.005,
            "max_depth": 10,
            "feature_fraction": 0.4,
            "bagging_fraction": 0.8,
            "bagging_freq": 8,
            "min_data_in_leaf": 15,
            "lambda_l1": 5.020328536795886e-05,
            "lambda_l2": 5.03166477561831e-06,
            "num_classes": len(le.classes_) if len(le.classes_) > 2 else 1,
        }

        lgb_model = lgbm.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train, categorical_feature=list_category)
        y_pred_val = lgb_model.predict_proba(X_valid)

        if len(le.classes_) > 2:
            metric_score["ROC AUC"] = roc_auc_score(y_valid, y_pred_val, multi_class='ovr') * 100
        else:
            metric_score["ROC AUC"] = roc_auc_score(y_valid, y_pred_val[:, 1]) * 100

    print("TRAINING DONE")
    if X_valid.shape[0] > 5000:
        X_sample = X_valid.sample(5000)
    else:
        X_sample = X_valid.copy()

    explainer = shap.Explainer(lgb_model)
    shap_values = explainer.shap_values(X_sample)
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_sample.columns, vals)),
                                      columns=["col_name", "feature_importance_vals"])
    feature_importance.sort_values(by=["feature_importance_vals"], ascending=False, inplace=True)

    # reverse name
    feature_importance["col_name"] = feature_importance["col_name"].map(lambda x: column_name_dict[x])

    # Explainer Dashboard
    if USING_REGRESSION:
        try:
            with st.spinner('Generating Explainer Dashboard'):
                db_explainer = RegressionExplainer(lgb_model, X_valid, y_valid, shap='tree')
                db_dashboard = ExplainerDashboard(db_explainer, port=5000, url_base_pathname='/explainer/')
                db_explainer.dump("explainer.joblib")
                db_dashboard.to_yaml("explainer.yaml", explainerfile="explainer.joblib")
        except:
            st.warning("Failed to create explainer dashboard.")

    print("BEFORE RETURN RESULT:")
    print(df[p_label].dtypes)
    return {
        'df_full': df.copy(),
        'factors': p_feature,
        'target_feature': p_label,
        'target_distribution': full_data[p_label], # this is for target distribution while training
        'performance': metric_score,
        'feature_importance': feature_importance,
        'shap_values': shap_values,
        'df': X_sample,
        'classes': le.classes_ if not USING_REGRESSION else None,
        'model': lgb_model,
        'strong_predictors': [c for c in strong_predictors if c not in keys + predictors],
        'model_function': 'LGBM',
        'is_explain': is_explain
    }

@st.cache_data(ttl=60 * 60)
def statistical_model(df:pd.DataFrame, target_feature:str, predictors:list=[], nb_features:int=50, keys:list=[], ignore_features:list=[], nb_categories:int=10, max_sample:int=1000):
    # detect if is there any column having strong correlction with target_feature
    if len(keys) == 0:
        keys = detect_potential_keys(df)

    ignore_features = [c for c in keys + ignore_features if c != target_feature]
    df_model = df.copy()
    if df_model.shape[0] > max_sample:
        df_model = df_model.sample(n=max_sample, random_state=212)

    is_numeric = df_model[target_feature].dtypes.name != 'object' and df_model[target_feature].nunique() > 5
    models = []
    # remove group having less than 0.01% members
    if not is_numeric:
        percent_count_members = 100*df_model[target_feature].value_counts(normalize=True)
        small_members = percent_count_members[percent_count_members < 0.01].index.tolist()
        df_model = df_model[~df_model[target_feature].isin(small_members)]

    strong_predictors, missing_features, selected_features = select_features(df_model, target_feature=target_feature,
                                                                             nb_features=nb_features,
                                                                             nb_categories=nb_categories).values()

    if len(predictors) == 0: # no particular factors
        selected_features = [c for c in selected_features if c not in ignore_features + [target_feature]]
        for col in selected_features:
            if df_model[col].dtypes.name == 'object':
                y = pd.get_dummies(df_model[col], prefix=col)
                df_model = pd.concat([df_model.drop(col, axis=1), y], axis=1)
                predictors += y.columns.tolist()
            else:
                predictors += [col]
        final_selected_features = [c for c in predictors if c != target_feature]
    else:
        final_selected_features = [c for c in predictors if c != target_feature]

    df_notna = df_model.copy()
    for feature in [target_feature] + final_selected_features:
        df_notna[feature] = df_notna[feature].fillna(df_notna[feature].mean())

    if df_notna.shape[0] == 0:
        return {
            'df_full': df,
            'df': df_model,
            'target_feature': target_feature,
            'factors': final_selected_features,
            'strong_predictors': [c for c in strong_predictors if c not in keys + predictors],
            'model': None, 'assumption_verify': None,
            'model_function': 'statistical'
        }
    #     df_notna = df_model[[target_feature] + final_selected_features].fillna(0.0001)

    X = df_notna[final_selected_features]
    y = df_notna[target_feature]
    X = sm.add_constant(X)
    X = variance_threshold_selector(X)

    # fit linear regression model
    metric_score = {}
    if is_numeric:
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)
        metric_score['MAE'] = mean_absolute_error(y, y_pred)
        metric_score['RMSE'] = mean_squared_error(y, y_pred, squared=False)

        resid = model.resid
        shapiro_test = shapiro(resid)
        kolmo = kstest(resid, 'norm')
        resid_test_df = pd.DataFrame([(shapiro_test.statistic, shapiro_test.pvalue),
                                      (kolmo.statistic, kolmo.pvalue)], columns=['Statistic Value', 'P Value'])
        resid_test_df.index = ['Shapiro Test', 'KS Test']
        _, hist = describe_1d_numeric_with_plot(pd.DataFrame(resid, columns=['Residual']), 'Residual', return_raw=True)
        hist.update_layout(height=450)

        return {
            'df_full': df,
            'df': df_notna,
            'target_feature': target_feature,
            'factors': final_selected_features,
            'performance': metric_score,
            'strong_predictors': [c for c in strong_predictors if c not in keys + predictors],
            'model': model, 'assumption_verify': (resid_test_df, hist, sm.qqplot(resid)),
            'model_function': 'statistical'
        }

    # fit logistic regression model
    else:
        y_target = pd.get_dummies(df_notna[target_feature], prefix=target_feature)
        if len(y_target.columns) > 2:
            for target_cate in y_target.columns:
                model = sm.Logit(y_target[target_cate], X).fit()
                models.append(model)
        else:
            model = sm.Logit(y_target.iloc[:, 0], X).fit()
            models.append(model)
        return {
            'df_full': df,
            'df': df_notna,
            'target_feature': target_feature,
            'factors': final_selected_features,
            'performance': None,
            'strong_predictors': [c for c in strong_predictors if c not in keys + predictors],
            'model': models,
            'model_function': 'statistical'
        }