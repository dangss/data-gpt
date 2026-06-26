import pandas as pd
import streamlit as st
from .chart_utils import *
from .simple_model_utils import *
from .app_utils import *
import re
from traceback import format_exc
from streamlit_tree_select import tree_select

## COMMON PATH ##
COMMON_PATH = f"{os.path.dirname(__file__)}"
def create_column_tree(column_names, prefix):
    tree = {}
    for name in column_names:
        parts = name.split('_')
        if len(parts) >= 2:
            category = parts[0]
            subcategory = parts[1]
            if category not in tree:
                tree[category] = {}
            if subcategory not in tree[category]:
                tree[category][subcategory] = []
            tree[category][subcategory].append({"value": name, "label": name})
        else:
            if parts[0] not in tree:
                tree[parts[0]] = dict()
            if parts[0] not in tree[parts[0]]:
                tree[parts[0]][parts[0]] = []
            tree[parts[0]][parts[0]].append({"value": name, "label": name})

    tree_structure = [{
        "value": f"{prefix}|{category_name}",
        "label": f"{category_name}'s feature group",
        "children": [
            {
                "value": f"{prefix}|{category_name}|{sub_category}",
                "label": f"{category_name}_{sub_category}\'s feature group",
                "children": children
            } if len(children) > 1 else children[0] for sub_category, children in category_value.items()
        ]
    } for category_name, category_value in tree.items()]
    return tree_structure

def change_model_features(result:dict):
    def rerun_modeling():
        model_functions = {
            "LGBM": lgbm_model,
            "statistical": statistical_model
        }
        data = result["LGBM"]["df_full"]
        factors = [factor for factor in st.session_state.factors["checked"] if factor in data.columns]
        full_model_result = {}
        for model_name, model_result in result.items():
            if model_name == "LGBM":
                if "is_explain" not in result[model_name].keys():
                    model_result = model_functions[result[model_name]["model_function"]](df=data, target_feature=result[model_name]["target_feature"], predictors=factors)
                else:
                    model_result = model_functions[result[model_name]["model_function"]](df=data, target_feature=result[model_name]["target_feature"],
                                                                                     predictors=factors, is_explain=result[model_name]["is_explain"])
            elif model_name == "statistical":
                model_result = model_functions[result[model_name]["model_function"]](df=data, target_feature=result[model_name]["target_feature"], predictors=factors)
            full_model_result[model_name] = copy.deepcopy((model_result))
        return full_model_result

    feature_form = st.form("Independent Variables")
    with feature_form:
        st.write("List of factors")
        st.session_state.factors = {"checked": result["LGBM"]["factors"]}
        factor_trees = create_column_tree(result["LGBM"]["factors"], prefix="used")
        high_correlation_tree = create_column_tree(result["LGBM"]["strong_predictors"], prefix="high_correlated")
        tree_select([
            {
                "value": "current_used",
                "label": f"Current Used Features ({len(result['LGBM']['factors'])} features)",
                "children": factor_trees
            },
            {
                "value": "high_correlated_features",
                "label": f"High Correlated Features (pps > {STRONG_PREDICTOR_THRESHOLD}) ({len(result['LGBM']['strong_predictors'])} features)",
                "children": high_correlation_tree
            }
        ], checked=result["LGBM"]["factors"], key="factors")
        st.form_submit_button("Change features", on_click=rerun_modeling)
    return feature_form

@st.experimental_fragment
def show_lgbm_result(result:dict):
    st.session_state["lgbm_result"] = copy.deepcopy(result)

    with st.form("Independent Variables"):
        lgbm_result = st.session_state["lgbm_result"]
        st.write("List of factors")
        factor_trees = create_column_tree(lgbm_result["factors"], prefix="used")
        high_correlation_tree = create_column_tree(lgbm_result["strong_predictors"], prefix="high_correlated")
        tree_select([
            {
                "value": "current_used",
                "label": f"Current Used Features ({len(lgbm_result['factors'])} features)",
                "children": factor_trees
            },
            {
                "value": "high_correlated_features",
                "label": f"High Correlated Features (pps > {STRONG_PREDICTOR_THRESHOLD}) ({len(lgbm_result['strong_predictors'])} features)",
                "children": high_correlation_tree
            }
        ], checked=lgbm_result["factors"], key="factors")
        change_submit = st.form_submit_button("Change features")

    if change_submit:
        lgbm_result = st.session_state["lgbm_result"]
        factors = [factor for factor in st.session_state.factors["checked"] if factor in lgbm_result["df_full"].columns]
        st.session_state["lgbm_result"] = lgbm_model(df=lgbm_result["df_full"], target_feature=lgbm_result["target_feature"], predictors=factors, is_explain=lgbm_result["is_explain"])

    result = st.session_state["lgbm_result"]
    shap_values = result["shap_values"]
    df = result["df"]
    classes = result["classes"]
    target_feature = result["target_feature"]
    feature_importance = result["feature_importance"]

    baseline_dict = dict()
    target_distribution = result["target_distribution"].dropna()
    if classes is None: # regression
        st.caption("Baseline Performance (compared to Average Value of target feature)")
        # Calculate the mean of the target feature
        mean_target = np.mean(target_distribution)
        # Calculate RMSE and MAE of the mean
        baseline_dict["MAE"] = mean_absolute_error(target_distribution, [mean_target] * len(target_distribution))
        baseline_dict["RMSE"] = mean_squared_error(target_distribution, [mean_target] * len(target_distribution), squared=False)
    else: # classification
        most_common_class = int(target_distribution.mode().iloc[0])
        st.caption(f"Baseline Performance (compared to Most Common Value of target feature: {classes[most_common_class]}")
        if len(classes) > 2:
            baseline_dict["ROC AUC"] = roc_auc_score(target_distribution, [most_common_class]*len(target_distribution), multi_class="ovr") * 100
        else:
            baseline_dict["ROC AUC"] = roc_auc_score(target_distribution, [most_common_class]*len(target_distribution)) * 100

    st.subheader("Model Performance")
    performance_cols = st.columns(len(result["performance"].keys()))
    round_number = 5
    for idx, metric in enumerate(result["performance"].keys()):
        with performance_cols[idx]:
            is_higher_better = int(metric == "ROC AUC")
            changed_score = (100*is_higher_better*(result["performance"][metric] - baseline_dict[metric]) / baseline_dict[metric]).round(round_number)
            st.metric(metric, value = "{:,.3f}".format(result["performance"][metric].round(round_number)), delta="{:,.3f} %".format(changed_score))

    baseline_cols = st.columns(len(baseline_dict.keys()))
    for idx, metric in enumerate(baseline_dict.keys()):
        with baseline_cols[idx]:
            st.metric(metric, value="{:,.3f}".format(baseline_dict[metric].round(round_number)))

    st.subheader("Label Distribution")
    feature_dist = get_feature_distribution(result["df_full"], col_name=target_feature)
    components.html(feature_dist, scrolling=True, width=None, height=650)

    max_features_to_display = min(30, len(df.columns))
    st.subheader(f"Top {max_features_to_display} features that contribute the most")

    # If shap values is a list, it contains list of shape values for each category
    IS_CLASSIFIER = isinstance(shap_values, list)

    shap.initjs()
    plot_size = (12, 10)
    shap_tabs = st.tabs(("Feature Importance Overview", "Shap Values Distribution", "Feature Correlation"))

    predictors = [c for c in df.columns if c != target_feature]
    with shap_tabs[0]:
        st.write("Feature Importance Overview")
        if IS_CLASSIFIER:
            shap.summary_plot(shap_values, df[predictors], max_display=max_features_to_display,
                              plot_type="bar", class_names=classes, plot_size=plot_size)
        else:
            shap.summary_plot(shap_values, df[predictors], max_display=max_features_to_display,
                              plot_type="bar", plot_size=plot_size)

        st.pyplot(bbox_inches="tight",dpi=300,pad_inches=0)

    with shap_tabs[1]:
        st.markdown(f"How each feature contribute to predict <b>{target_feature}</b>", unsafe_allow_html=True)

        if IS_CLASSIFIER:
            category_tabs = st.tabs((le.classes_.tolist()))
            for idx, tab in enumerate(category_tabs):
                with tab:
                    st.markdown(f"Shap Value on predict <b>{le.classes_[idx]}</b>", unsafe_allow_html=True)
                shap.summary_plot(shap_values[idx], df[predictors], max_display=max_features_to_display, plot_size=plot_size)
                st.pyplot(bbox_inches="tight", dpi=300, pad_inches=0)
        else:
            shap.summary_plot(shap_values, df[predictors], max_display=max_features_to_display, plot_size=plot_size)
            st.pyplot(bbox_inches="tight", dpi=300, pad_inches=0)

    if classes is None: # USING REGRESSION
        try:
            free_port(EXPLAINER_PORT)
            explainer_command = f"explainerdashboard run explainer.yaml --no-browser"
            run_seperate_command(explainer_command)

            st.link_button("Click here to view Explainer Dashboard", url=EXPLAINER_URL)

        except Exception as err:
            print(err)
            pass

    with shap_tabs[2]:
        pred_features = feature_importance["col_name"].tolist()
        with st.spinner("Generating 2D Visualization"):
            visualization_2d = get_report_2d_on_target(df=result["df_full"], target_feature=target_feature, predictors=pred_features, title=f"2D Visualization on {target_feature}", return_html=True)
        components.html(visualization_2d, width=None, height=1000, scrolling=True)

def show_statistical_result(result):
    models = result["model"]
    predictors = result["factors"]
    target_feature = result["target_feature"]

    if models is not None:
        # view model summary
        if result["performance"] is not None:
            st.subheader("Model Performance")
            performance_cols = st.columns(len(result["performance"].keys()))
            for idx, metric in enumerate(result["performance"].keys()):
                with performance_cols[idx]:
                    st.metric(metric, value="{:,}".format(result["performance"][metric].round(2)))

        st.subheader("Model summary")
        if isinstance(models, list): # multiple models (for each category)
            model_names = [model.model.endog_names for model in models]
            if len(models) > 1:
                for idx, model in enumerate(models):
                    with st.expander(model_names[idx], expanded=(idx == 0)):
                        for table in beautify_model_summary(model.summary2(alpha=STATISTICAL_ALPHA)):
                            st.dataframe(table, use_container_width=True)
        else: # 1 model
            for table in beautify_model_summary(models.summary2()):
                st.dataframe(table, use_container_width=True)

        # view Feature Correlation
        st.subheader(f"How each factor correlates to the TARGET")
        df = result["df"]
        df_factors = pps.predictors(df[[target_feature] + predictors], target_feature)[["x", "ppscore"]].rename(columns={"x": "Factor"})

        if df[target_feature].dtypes.name != "object":
            df_corr = df[[target_feature] + predictors].corr()[[target_feature]].reset_index().rename(columns={"index": "Factor", target_feature: "correlation"})
            df_factors = pd.merge(df_factors, df_corr, on="Factor")
        st.dataframe(df_factors.set_index("Factor"), use_container_width=True)

        if "assumption_verify" in result.keys():
            st.subheader("Assumption Verify")
            st.dataframe(result["assumption_verify"][0], use_container_width=True)
            assumption_columns = st.columns(2)
            with assumption_columns[0]:
                st.plotly_chart(result["assumption_verify"][1], use_container_width=True)
            with assumption_columns[1]:
                st.caption("Q-Q Plot of residuals")
                st.write(result["assumption_verify"][2])
        # with model_tabs[1]:
        #     with st.spinner('Generating 2D Visualization'):
        #         visualization_2d = get_report_2d_on_target(
        #             dataset=result['df_full'],
        #             target_feature=target_feature,
        #             predictors=predictors)
        #     components.html(visualization_2d, width=None, height=REPORT_HEIGHT, scrolling=True)

    else:
        st.warning("There is no data row having all values to build a statistical model. We will show the 2D-visualization of TARGET and each factor instead.")
        with st.spinner("Generating 2D Visualization"):
            visualization_2d = get_report_2d_on_target(df=result["df_full"], target_feature=target_feature, predictors=predictors, return_html=True)
        components.html(visualization_2d, width=None, height=REPORT_HEIGHT, scrolling=True)

def show_2d_result(result:dict):
    target_feature = result["target"]
    with st.spinner("Generating 2D Visualization"):
        visualization_2d = get_report_2d_on_target(df=result["data"], target_feature=target_feature,
                                                   title=f"2D Visualization on {target_feature}", return_html=True)
    components.html(visualization_2d, width=None, height=1000, scrolling=True)

def show_custom_result(result: dict):
    for values in result.values():
        st.plotly_chart(values)

def show_report(result: dict):
    data_html = result["data_html"]
    if data_html is not None:
        report_height = 0
        for title, section_height in [("<h1>A Dataset</h1>", 650), ("<h1>B Data Overview</h1>", 650),
                                      ("<h1>C 1D Visualization</h1>", 800),
                                      ("<h1>D 2D Visualization</h1>", 800), ("<h1>E 3D Visualization</h1>", 800)]:
            if title in data_html:
                report_height += section_height
    else:
        report_height=3000

    dragndrop_html = result["dragndrop_html"]

    if dragndrop_html is not None and data_html is not None:
        report_tabs = st.tabs(("AutoEDA", "Data Visualization"))
        with report_tabs[0]:
            components.html(data_html, width=None, height=report_height, scrolling=True)
        with report_tabs[1]:
            components.html(dragndrop_html, width=None, height=report_height, scrolling=True)
    elif data_html is not None:
        components.html(data_html, width=None, height=report_height, scrolling=True)
    elif dragndrop_html is not None:
        components.html(dragndrop_html, width=None, height=report_height, scrolling=True)
    else:
        print("Report has no component.")
        return False
    return True

# @st.cache_data(ttl=60 * 60, show_spinner=False)
def show_model(result: dict):
    with st.spinner("Showing Result"):
        if "model_function" in result.keys():
            STREAMLIT_MODEL_FUNCTIONS[result["model_function"]](result)
        else:
            st.write("This is custom model result from dynamic_model.")

def show_models(model_list:dict):
    model_tabs = st.tabs([x.upper() + " model" for x in model_list.keys()])
    for idx, model_name in enumerate(model_list.keys()):
        with model_tabs[idx]:
            show_model(model_list[model_name])

def show_tables(tables:dict=dict()):
    if len(tables.keys()) == 0:
        st.caption("The question does not involve FACTOR information.")
    else:
        with st.container(border=True):
            st.write("Selected Tables:")
            for table in tables.keys():
                st.link_button(table, url="/Metadata", help="Click here to view metadata of this table")
                with st.expander("Selected columns"):
                    st.write(tables[table])


STREAMLIT_VIZ_FUNCTIONS = {
    "custom_chart": show_custom_result,
    "report": show_report
}

STREAMLIT_MODEL_FUNCTIONS = {
    "LGBM": show_lgbm_result,
    "statistical": show_statistical_result,
    "only_2d": show_2d_result
    # 'custom': show_custom_result
}

@st.cache_data(ttl=60 * 60, show_spinner=False)
def show_viz(result: dict):
    if "viz_function" not in result.keys():
        STREAMLIT_VIZ_FUNCTIONS["custom_chart"](result)
    else:
        STREAMLIT_VIZ_FUNCTIONS[result["viz_function"]](result)
@st.experimental_dialog("Data Report", width="large")
def gen_data_report(data:pd.DataFrame, report_start=3, gen_report=True):
    show_dragndrop_option = True
    if isinstance(data, pd.DataFrame) and data.shape[0] < 1000:
        try:
            if not gen_report:
                data_html = convert_data_to_html(data)
                components.html(data_html, width=None, height=350, scrolling=True)
            else:
                if data.shape[0] <= report_start:
                    data_html = convert_data_to_html(data)
                    components.html(data_html, width=None, height=350, scrolling=True)
                elif gen_report and data.shape[1] <= 20: # less than 20 columns
                    print("GEN REPORT [OPTION FULL CHART]")
                    general_report = get_report(df=data, show_1d=True, show_2d=True, show_3d=True, show_dragndrop=show_dragndrop_option)
                    show_viz(general_report)
                else:
                    print("GEN REPORT [OPTION 1D ONLY]")
                    general_report = get_report(df=data, show_1d=True, show_2d=False, show_3d=False, show_dragndrop=False)
                    show_viz(general_report)
        except Exception as err:
            print("Error while generating autoEDA")
            error = format_exc()
            print(error)
            data_html = convert_data_to_html(data)
            components.html(data_html, width=None, height=800, scrolling=True)
    else:
        if data is None:
            st.write("No data to visualize")
        else:
            if not gen_report:
                data_html = convert_data_to_html(data.head(1000))
                components.html(data_html, width=None, height=350, scrolling=True)
            elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
                if data.shape[1] <= 20:
                    general_report = get_report(df=data, show_1d=True, show_2d=False, show_3d=False, show_dragndrop=False)
                    show_viz(general_report)
                else:
                    st.dataframe(data.head(1000))
            else:
                st.write(data)

@st.experimental_fragment
def show_data(data:pd.DataFrame, title="View AutoEDA", button_type="secondary"):
    if isinstance(data, pd.DataFrame):
        data_html = convert_data_to_html(data.head(1000))
        components.html(data_html, width=None, height=350, scrolling=True)

        if st.button(title, disabled=st.session_state.user_question["full_process"].current_step is not None, help="View detailed information of the data'", type=button_type):
            gen_data_report(data=data)
    else:
        st.write(data)

def show_dictionary(result: dict, title: str = "Dictionary"):
    def display_dict(d):
        for key, val in d.items():
            st.markdown(f"<b>{key.upper()}</b>", unsafe_allow_html=True)
            if isinstance(val, dict):
                display_dict(val)
            else:
                st.text(val)

    with st.expander(title, expanded=True):
        display_dict(result)

def show_analysis(result):
    if isinstance(result, pd.DataFrame):
        show_data(data=result, title="View result")
    elif isinstance(result, dict):
        show_dictionary(result, title="Result")
    else:
        st.write(result)

