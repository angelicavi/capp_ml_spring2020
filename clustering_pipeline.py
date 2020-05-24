import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import data_processing as dp
import classifiers as cfs
import matplotlib.pyplot as plt

'''
Clustering Pipeline

Angelica Valdiviezo I.

Last updated: May 2020

'''
SEED = 0

###############################################################################
############################# 1) READ DATA ####################################
###############################################################################

def get_processed_df(file):
    '''
    Loads the projects file and processes it by:
        1. Imputing median values for continuous columns. The median values are
            estimated by groups (first by schoolid, if no median value for that
            school, then for the school county)
        2. Creates dummies for discrete columns.
        3. Return only numeric columns and predictors (to be used on a supervised ML pipeline).
    Input: filepath
    Output: clean df ready for clustering
    '''
    # Get the csv file
    df = dp.get_projects_df(file)

    # Impute for numeric columns by school id and school county (doesn't include lat and long)
    print("Imputing median values for numeric columns")
    numeric_cols = dp.get_cols_to_discrete(df)
    for col in numeric_cols:
        df = fill_median_by_group(df, col, 'schoolid')
        df = fill_median_by_group(df, col, 'school_county')

    # Create dummies for discrete cols
    print("\nCreating dummies for discrete columns")
    cols_to_binary = dp.clean_cols_to_binary(df)
    df = dp.create_dummies(df, cols_to_binary)

    # Create label
    date_cols = ["date_posted", "datefullyfunded"]
    df = dp.create_period_outcome(df, date_cols[0], date_cols[1])
    obs, _ = df.shape
    outcome = dp.get_outcome_lbl()
    df = df.dropna(subset=[outcome])
    new_obs, _ = df.shape
    print('Dropped {} obs with no valid value for outcome'.format(obs - new_obs))

    return df

def get_processed_pred_df(pred_file, all_df):
    '''
    Processes the prediction sample
    '''
    discrete_cont_dummies = [
        'total_price_including_optional_support_discrete_bottom_outlier',
        'total_price_including_optional_support_discrete_low',
        'total_price_including_optional_support_discrete_med',
        'total_price_including_optional_support_discrete_high',
        'total_price_including_optional_support_discrete_top_outlier',
        'total_price_including_optional_support_discrete_nan',
        'students_reached_discrete_bottom_outlier',
        'students_reached_discrete_low',
        'students_reached_discrete_med',
        'students_reached_discrete_high',
        'students_reached_discrete_top_outlier',
        'students_reached_discrete_nan']

    all_df = all_df[['projectid','teacher_acctid', 'schoolid', 'not_funded_within_60days']]
    df = pd.read_csv(pred_file)
    df = df.merge(all_df, how='left', on=['projectid','teacher_acctid', 'schoolid'])
    df = df.drop(['Unnamed: 0', 'school_ncesid'] + discrete_cont_dummies, axis=1)

    df = df[df.prediction == 1]
    df.drop('prediction', axis=1, inplace=True)

    return df

#### HELPERS ###

def fill_median_by_group(original_df, col, grouping_col):
    '''
    Fills missing values of a continuos column with the median value for
    a certain group. Default is by schoolid.
    '''
    if original_df[col].isnull().sum() == 0:
        print('\tNo missing values in column:', col)
        return original_df

    print('\tFilling {} with median values by {}'.format(col, grouping_col))
    median_vals = original_df.groupby([grouping_col]).agg({col:np.median})
    median_vals = median_vals.to_dict()[col]

    df = original_df[original_df[col].isnull()].copy(deep=True)
    original_df = original_df.drop(original_df[original_df[col].isnull()].index)

    def _get_median_val(row):
        col_val = row[col]
        school = row[grouping_col]
        if pd.isna(col_val):
            col_val = median_vals[school]
        return col_val

    df[col] = df.apply(_get_median_val, axis=1)
    original_df = original_df.append(df)

    # Some groups did not have a median value, we will still have nas
    if original_df[col].isnull().sum() != 0:
        num_missing = original_df[col].isnull().sum()
        print('\t\tStill have {} missing values'.format(num_missing))

    return original_df

###############################################################################
###################### 2) MAKE AND MODIFY CLUSTERS ############################
###############################################################################

def get_cluster_cat(n_clusters, df, temp_col=False):
    '''
    Get clusters and add column with cluster category in df.
    '''
    predictors = cfs.get_predictors(df)
    numeric_cols = dp.get_cols_to_discrete(df)
    # In case a cluster column or a log colu was already created
    clean_numeric_cols = [col for col in numeric_cols if not "cluster" in col]
    clean_numeric_cols = [col for col in clean_numeric_cols if not "log_" in col]

    cols_for_clustering = [predictors + clean_numeric_cols]

    for col in ['Unnamed: 0', 'school_ncesid', 'prediction']:
        try:
            cols_for_clustering.remove(col)
            print("Removed col:", col)
        except ValueError as e:
            continue


    kmeans = KMeans(n_clusters,
                    random_state=SEED,
                    n_jobs=-1)\
                   .fit(df[predictors + clean_numeric_cols])

    col_name = 'cluster_k_{}'.format(n_clusters)
    if temp_col:
        col_name = 'cluster_k_{}_temp'.format(n_clusters)

    df[col_name] = pd.Series(kmeans.labels_)


    return df

def merge_clusters(df, cluster_col, to_merge, new_cluster):
    '''
    Merges two or more categories of clusters.

    Inputs:
        - df with already formed clusters
        - cluster_col (str) with cluster categories
        - to_merge (list of str) of previous cluster categories to merge
        - num_cluster (int) value of the new cluster. For clarity, this 
            value should not have been used in this previous clusterization

    Output: df with new categories in the same cluster_col
    '''
    merged_df = df[df[cluster_col].isin(to_merge)].copy(deep=True)
    df = df[~df[cluster_col].isin(to_merge)]

    merged_df[cluster_col] = new_cluster

    df = df.append(merged_df)

    print('Clusters {} were merged into cluster {}'.format(to_merge, new_cluster))

    return df

def cluster_within_cluster(df, cluster_col, cluster, num_new_clusters):
    '''
    Creates 'sub-clusters' within a given cluster.
    '''
    new_clusters_df = df[df[cluster_col] == cluster].copy(deep=True)

    df = df[df[cluster_col] != cluster]

    temp_df = get_cluster_cat(num_new_clusters, new_clusters_df, temp_col=True)
    temp_col = 'cluster_k_{}_temp'.format(num_new_clusters)

    new_clusters_df.drop(cluster_col, axis=1, inplace=True)
    new_clusters_df[cluster_col] = temp_df[temp_col] + 10

    new_clusters_df = new_clusters_df.drop(temp_col, axis=1)
    new_clusters = list(new_clusters_df[cluster_col].unique())

    df = df.append(new_clusters_df, sort=True)
    print("Cluster {} was subclustered into {}".format(cluster, new_clusters))

    return df

###############################################################################
############################ 3) DESCRIBE CLUSTERS #############################
###############################################################################
def hist_clusters(df, num_clusters, sample):
    '''
    '''
    cluster_col = 'cluster_k_{}'.format(num_clusters)
    ax = df[cluster_col].value_counts().sort_values().plot(kind='barh')
    fig = ax.get_figure()
    fig.savefig('results/figures/cluster{}_hist_{}.png'.format(num_clusters, sample))
    plt.show()

def plot_cluster_by_theme(df, num_clusters, themes, sample):
    '''
    Plot clusters by thematic columns (themes). 
    '''
    theme_dict = get_col_categories()

    for theme in themes:
        try:
            cols = theme_dict[theme]['cols']
            clean_cols = theme_dict[theme]['clean_cols']
            bar_pcts_cols(df, num_clusters, cols, clean_cols, sample, theme)
        except KeyError as e:
            print('Invalid theme:', theme)

def plot_cluster_for_binary(df, num_clusters, sample, binary_cols=None):
    '''
    Plot clusters by thematic columns (themes). 
    '''
    if not binary_cols:
        binary_cols = get_binary_cols(df)

    clean_cols = [" ".join(col.split("_")).capitalize() for col in binary_cols]

    bar_pcts_cols(df, num_clusters, binary_cols, clean_cols, sample)

def plot_clusters_num_vars(df, num_clusters, num_var1, num_var2, sample, log=False, hide_outliers=False, outlier_pct=97, save=True):
    '''
    Plot clusters for continuos varibles (num_var1, num_var2). 
    '''
    cluster_col = 'cluster_k_{}'.format(num_clusters)
    variables = [num_var1, num_var2]
    fig_name = 'results/figures/cluster{}_scatter_{}.png'.format(num_clusters, sample)
    fig_title = "DonorChoose.org projects' Price versus Students Reached,\nby cluster"
    if log:
        df['log_' + num_var1] = np.log(df[num_var1])
        df['log_' + num_var2] = np.log(df[num_var2])
        variables = ['log_' + num_var1, 'log_' + num_var2]
        fig_name = 'results/figures/cluster{}_log_scatter.png'.format(num_clusters)
        fig_title = "DonorChoose.org projects' Log of Price versus Log of\n Students Reached, by cluster"

    groups = df.groupby(cluster_col)
    fig, ax = plt.subplots(figsize=(8,8))
    y_pct = 0
    x_pct = 0
    for pred_class, group in groups:
        ax.scatter(group[variables[0]],
                   group[variables[1]],
                   label=pred_class)
        xpct = np.percentile(group[variables[0]], outlier_pct)
        ypct = np.percentile(group[variables[1]], outlier_pct)
        if x_pct < xpct:
            x_pct = xpct
        if y_pct < ypct:
            y_pct = ypct
    
    if log:
        xlabel = "Log " + " ".join(num_var1.split("_")).capitalize()
        ylabel = "Log " + " ".join(num_var2.split("_")).capitalize()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if hide_outliers:
        ax.set_xlim(left=0,right=x_pct)
        ax.set_ylim(bottom=0,top=y_pct)
        fig_name = 'results/figures/cluster{}_scatter_{}_no_outliers.png'.format(num_clusters, sample)
        fig_title += ' (hidden outliers)'


    ax.legend(title='Cluster')
    ax.set_xlabel(" ".join(variables[0].split("_")).capitalize())
    ax.set_ylabel(" ".join(variables[1].split("_")).capitalize())
    plt.title(fig_title)
    if save:
        plt.savefig(fig_name)


    return plt

################### HELPERS ##################
def bar_pcts_cols(df, num_clusters, cols, clean_cols, sample, theme=None, save=True):
    '''
    Plots bar charts for a given k (num of clusters) and a possibly theme of columns.
    '''
    cluster_col = 'cluster_k_{}'.format(num_clusters)

    plot_df = clean_df_bar_plot(df, cols, cluster_col)
    pct_df = plot_df[cols]

    x = np.arange(len(cols))*15
    bars_coords, width = get_coords(x, num_clusters)

    fig, ax = plt.subplots(figsize=(8,8))
    for k, coords in enumerate(bars_coords):
        rects = ax.bar(coords,
                    np.array(pct_df[plot_df[cluster_col] == k])[0],
                    width,
                    label='Cluster {}'.format(k))
        autolabel(rects, ax)

    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(clean_cols)

    ax.set_ylabel("Pct of observations")
    ax.set_ylim([0,105])

    by_type = 'binary_columns'
    if theme:
        clean_theme = " ".join(theme.split("_")).capitalize()
        ax.set_xlabel(clean_theme)
        by_type = theme
    title = 'Proportion of observations by cluster and {}'.format(" ".join(by_type.split("_")))
    plt.title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.savefig('results/figures/cluster{}_{}_pct_{}.png'.format(num_clusters, by_type, sample))
    plt.show()

def get_coords(x, k):
    '''
    Makes coords for a given number of cols (x) and a number of 
    clusters (k).
    '''
    coords = [[] for x in range(k)]
    if k == 2:
        width = 6
        coords[0] = x - width/2
        coords[1] = x + width/2
    elif k == 3:
        width = 4
        coords[0] = x - width
        coords[1] = x
        coords[2] = x + width
    elif k == 4:
        width = 3
        coords[0] = x - 3/2 * width
        coords[1] = x - width/2
        coords[2] = x + width/2
        coords[3] = x + 3/2 * width
    elif k == 5:
        width = 2
        coords[0] = x - 3/2 * width
        coords[1] = x - width/2
        coords[2] = x + width/2
        coords[3] = x + 3/2 * width

    return coords, width

def autolabel(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    Source:https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def clean_df_bar_plot(df, cols, cluster_col):
    '''
    Reshapes a dataframe to get % bars
    '''
    plot_df = df[cols + [cluster_col]]\
                .groupby(cluster_col)\
                .agg([sum, 'count'])

    original_cols = plot_df.columns

    for col in cols:
        plot_df[(col, 'pct')] = plot_df[(col, 'sum')] / plot_df[(col,'count')] * 100

    plot_df = plot_df.drop(original_cols, axis=1)
    plot_df.columns = plot_df.columns.droplevel(1)
    plot_df = plot_df.reset_index()
    return plot_df

def get_col_categories():
    '''
    Returns a dictionary with the name of the thematic category and
    its associated dictionary of columns and clean labels for plotting.
    '''

    poverty_cols = ['poverty_level_low poverty',
              'poverty_level_moderate poverty',
              'poverty_level_high poverty',
              'poverty_level_highest poverty']
    grade_cols = ['grade_level_Grades PreK-2',
                    'grade_level_Grades 3-5',
                    'grade_level_Grades 6-8',
                    'grade_level_Grades 9-12']
    p_focus_cols = ['primary_focus_area_Applied Learning',
           'primary_focus_area_Health & Sports',
           'primary_focus_area_History & Civics',
           'primary_focus_area_Literacy & Language',
           'primary_focus_area_Math & Science',
           'primary_focus_area_Music & The Arts',
           'primary_focus_area_Special Needs']
    s_focus_cols = ['secondary_focus_area_Applied Learning',
           'secondary_focus_area_Health & Sports',
           'secondary_focus_area_History & Civics',
           'secondary_focus_area_Literacy & Language',
           'secondary_focus_area_Math & Science',
           'secondary_focus_area_Music & The Arts',
           'secondary_focus_area_Special Needs']
    metro_cols = ['school_metro_rural', 'school_metro_suburban', 'school_metro_urban']
    resource_cols = ['resource_type_Books', 'resource_type_Other', 'resource_type_Supplies',
           'resource_type_Technology', 'resource_type_Trips',
           'resource_type_Visitors']
    cols = {
        'poverty_level': {'cols': poverty_cols,
                           'clean_cols': [col.replace('poverty_level_','').capitalize() for col in poverty_cols]},
        'grade_level': {'cols': grade_cols,
                           'clean_cols': [col.replace('grade_level_','').capitalize() for col in grade_cols]},
        'primary_focus_area': {'cols': p_focus_cols,
                           'clean_cols': [col.replace('primary_focus_area_','').capitalize() for col in p_focus_cols]},
        'secondary_focus_area': {'cols': s_focus_cols,
                           'clean_cols': [col.replace('secondary_focus_area_','').capitalize() for col in s_focus_cols]},
        'school_metro': {'cols': metro_cols,
                           'clean_cols': [col.replace('school_metro_','').capitalize() for col in metro_cols]},
        'resource_type': {'cols': resource_cols,
                           'clean_cols': [col.replace('resource_type_','').capitalize() for col in resource_cols]}
            }

    return cols

def get_binary_cols(df):
    '''
    Obtain the binary cols that are not thematic.
    Output: list of column names.
    '''
    thematic_dict = get_col_categories()
    thematic_cols = []
    for theme in thematic_dict:
        for col in thematic_dict[theme]['cols']:
            thematic_cols.append(col)

    binary_cols = []
    for col in df.columns:
        try:
            binary = df[col].nunique() < 3
            not_thematic = col not in thematic_cols
            not_cluster = 'cluster_k' not in col
            not_nan_col = 'nan' not in col
            if binary and not_thematic and not_cluster and not_nan_col:
                binary_cols.append(col)
        except:
            continue
    return binary_cols
