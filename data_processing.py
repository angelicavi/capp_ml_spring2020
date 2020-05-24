'''
Machine Learning Pipeline

Angelica Valdiviezo I.

Last updated: May 2020

This program cleans and pre processes data for a Machine
Learning project.
    1.  Reads the data. This has been tailored for data from the 
        project https://www.kaggle.com/c/GiveMeSomeCredit
    2.  Visualize and explore the data.
    3.  Pre-process the data so it is ready to run models.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gp
from scipy import stats
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

###############################################################################
############################# 1) READ DATA ####################################
###############################################################################

def get_projects_df(data_filepath=None):
    '''
    Read the projects csv file into a pandas df. Converts date columns to datetime format and string boolean columns to boolean.
    Input:
        -data_filepath (str) of the location of the csv file
    Ouput: pandas dataframe.
    '''
    if not data_filepath:
        data_filepath = "data/projects_2012_2013.csv"

    var_types = {'projectid':str,
                 'teacher_acctid':str,
                 'schoolid':str,
                 'school_ncesid':str,
                 'school_latitude':float,
                 'school_longitude':float}
    date_cols = ['datefullyfunded', 'date_posted']
    bool_cols = ['school_charter', 'school_magnet',
                'eligible_double_your_impact_match']
    to_binary = {'t': True, 'f': False}
    to_gender = {'Mrs.': True,
                 'Ms.': True,
                 'Mr.': False,
                 'Dr.': False} # we only have two doctors, I think it is best not to add another category as we will convert them in dummies

    df = pd.read_csv(data_filepath, dtype=var_types)

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%m/%d/%y')

    for col in bool_cols:
        df[col] = df[col].map(to_binary)

    df['female_teacher'] = df['teacher_prefix'].map(to_gender)
    df.drop(['teacher_prefix'], axis=1, inplace=True)

    return df

def get_gdf(df, lng_col, lat_col):
    '''
    Read a Pandas DataFrame csv file into a pandas df.
    Input:
        -df of data
        -lng_col(str): column with longitude of point
        -lat_col(str): column with latitude of point
    Ouput: 
    GeoDataFrame with a Shapely Geometry Point from the original latitude and
    longitude columns (which are dropped)
    '''
    gdf = gp.GeoDataFrame(df,
                          geometry=gp.points_from_xy(df[lng_col], df[lat_col]),
                          crs={'init' :'epsg:4326'})
    gdf.drop([lng_col, lat_col], axis=1, inplace=True)
    print("Converted {} and {} to Shapely Point and dropped from GDF"\
                .format(lng_col, lat_col))

    return gdf

def get_ids_df(df, id_cols):
    '''
    TO DO
    Creates a deep copy with IDs and same index
    Modifies origial df in place to delete the ids
    '''
    rv = df[id_cols].copy(deep=True)
    df.drop(id_cols, axis=1, inplace=True)
    print("Dropped ID column(s) from original df")

    return rv

def create_period_outcome(df, initial_date, end_date):
    '''
    TO DO
    '''
    outcome_label = get_outcome_lbl()

    df[outcome_label] = (df[end_date] - df[initial_date]) > pd.to_timedelta(60, unit='days')
    max_date = df[initial_date].max()
    sixty_day_window = (max_date - df[initial_date]) < pd.to_timedelta(60, unit='days')

    df.loc[sixty_day_window, outcome_label] = np.nan
    print("Created outcomes label")
    return df

###############################################################################
############################# 2) EXPLORE DATA #################################
###############################################################################

def describe_data(df, date_cols=None, weak_b=0.45, moderate_b=0.75, describe_each=False):
    '''
    Prints basic stats of a pandas dataframe.
    Inputs:
        -df (pandas dataframe): data.
        -weak_b (float): upper bound for defining the range of absolute
        values of the correlation coefficiente that would classify as
       relative "weak" correlation. Serves as lower bound (not inclusive) for "moderate"
        correlation.
        -moderate_b (float): upper bound for defining the range of absolute
        values of the correlation coefficiente that would classify as
        "moderate" correlation. Serves as lower bound for "strong"
        correlation.
    '''
    print("Number of obs:", df.shape[0])
    print()
    missing_cols = get_missing_cols(df)
    print("Columns with missing data:", missing_cols)
    print()
    corrs = get_corr_vars(df.select_dtypes(include=[np.number]), weak_b, moderate_b)
    if corrs["strong"]:
        print("Strongly correlated numeric features:")
        for features in corrs["strong"]:
            print("\t", features)
        print()
    if corrs["moderate"]:
        print("Moderately correlated numeric features:")
        for features in corrs["moderate"]:
            print("\t", features)
    print("Description of numeric features:\n")
    print(df.describe())

    if date_cols:
        print("\nDescription of date columns:")
    for col in date_cols:
        print('\nDate column {}'.format(col))
        print(df[col].describe())

    print("\nString features:",
          list(df.select_dtypes(include=[object]).columns))
    print("\nBoolean features:",
          list(df.select_dtypes(include=[bool]).columns))
    print("\nPotencial cols to binary:", get_cols_to_binary(df))

def view_data(df, list_plots=None):
    '''
    Provides basic visualizations of the data.

    Input:
        - df: a Pandas Dataframe.
        - list_plots (optional): list of dicts, where each dictionary
        represents a scatterploof to get, which should have for keys:
                - 'xcol': target or other columns of interest)
                - 'ycol': other feature of interests.
                - 'colorcol': feature to colorcode by. optional.

    Outputs: correlation matrix and scatterplots
    '''
    print("Correlation matrix:")
    view_corr_matrix(df)

    if list_plots:
        print("\nScatterplot(s):")
        view_scatterplots(df, list_plots)

    view_count_plots(df)
    view_histograms(df)
    view_boxplots(df)

### HELPER FUNCTIONS
def view_corr_matrix(df):
    '''
    Get a plot that visually shows the (absolute) correlations between
    variables of the df.
    '''
    corr = df.corr().abs()

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(corr, mask=mask, square=True, cmap='Blues')
    plt.show()

def view_scatterplots(df, list_plots):
    '''
    Gets scatterplots of different columns (usually target versus feature of
    interest), with the option to colorcode by other feature.

    Inputs:
        - df: a Pandas Dataframe.
        - list_plots: list of dicts, where each dictionary represents a
            scatterploof to get, which should have for keys:
                - 'xcol': target or other columns of interest)
                - 'ycol': other feature of interests.
                - 'colorcol': feature to colorcode by. optional.

    Output:
        Scatterplot(s)

    '''

    for plt in list_plots:
        xcol = plt['xcol']
        ycol = plt['ycol']
        get_scatterplot(df, xcol, ycol, plt.get('colorcol', None))

def view_histograms(df):
    '''
    TO DO
    '''
    numeric_cols = get_cols_to_discrete(df)

    for col in numeric_cols:
        df[col].hist()
        plt.title("Histogram of {}".format(col))
        plt.show();

def view_count_plots(df):
    '''
    TO DO
    '''
    cols_to_binary = get_cols_to_binary(df)

    for i, col in enumerate(cols_to_binary):
        ax = sns.countplot(x=col, data=df)
        plt.title("Countplot for col: {}".format(col))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        plt.tight_layout()
        plt.show();

def view_boxplots(df):
    '''
    TO DO
    '''
    cols_to_discretize = get_cols_to_discrete(df)

    for i, col in enumerate(cols_to_discretize):
        sns.boxplot(x=df[col])
        plt.title("Boxplot for col: {}".format(col))
        plt.show();

def get_scatterplot(df, xcol, ycol, colorcol=None):
    '''
    Get a scatterplot of x (usually target) against y (a feature of interest).

    Inputs:
        - df: Pandas DataFrame with data.
        - xcol: str
        - ycal: str
        - colorcode: str, column to color by. optional.

    '''
    #df.plot.scatter(x=xcol, y=ycol, c=colorcol, cmap='viridis',legend=True, logx=logx, logy=logy)

    # df.plot.scatter(x=df[xcol], y=df[ycol], c=colorcol, cmap='viridis', legend=True, logx=False, logy=False)

    df.plot(kind='scatter', x=xcol, y=ycol, c=colorcol)

    title = 'Scatterplot of \n {} and {}'.format(xcol, ycol)
    if colorcol:
        title += ' color coded by {}'.format(colorcol)
    plt.title(title)
    plt.show()

def get_corr_vars(df, weak_b, moderate_b):
    '''
    Generate a list of moderately and strongly correlated variables.
    Inputs:
        -df (pandas dataframe): data.
        -weak_b (float): upper bound for defining the range of absolute
        values of the correlation coefficiente that would classify as
        "weak" correlation. Serves as lower bound (not inclusive) for "moderate"
        correlation.
        -moderate_b (float): upper bound for defining the range of absolute
        values of the correlation coefficiente that would classify as
        "moderate" correlation. Serves as lower bound for "strong"
        correlation.
    '''
    cols = df.columns
    redundant_corr = set()
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            redundant_corr.add((cols[i], cols[j]))

    abs_corr = df.corr().abs().unstack().drop(labels=redundant_corr)
    abs_corr = abs_corr.sort_values(ascending=False)

    mod_strong_corr = {"moderate": [], "strong": []}
    for features, corr in abs_corr.items():
        if corr > moderate_b:
            mod_strong_corr["strong"].append(features)
        elif (weak_b < corr) and (corr <= moderate_b):
            mod_strong_corr["moderate"].append(features)

    return mod_strong_corr

def get_cols_to_discrete(df):
    '''
    Obtain the preliminary list of the numeric columns which can
    be discretized.
    Input: df.
    Output: list of column names
    '''
    cols_to_discrete = [col for col in df.columns if col in 
            df.select_dtypes(include=[np.number])]
    rv = []
    for col in cols_to_discrete:
        if not ('latitude' in col or 'longitude' in col):
            rv.append(col)

    return rv

def get_cols_to_binary(df, unique_val_lim=10):
    '''
    Obtain the preliminary list of the discrete columns which can
    be converted to binary (with dummies).

    Inputs:
        - df: a pandas df
        - unique_val_lim (int): max number of categories in a given
            value. If a column has more unique values than this, it will
            not be considered a potential column to binary. Default is 10.

    Output: list of column names.
    '''
    non_num_cols = [col for col in df.columns if col not in 
            df.select_dtypes(include=[np.number])]

    cols_to_binary = []

    for col in non_num_cols:
        try:
            if df[col].nunique() < unique_val_lim:
                cols_to_binary.append(col)
        except:
            continue

    cols_to_binary.sort()

    return cols_to_binary

def clean_cols_to_binary(df, target_col=None, irrelevant_cols=None):
    '''
    Obtains the final list of proposed columns to binary by cleaning
    columns not relevant to be converted into dummies and the target column.

    Inputs:
        - df: a pandas df
        - target_col (str): name of the target column.
        - irrelevant_cols(list of str): columns for which dummies are irrelevant. Default: 'eligible_double_your_impact_match',
        'female_teacher', 'school_magnet','school_charter'.

    Ouput: clean list of columns to binary.

    '''
    cols_to_binary = get_cols_to_binary(df)

    if not irrelevant_cols:
        irrelevant_cols = ['eligible_double_your_impact_match',
                           'female_teacher',
                           'school_magnet',
                           'school_charter']

    if target_col:
        irrelevant_cols += [target_col]

    for col in irrelevant_cols :
        try:
            cols_to_binary.remove(col)
        except:
            continue

    return cols_to_binary

def get_missing_cols(df):
    '''
    Returns the list of columns of a dataframe that have missing values.
    '''
    return df.columns[df.isna().any()].tolist()

def get_outcome_lbl():
    '''
    TO DO
    '''
    return 'not_funded_within_60days'

###############################################################################
############################# 3) PRE PROCESS DATA #############################
###############################################################################

def get_temporal_dfs(df, d_col, target_col, pred_win_len, prediction_gap=0, cols_to_discrete=None):
    '''
    Function to obtain dfs for temporal validation sets.

    Inputs:
        - df
        - d_cols. (str): date column
        - target_col (str): outcome
        - pred_win_len(int): days for the rolling window.
        - prediction_gap(int): days of the gap to predict. Default is 0.
        - cols_to_discrete(lst of str): columns to make discrete. Default is None and
            the function will obtain them.
    
    Output:
        - dictionary 
    '''
    if not cols_to_discrete:
        cols_to_discrete = get_cols_to_discrete(df)

    start_date = df[d_col].min()
    end_date = df[d_col].max()
    gap = relativedelta(days=+prediction_gap)
    pred_window = relativedelta(days=+pred_win_len)
    split_date = start_date + pred_window

    # Create dummies before processing DFs so all datasets have the same number of features
    cols_to_binary = clean_cols_to_binary(df, target_col)
    print("Creating binary columns:", cols_to_binary)
    df = create_dummies(df, cols_to_binary)
    print('\nComplete data set has {} obs and {} variables'\
          .format(df.shape[0], df.shape[1]))

    rv = {}
    while split_date < end_date - pred_window:
        print('\nWorking with train/test set split at:', split_date)
        rv[split_date] = {}

        train_dict = {}
        train_dict['start_date'] = start_date
        # this is the effective train date to account for the gap
        train_dict['end_date'] = split_date - gap
        rv[split_date]['train'] = train_dict

        test_dict = {}
        test_dict['start_date'] = split_date + relativedelta(days=+1)
        test_dict['end_date'] = split_date + pred_window - gap
        rv[split_date]['test'] = test_dict

        rv[split_date] = add_processed_dfs(df, d_col, rv[split_date],
                                           cols_to_binary,
                                           cols_to_discrete)
        split_date += pred_window

    return rv

### HELPER FUNCTIONS
def add_processed_dfs(df, d_col, split_rv, cols_to_binary, cols_to_discrete):
    '''
    Processes a df 
    '''
    print_tab = '\t'

    train_start_d = split_rv['train']['start_date']
    train_end_d = split_rv['train']['end_date']
    test_start_d = split_rv['test']['start_date']
    test_end_d = split_rv['test']['end_date']

    # Get the train and test df, process them, add them to the corresponding split dictionary
    train_df = df[df[d_col] >= train_start_d][df[d_col] <= train_end_d]\
                .copy(deep=True)
    print(print_tab, 'Working with train df')
    train_df = process_df(train_df, cols_to_binary, cols_to_discrete, print_tab=print_tab)
    split_rv['train']['df'] = train_df
    print(print_tab, 'Train DF starts at {}, ends at {}, has {} obs and {} cols'.format(train_df[d_col].min(),
                                   train_df[d_col].max(), train_df.shape[0],
                                   train_df.shape[1]))

    print()
    test_df = df[df[d_col] >= test_start_d][df[d_col] <= test_end_d]\
                .copy(deep=True)
    print(print_tab, 'Working with test df')
    test_df = process_df(test_df, cols_to_binary, cols_to_discrete, print_tab=print_tab)
    split_rv['test']['df'] = test_df
    print(print_tab, 'Test DF starts at {}, ends at {}, has {} obs and {} cols'.format(test_df[d_col].min(), test_df[d_col].max(),
                                   test_df.shape[0], test_df.shape[1]))

    return split_rv

def process_df(df, cols_to_binary, cols_to_discrete=None,
               quantiles=[0, 0.01 ,.33, .66, .99, 1],
               q_labels=["bottom_outlier","low","med","high","top_outlier"],
               print_tab=''):
    '''
    Processes data by:
        1) Filling missing values with median (for numeric cols)
        2) Discretizing continuos cols (and dropping the numeric cols)
        3) Create dummies for columns of step 2

    Input:
        -df, pandas dataframe.
        -cols_to_discrete (list of strings): continuos cols to discretize.
        -cols_to_binary (list of strings): categorical cols to convert to
            binary.
    # TODO correct inpus
        - bins_ (int): number of bins to discretize into. 
            Defult is 3 (high, medium, low)
        - labels_: Labels for the returned bins. Must be the same length as
            the resulting bins. If False, returns only integer indicators of
            the bins.

    Ouput: preprocessed dataframe.
    '''
    if print_tab:
        print_tab += '\t'

    # Fill missing values of continous columns
    print(print_tab, "Filling missing values...")
    clean_df = fill_missing_vals(df, cols_to_discrete)
    if not cols_to_discrete:
        cols_to_discrete = get_cols_to_discrete(clean_df)

    # Discretize continuos columns
    print(print_tab, "Discretizing continous columns...")
    discretize_cont(df, cols_to_discrete, quantiles, q_labels)

    # Create dummy columns for the columns we just discretized
    print(print_tab, "Creating missing binary columns of just created discretized columns...")

    # Find the columns we just discretized and create their dummies.
    missing_binary = [col for col in df.columns if 'discrete' in col]
    clean_df = create_dummies(clean_df, missing_binary, print_tab)
    # TODO - clean names, they keep the 'discrete' suffix

    # Drop continous columns that where discretized and then dummy
    clean_df.drop(cols_to_discrete, axis=1, inplace=True)

    # Convert remaining boolean columns
    boolean_cols = list(df.select_dtypes(include='bool').columns)
    for col in boolean_cols:
        clean_df[col] = clean_df[col].astype(int)

    return clean_df

def fill_missing_vals(df, cols_to_fill=None):
    '''
    Manages missing values in a dataframe by filling them with the median of
    the non missing values of the variable. Modifies in place.
    '''
    if not cols_to_fill:
        cols_to_fill = list(df.select_dtypes(include=[np.number]).columns)

    for col in cols_to_fill:
        try:
            df[col].fillna(df[col].median(), inplace=True)
        except:
            continue

    # for col in df.select_dtypes(include=[object]):
    #     df[col].fillna('', inplace=True)
    #     print('\tFilled NaN in string column {} with empty string'.format(col))

    return df

def discretize_cont(df, cols_to_discrete, quantiles, q_labels,
                    cols_to_binary=None):
    '''
    Discretizes a continous variable in a dataframe. Modifies the
    dataframe in place. Resulting columns hace the suffix 'discrete'.
    If cols_to_binary is passed, the function will append discretized
    columns into the list.

    Inputs:
        - df (pandas df): a dataframe.
        - cols_to_discrete (list): cont. cols to discretize.
        # TODO FINISH
        - quantiles: (list) 
        - q_labels: .
        - cols_to_binary (list): discrete variables to create dummies from. 
    '''
    # Removing latitude and longitude

    new_discretes = []
    for col in cols_to_discrete:
        col_new_name = col + "_discrete"
        df[col_new_name] = pd.qcut(df[col],
                                  q=quantiles,
                                  labels=q_labels,
                                  duplicates='drop')
        if cols_to_binary and (col_new_name not in cols_to_binary):
            cols_to_binary.append(col_new_name)
    
    # 1 cambio
    if cols_to_binary:
        return cols_to_binary
    

def create_dummies(df, cols_to_binary, print_tab=''):
    '''
    Creates dummies for a list of categorical columns.
    
    Input:
        -df, a pandas dataframe.
        -cols_to_binary (list): list of categorical cols to create
            dummies for.
    '''
    if print_tab:
        print_tab += '\t'

    return pd.get_dummies(df, dummy_na=True, columns=cols_to_binary)

