# Collection of functions I have created to for ease and also many contributions from other data dorks such as myself. 
# Credits can be found at the end of each function


# import libraries general libraries
import pandas as pd
import numpy as np

# Modules for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot, \
    plot  # iplot() = plots the figure(fig) that is created by data and layout
import plotly.express as px
import plotly.figure_factory as ff
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

#plt.rcParams['figure.figsize'] = [6, 6]

# ignore DeprecationWarning Error Messages
import warnings

warnings.filterwarnings('ignore')
pio.templates.default = 'seaborn'

mdict = dict( l=50,r=50, b=50,t=50,pad=4)
bgcolor="LightSteelBlue"

# style: https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html
# color: https://matplotlib.org/stable/tutorials/colors/colormaps.html
class bold_color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def clean_header(df):
    """
    This functions removes weird characters and spaces from column names, while keeping everything lower case
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')



def data_research(data, data_name='data', un=False):
    print(f'Examining "{data_name}"\n')
    
    #basic
    print(data.info(),'\n')
    #head
    print()
    display(data.head(2))
    #display(data.info())
    display(data.describe(),'\n')
    #display(data.columns)
    print(data.columns)
    print()
    #duplicates
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print('There are no duplicated entries.','\n')
    else:
        print(f'There are {duplicates} duplicates.','\n')
        
    #missing
    #print('\n\033[1mObservation:\033[0m The Missing Values counts in {} are.'.format(data_name),'\n')
    data_missing = pd.DataFrame(data.isnull().sum())
    if data_missing[0].sum() > 0:
        print(data_missing)
    else:
        print('\n\033[1mObservation:\033[0m There are no missing values in {}.'.format(data_name),'\n')

    print()
    print()
    print("The percentage of missing data from each column is:\n")
    null_data = (((data.isnull().sum().sort_values(ascending=False))/len(data))*100).round(2)
    display(null_data.to_frame())
    
    #missing_bar(data)
    
    #Checking number of unique rows in each feature

    features= data.columns
    nu = data[features].nunique().sort_values()
    nf = []; cf = []; nnf = 0; ncf = 0; #numerical & categorical features

    for i in range(data[features].shape[1]):
        if nu.values[i]<=7:cf.append(nu.index[i])
        else: nf.append(nu.index[i])

    print('\n\033[1mObservation:\033[0m The Datset has {} numerical & {} categorical features.'.format(len(nf),len(cf)))
    
def table(df):
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=df.transpose().values.tolist(),align='center',font=dict(color='black', size=12)))
    ])

    fig.show()

def get_redundant_pairs(data):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = data.columns
    for i in range(0, data.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(data, n):
    au_corr = data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def heatmaps_tri(df):
    corr = round(df.corr(),3)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask)

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                    x=df_mask.columns.tolist(),
                                    y=df_mask.columns.tolist(),
                                    colorscale=px.colors.cyclical.Edge,
                                    showscale=True, ygap=1, xgap=1
                                    )

    fig.update_xaxes(side="bottom")

    fig.update_layout(
        #font=fontdict,
        #title_font=titledict,
        title_text='Correlations', 
        title_x=0.5, 
        width=1000, 
        height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed'#,
        #template='plotly_dark'
    )

    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    fig.show()

def style(table):
    """
    quick styling
    """
    view = table.style.background_gradient(cmap='Pastel1')
    return view


def percentage(s):
    """
    Converts a series to round off - percentage string format.
    """
    x = s.apply(lambda x: round(x / s[:].sum() * 100, 2))
    x = x.apply(lambda x: str(x) + '%')
    return x


def query_this(df, col, look):
    """
    Easy == Query
    """
    query_to_return = df.query('{} == "{}"'.format(col, look))
    return query_to_return


def missing_bar(df) -> go.Figure:
    """Plots Missing Data for Whole Dataset."""
    title = 'Stack Overflow Developer Survey Results 2021 <b>Missing Data by Features</b>'

    # counts missing data
    missing_data = df.isna().sum()
    missing_data = missing_data.to_frame().reset_index().rename(
        columns={'index': 'data_cols', 0: 'counts'})
    missing_data = missing_data.sort_values(by='counts', ascending=False)
    missing_perc = np.round(
        (df.isna().sum().sum() / df.size) * 100, 2)

    # figure colors
    colors = ['#f2f0eb'] * len(missing_data)
    colors[:10] = ['blue']

    # create figure
    fig = go.Figure()
    for labels, values \
            in zip(missing_data.data_cols.to_list(), (missing_data.counts/len(df)*100)):
        fig.add_trace(go.Bar(
            y=[labels],
            x=[values],
            name=labels,
            orientation='h'))

    # tweak layout
    fig.update_traces(marker_colorscale=px.colors.diverging.Portland)
    fig.update_xaxes(title='Missing Amount (Percentage)')
    fig.update_yaxes(title='Features', tickmode='linear')
    fig.update_layout(#font=fontdict,
                        #title_font=titledict,
                        title_x=0.5,
                        height=1000)
    fig.add_annotation(xref='paper', yref='paper',
                       x=0.5, y=1.10, text='Total Data Missing: '+str(missing_perc)+'%',
                       #font=titledict3,
                       bordercolor="#c7c7c7",
                       borderwidth=2,
                       borderpad=4,
                       bgcolor="black",
                       opacity=0.9,
                       showarrow=False)



    #add_bubble(fig)

    fig.show()
    #return paste_px_format(
    #    fig, title=title, height=1000, showlegend=False)


def missing_percentage(df):
    for col in df.columns:
        missing_percentage = df[col].isnull().mean()
        print(f'{col} - {missing_percentage :.1%}')


# Describing data
def group_median_aggregation(df, group_var, agg_var):
    # Grouping the data and taking median
    grouped_df = df.groupby([group_var])[agg_var].median().sort_values(ascending=False)
    return grouped_df





def split_multicolumn(col_series):
    result_df = col_series.to_frame()
    options = []
    # Iterate over the column
    for idx, value in col_series[col_series.notnull()].iteritems():
        # Break each value into list of options
        for option in value.split(';'):
            # Add the option as a column to result
            if not option in result_df.columns:
                options.append(option)
                result_df[option] = False
            # Mark the value in the option column as True
            result_df.at[idx, option] = True
    return result_df[options]


# function that imputes median
def impute_median(series):
    return series.fillna(series.median())


def calculate_min_max_whisker(df):
    """
    Calculates the values of the 25th and 75th percentiles
    It takes the difference between the two to get the interquartile range (IQR).
    Get the length of the whiskers by multiplying the IQR by 1.5
    Calculate the min and max whisker value by subtracting
    Add the whisker length from the 25th and 75th percentile values.
    """
    q_df = df.quantile([.25, .75])
    q_df.loc['iqr'] = q_df.loc[0.75] - q_df.loc[0.25]
    q_df.loc['whisker_length'] = 1.5 * q_df.loc['iqr']
    q_df.loc['max_whisker'] = q_df.loc['whisker_length'] + q_df.loc[0.75]
    q_df.loc['min_whisker'] = q_df.loc[0.25] - q_df.loc['whisker_length']
    return q_df


# credit: https://www.kaggle.com/jaepin
# Visualization to check % of missing values in each column
def paste_px_format(figure, **kwargs):
    """Updates Layout of the Figure with custom setting"""
    return figure.update_layout(**kwargs,
                                font={'color': 'Gray', 'size': 10},
                                width=780, margin={'pad': 10})


def add_bubble(fig, **kwargs):
    """Creates shape ontop of the figure"""
    return fig.add_shape(
        type="circle",
        line_color="white",
        fillcolor="lightblue",
        opacity=0.6,
        xref='paper', yref='paper',
        x0=0.5, y0=0.5)


def whitespace_remover(df):
    """
    The function will remove extra leading and trailing whitespace from the data.
    Takes the data frame as a parameter and checks the data type of each column.
    If the column's datatype is 'Object.', apply strip function; else, it does nothing.
    Use the whitespace_remover() process on the data frame, which successfully removes the extra whitespace from the columns.
    https://www.geeksforgeeks.org/pandas-strip-whitespace-from-entire-dataframe/
    """
    # iterating over the columns
    for i in df.columns:

        # checking datatype of each columns
        if df[i].dtype == 'str':

            # applying strip function on column
            df[i] = df[i].map(str.strip)
        else:
            # if condition is False then it will do nothing.
            pass







#Created on Sat Feb 23 14:42:46 2019
#@author: DATAmadness


##################################################
# A function that will accept a pandas dataframe
# and auto-transforms columns that exceeds threshold value
#  -  Offers choice between boxcox or log / exponential transformation
#  -  Automatically handles negative values
#  -  Auto recognizes positive /negative skewness

# Further documentation available here:
# https://datamadness.github.io/Skewness_Auto_Transform

import seaborn as sns
import numpy as np
import math
import scipy.stats as ss
import matplotlib.pyplot as plt

def skew_autotransform(DF, include = None, exclude = None, plot = False, threshold = 1, exp = False):
    
    #Get list of column names that should be processed based on input parameters
    if include is None and exclude is None:
        colnames = DF.columns.values
    elif include is not None:
        colnames = include
    elif exclude is not None:
        colnames = [item for item in list(DF.columns.values) if item not in exclude]
    else:
        print('No columns to process!')
    
    #Helper function that checks if all values are positive
    def make_positive(series):
        minimum = np.amin(series)
        #If minimum is negative, offset all values by a constant to move all values to positive teritory
        if minimum <= 0:
            series = series + abs(minimum) + 0.01
        return series
    
    
    #Go throug desired columns in DataFrame
    for col in colnames:
        #Get column skewness
        skew = DF[col].skew()
        transformed = True
        
        if plot:
            #Prep the plot of original data
            sns.set_style("darkgrid")
            sns.set_palette("Blues_r")
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            ax1 = sns.distplot(DF[col], ax=axes[0])
            ax1.set(xlabel='Original ' + col)
        
        #If skewness is larger than threshold and positively skewed; If yes, apply appropriate transformation
        if abs(skew) > threshold and skew > 0:
            skewType = 'positive'
            #Make sure all values are positive
            DF[col] = make_positive(DF[col])
            
            if exp:
               #Apply log transformation 
               DF[col] = DF[col].apply(math.log)
            else:
                #Apply boxcox transformation
                DF[col] = ss.boxcox(DF[col])[0]
            skew_new = DF[col].skew()
         
        elif abs(skew) > threshold and skew < 0:
            skewType = 'negative'
            #Make sure all values are positive
            DF[col] = make_positive(DF[col])
            
            if exp:
               #Apply exp transformation 
               DF[col] = DF[col].pow(10)
            else:
                #Apply boxcox transformation
                DF[col] = ss.boxcox(DF[col])[0]
            skew_new = DF[col].skew()
        
        else:
            #Flag if no transformation was performed
            transformed = False
            skew_new = skew
        
        #Compare before and after if plot is True
        if plot:
            print('\n ------------------------------------------------------')     
            if transformed:
                print('\n %r had %r skewness of %2.2f' %(col, skewType, skew))
                print('\n Transformation yielded skewness of %2.2f' %(skew_new))
                sns.set_palette("Paired")
                ax2 = sns.distplot(DF[col], ax=axes[1], color = 'r')
                ax2.set(xlabel='Transformed ' + col)
                plt.show()
            else:
                print('\n NO TRANSFORMATION APPLIED FOR %r . Skewness = %2.2f' %(col, skew))
                ax2 = sns.distplot(DF[col], ax=axes[1])
                ax2.set(xlabel='NO TRANSFORM ' + col)
                plt.show()
                

    return DF