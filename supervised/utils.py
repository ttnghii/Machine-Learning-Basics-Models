import math
from IPython.display import display
import pandas as pd
import numpy as np
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import ClassifierMixin
from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score, confusion_matrix, classification_report, \
    ConfusionMatrixDisplay, roc_curve, auc


font = {
    'fontsize': 16,
    'fontstyle': 'normal',
    'backgroundcolor': 'black',
    'color': 'white',
}


def read_dataset(path: str) -> pd.DataFrame:
    ''' 
        Read .csv file from a path and return a DataFrame
    '''
    df = pd.read_csv(path)
    print('The first three rows of this data frame:')
    display(df.head(3))
    print('Description of this dataframe:')
    display(df.describe().T)
    return df


def add_noise(x: float, noise_level: float = 0.01) -> float:
    '''
        Add noise to your data to handle duplicated values
        :param df: pandas DataFrame
        :return: pandas DataFrame
    '''
    return x + noise_level * np.random.randn()


# Custom plots for categorical data


def custom_pieplot(
    df: pd.DataFrame, ncols: int, figsize: tuple, color: str
) -> None:
    '''
        Draw pie plot for each column of a categorical dataframe
    '''
    _, axes = plt.subplots(
        nrows=math.ceil(df.shape[1] / ncols), ncols=ncols, figsize=figsize
    )
    axes = np.ravel(axes)  # Convert axes to 1D array

    for i, col in enumerate(df.columns):
        ax = axes[i]
        ax.pie(
            df[col].value_counts(),
            # labels=df[col].value_counts().index,
            # autopct='%1.1f%%',
            colors=sns.color_palette(color),
            shadow=True,
            startangle=90,
        )
        ax.set_title(f'Pie plot of {col}', fontdict=font, pad=15)

    for j in range(len(df.columns), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def custom_barplot(
    df: pd.DataFrame, target_col: pd.Series, ncols: int, figsize: tuple, color: str
) -> None:
    '''
        Draw bar plot for each column of a categorical dataframe
        to show the relationship between them with target column.
    '''
    _, axes = plt.subplots(
        nrows=math.ceil(df.shape[1] / ncols), ncols=ncols, figsize=figsize
    )
    axes = np.ravel(axes)
    columns_list = [col for col in df.columns]
    palette = sns.color_palette(color, df.shape[1])

    for i, col in enumerate(columns_list):
        ax = axes[i]
        sns.barplot(data=df, x=target_col, y=col, ax=ax, color=palette.pop(0))
        ax.bar_label(ax.containers[0], fontsize=14)
        ax.set_title(f'Barplot of {col}', fontdict=font, pad=15)

    for j in range(len(df.columns), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


# Custom plots for numerical data


def custom_histplot(
    df: pd.DataFrame, ncols: int, figsize: tuple, color: str
) -> None:
    '''
        Draw hist plot for each column of a numerical dataframe
    '''
    _, axes = plt.subplots(
        nrows=math.ceil(df.shape[1] / ncols), ncols=ncols, figsize=figsize
    )
    axes = np.ravel(axes)  # Convert axes to 1D array
    palette = sns.color_palette(color, df.shape[1])

    for i, col in enumerate(df.columns):
        ax = axes[i]
        sns.histplot(
            data=df, x=col, ax=ax, kde=True, color=palette[i]
        )
        ax.set_title(f'Hist plot of {col}', fontdict=font, pad=15)
   
    for j in range(len(df.columns), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def custom_boxplot(df: pd.DataFrame, ncols: int, color: str) -> None:
    '''
        Draw box plot for each column of a numerical dataframe
        to classify outliers.
    '''
    _, axes = plt.subplots(
        nrows=math.ceil(df.shape[1] / ncols),
        ncols=ncols,
        figsize=(14, 2*df.shape[1]),
        sharex=False,
        sharey=False
    )
    axes = np.ravel(axes)
    palette = sns.color_palette(color, df.shape[1])

    for i, col in enumerate(df.columns):
        ax = axes[i]
        sns.boxplot(data=df, x=col, ax=ax, color=palette.pop(0))
        ax.set_title(f'Box plot of {col}', fontdict=font, pad=15)
    
    for j in range(len(df.columns), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def custom_countplot(
    df: pd.DataFrame, target: str, ncols: int, figsize: tuple, color: str
) -> None:
    '''
        Draw countplot for catogirical dataframe to show the relationship of
        these columns with target column.
    '''
    _, axes = plt.subplots(
        nrows=math.ceil(df.shape[1] / ncols),
        ncols=ncols,
        figsize=figsize,
        sharex=False,
        sharey=False
    )
    axes = np.ravel(axes)

    for i, col in enumerate(df.columns):
        ax = axes[i]
        sns.countplot(data=df, x=col, hue=target, ax=ax, palette=color)
        ax.set_title(f'Count plot of {col}', fontdict=font, pad=15)

    for j in range(len(df.columns), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


# Custom plots for both categorical and numerical data



def custom_heatmap(df: pd.DataFrame, figsize: tuple, color: str) -> None:
    '''
        Draw heatmap for a dataframe
    '''
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), square=True, annot=True, cmap=color)
    plt.title('Correlation Matrix', fontdict=font, pad=12)
    plt.show()


# Process specific values

def remove_outliers_byusing_quantile(
    df: pd.DataFrame, thresh: float = 1.5
) -> pd.DataFrame:
    '''
        Remove outlier if this data points not in [low_bound, upper_bound].
    '''
    print(f'Dataset shape Before remove outlier: {df.shape}')
    outliers = []
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - (thresh * iqr), q3 + (thresh * iqr)
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers.append(mask)
        print(f'{col}: {mask.sum()}')

    if outliers:
        outliers_idx = np.any(outliers, axis=0)
        df = df[~outliers_idx]

    print(f'Dataset shape After remove outlier: {df.shape}')
    return df


def remove_outlier_byusing_zscores(
    df: pd.DataFrame, thresh: float = 3.0
) -> pd.DataFrame:
    '''
        Remove outlier if this data points not in [low_bound, upper_bound].
        Must apply StandardScaler for df before apply this function.
    '''
    print(f'Dataset shape Before remove outlier: {df.shape}')
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = (z_scores > thresh).any(axis=1)
    df = df[~outliers]
    print(f'Dataset shape After remove outlier: {df.shape}')
    return df


def remove_uncorrelated_feature(
    df: pd.DataFrame, thresh: float = 0.1
) -> pd.DataFrame:
    '''
        Remove uncorrelated feature
    '''
    uncorrelated_feature = set()
    corr_matrix = df.corr()

    for col in corr_matrix.columns:
        if abs(corr_matrix[col][-1]) < thresh:
            uncorrelated_feature.add(col)

    df.drop(columns=uncorrelated_feature, axis=1, inplace=True)
    return df


# For modeling

def prepare_x_y(df: pd.DataFrame, target: str) -> tuple():
    '''
        Feature engineering and create x and y
            :param df: pandas dataframe
            :return: (x, y) output feature matrix (dataframe), target (series)
    '''
    x = df.drop(columns=[target]).values
    y = np.array(df[target]).reshape((-1, 1))
    return x, y


def calculate_performance(y_true: np.array, y_pred: np.array) -> None:
    '''
        :param y_true: ground truth values
        :param y_pred: predictions
        :return: metrics to evaluate model's performance
    '''
    performance = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    print(f'Precision: {performance["precision"]}')
    print(f'Recall: {performance["recall"]}')
    print(f'Accuracy: {performance["accuracy"]}')
    print(f'F1: {performance["f1"]}')
    print(f'Confusion matrix:\n{confusion_matrix(y_true, y_pred)}')
    print(f'Classification report:\n{classification_report(y_true, y_pred)}')


def display_cfs_matrix(y_true: np.array, y_pred: np.array, color: str) -> None:
    '''
        Show confusion matrix
    '''
    display_ = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred)
    )
    display_.plot(cmap=color)
    plt.show()

def custom_roc_curve(
    model1: ClassifierMixin,
    name_model1: str,
    model2: ClassifierMixin,
    name_model2: str,
    x_test: np.array,
    y_test: np.array
):
    '''
        Show ROC curve
    '''
    y_prob1 = model1.predict_proba(x_test)[:, 1]
    fpr1, tpr1, _ = roc_curve(y_test, y_prob1)
    roc_auc1 = auc(fpr1, tpr1)

    y_prob2 = model2.predict_proba(x_test)[:, 1]
    fpr2, tpr2, _ = roc_curve(y_test, y_prob2)
    roc_auc2 = auc(fpr2, tpr2)

    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr1,
        tpr1,
        color='green',
        linestyle=":",
        linewidth=4,
        label=f'ROC curve of {name_model1} model (AUC = {roc_auc1:.2f})'
    )
    plt.plot(
        fpr2,
        tpr2,
        color='deeppink',
        linestyle=":",
        linewidth=4,
        label=f'ROC curve of {name_model2} model (AUC = {roc_auc2:.2f})'
    )
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def display_model_metric(y_test: np.array, y_pred: np.array, name: str) -> List[list]:
    '''
        Show metric of model
    '''
    show, all_models = [], []
    show.append([
        name,
        round(accuracy_score(y_test, y_pred), 2),
        round(precision_score(y_test, y_pred, zero_division=0), 2),
        round(recall_score(y_test, y_pred), 2),
        round(f1_score(y_test, y_pred), 2),
    ])
    all_models.append([
        name,
        round(accuracy_score(y_test, y_pred), 2),
        round(precision_score(y_test, y_pred, zero_division=0), 2),
        round(recall_score(y_test, y_pred), 2),
        round(f1_score(y_test, y_pred), 2),
    ])
    return show



# def custom_pr_curve(
#     model: ClassifierMixin,
#     name_model: str,
#     x_test: np.array,
#     y_test: np.array
# ):
#     # Predict probabilities
#     y_pred = model.predict_proba(x_test)[:, 1] # keep probabilities for the positive outcome only
#     # Predict class values
#     yhat = model.predict(x_test)
#     prec, rec, thresh = precision_recall_curve(testY, y_pred)
#     f1, auc_ = f1_score(testY, yhat), auc(rec, prec)
#     # Summarize scores
#     print(f'{name_model}: F1={round(f1, 3)}, Auc={round(auc_, 3)}')

#     # Plot the precision-recall curves
#     no_skill = len(testY[testY==1]) / len(testY)
#     plt.plot(
#         [0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', color='green'
#     )
#     plt.plot(rec, prec, marker='.', label=name_model, color='deeppink')
#     # Axis labels
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend(loc='lower right')
#     plt.grid(True)
#     plt.show()