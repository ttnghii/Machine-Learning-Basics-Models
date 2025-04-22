import math
from IPython.display import display
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


font = {
    'fontsize': 16,
    'fontstyle': 'italic',
    'backgroundcolor': 'black',
    'color': 'white'
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


def draw_pieplot(df: pd.DataFrame, ncols: int, figsize: tuple) -> None:
    '''
        Draw pie plot for each column of a dataframe
    '''
    _, axes = plt.subplots(
        nrows=math.ceil(df.shape[1] / ncols),
        ncols=ncols,
        figsize=figsize
    )

    for i, col in enumerate(df.columns):
        ax = axes[i // ncols][i % ncols]
        ax.pie(
          df[col].value_counts(),
          labels=df[col].value_counts().index,
          colors=sns.color_palette('YlOrBr'),
          autopct='%1.1f%%',
          shadow=True,
          startangle=90
        )
        ax.set_title(f'Pie plot of {col}', fontdict=font, pad=15)
    plt.tight_layout()
    plt.show()
