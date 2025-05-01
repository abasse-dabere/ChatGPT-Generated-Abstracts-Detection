import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report

def get_statistics_df(df, column):
    """
    Calculate descriptive statistics for a specified column in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): The column name to calculate statistics for.
    Returns:
        pd.DataFrame: DataFrame containing the descriptive statistics.
    """
    statistics = df[column].groupby(df['label']).describe()
    statistics = statistics.loc[:, ['count', 'mean', '50%', 'std', 'min', 'max']]
    statistics.columns = ['count', 'mean', 'median', 'std', 'min', 'max']
    return statistics

def kdeplot_comparison(df, column, name):
    """
    Plot KDE plots for the specified column in the DataFrame.
    This function generates KDE plots for the following comparisons:
    - Human vs. Chatgpt-generated text
    - Human vs. Polish-generated text
    - Human vs. Mix-generated text
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): The column name to plot.
    """
    for label in df['label'].unique():
        if label == 'human': continue
        plt.figure(figsize=(10, 5))
        sns.kdeplot(df.loc[df['label'] == 'human'][column], label='human', color='blue', fill=True, alpha=0.5)
        sns.kdeplot(df.loc[df['label'] == label][column], label=label, color='red', fill=True, alpha=0.5)
        plt.title(f'{name} : Human vs {label.capitalize()}')
        plt.xlabel(name)
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def histplot_comparison(df, column, name):
    """
    Plot histogram plots for the specified column in the DataFrame.
    This function generates histogram plots for the following comparisons:
    - Human vs. Chatgpt-generated text
    - Human vs. Polish-generated text
    - Human vs. Mix-generated text
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): The column name to plot.
    """
    for label in df['label'].unique():
        if label == 'human': continue
        plt.figure(figsize=(10, 5))
        sns.histplot(df.loc[df['label'] == 'human'][column], label='human', color='blue', kde=False, stat='density', bins=30)
        sns.histplot(df.loc[df['label'] == label][column], label=label, color='red', kde=False, stat='density', bins=30)
        plt.title(f'{name} : Human vs {label.capitalize()}')
        plt.xlabel(name)
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def welch_ttest(m0, s0, n0, m1, s1, n1):
    """
    Perform Welch's t-test (two-sample t-test with unequal variances)
    from summary statistics: means (m0, m1), standard deviations (s0, s1),
    and sample sizes (n0, n1).

    Null hypothesis (H0): The two population means are equal: μ0 = μ1.
    Alternative hypothesis (H1): The two population means are different: μ0 ≠ μ1.

    Returns:
        t_stat  The Welch t-statistic.
        df      Degrees of freedom (Welch–Satterthwaite approximation).
        p_value Two-sided p-value.
    """
    se = np.sqrt(s0**2 / n0 + s1**2 / n1)
    t_stat = (m0 - m1) / se
    df = (s0**2/n0 + s1**2/n1)**2 / ((s0**4)/(n0**2*(n0 - 1)) + (s1**4)/(n1**2*(n1 - 1)))
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    return t_stat, df, p_value

def compare_groups(statistics, label0, label1):
    """
    Extract summary stats for two labels and run a Welch’s t-test.
    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    label0 : str
        First label to compare.
    label1 : str
        Second label to compare.
    Returns
    -------
    t_stat : float
        t-statistic from the Welch's t-test.
    df : int
        Degrees of freedom from the Welch's t-test.
    p_value : float
        p-value from the Welch's t-test.
    """
    # Retrieve the rows for each group
    row0 = statistics.loc[label0]
    row1 = statistics.loc[label1]

    # Call the Welch t-test function
    return welch_ttest(
        m0=row0['mean'], s0=row0['std'], n0=int(row0['count']),
        m1=row1['mean'], s1=row1['std'], n1=int(row1['count']),
    )

def plot_confusion_matrix(y_true, y_pred, labels = [0, 1], title = 'Confusion Matrix'):
    """
    Plot a confusion matrix using seaborn.
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        labels (list): List of labels for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[labels[0], labels[1]], yticklabels=[labels[0], labels[1]])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def print_metrics(y_true, y_pred, y_proba):
    """
    Print classification metrics including accuracy, precision, recall, and F1-score.
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        y_proba (list): Predicted probabilities.
    """
    accuracy = (y_pred == y_true).mean()
    print(f'Accuracy: {accuracy:.4f}')
    auc = roc_auc_score(y_true, y_proba)
    print(f'AUC: {auc:.4f}')