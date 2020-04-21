import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MissVals_Deleter(BaseEstimator, TransformerMixin):
    def __init__(self, missing_perc = 50): # no *args or **kargs
        self.missing_perc = missing_perc
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        missing_df=missing_values_table(X)
        missing_columns = list(missing_df[missing_df['% of Total Values'] > self.missing_perc].index)
        print('By the remove percentage criterion %d, we may remove %d columns.' % (self.missing_perc, len(missing_columns)))
        X = X.drop(columns = list(missing_columns))
        return X

class Outliers_Remover(BaseEstimator, TransformerMixin):
    def __init__(self, colnames_list): # colnames list
        self.colnames_list=colnames_list
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        for colname in self.colnames_list:
            # Interquartile range
            iqr = X[colname].describe()['75%'] - X[colname].describe()['25%']
            # Remove outliers from the pd.df.describe low and high quartile
            X= X[(X[colname] > (X[colname].describe()['25%'] - 2 * iqr))]
            X = X[(X[colname] < (X[colname].describe()['75%'] + 2 * iqr))]

        return X

class Feature_AdderEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target, cat_list): # no *args or **kargs
        self.target = target
        self.cat_list = cat_list
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        # Select the numeric columns
        numeric_subset = X.select_dtypes('number')
        # Create columns with square root and log of numeric columns
        for col in numeric_subset.columns:
            # Skip the target column
            if col == self.target:
                next
            else:
                numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
                numeric_subset['log_' + col] = np.log(numeric_subset[col])

        # Select the chosen categorial subset
        categorical_subset = X[self.cat_list]
        # One hot encode
        categorical_subset = pd.get_dummies(categorical_subset)

        features = pd.concat([numeric_subset, categorical_subset], axis = 1)

        # Drop buildings without an energy star score
        X = features.dropna(subset = [self.target])

        # Reset index for further manipulation in order to avoid problems
        X.reset_index(inplace=True)

        # replace all inf values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # set dtype in order to avoid problems
        X = X.astype('float32')

        return X

class Feature_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, target, cat_list): # no *args or **kargs
        self.target = target
        self.cat_list = cat_list
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        # Select the numeric columns
        numeric_subset = X.select_dtypes('number')

        # Select the chosen categorial subset
        categorical_subset = X[self.cat_list]
        # One hot encode
        categorical_subset = pd.get_dummies(categorical_subset)

        features = pd.concat([numeric_subset, categorical_subset], axis = 1)

        # Drop buildings without an energy star score
        X = features.dropna(subset = [self.target])

        # Reset index for further manipulation in order to avoid problems
        X.reset_index(inplace=True)

        # replace all inf values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # set dtype in order to avoid problems
        X = X.astype('float32')

        return X

class CollinearFeatures_Remover(BaseEstimator, TransformerMixin):
    def __init__(self, target, threshold=0.6): # colnames list
        self.target=target
        self.threshold=threshold
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        '''
        Objective:
            Remove collinear features in a dataframe with a correlation coefficient
            greater than the threshold. Removing collinear features can help a model
            to generalize and improves the interpretability of the model.

        Inputs:
            threshold: any features with correlations greater than this value are removed

        Output:
            dataframe that contains only the non-highly-collinear features
        '''

        # Dont want to remove correlations between Energy Star Score
        y = X[self.target]
        X = X.drop(columns = [self.target])

        # Calculate the correlation matrix
        corr_matrix = X.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the threshold
                if val >= self.threshold:
                    # Print the correlated features and the correlation value
                    # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                    drop_cols.append(col.values[0])

        # Drop one of each pair of correlated columns
        drops = set(drop_cols)
        X = X.drop(columns = drops)

        # Add the score back in to the data
        X[self.target] = y

        return X

class SpecColumn_Remover(BaseEstimator, TransformerMixin):
    def __init__(self, colnames_list): # colnames list
        self.colnames_list=colnames_list
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        # Special drops
        X = X.drop(columns = self.colnames_list)
        return X


class StratTrainTest_Splitter(BaseEstimator, TransformerMixin):
    def __init__(self, target, bins, labels, test_size = 0.3, random_state = 4711, verbose = False): # no *args or **kargs
        self.target = target
        self.bins = bins
        self.labels = labels
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        no_score = X[X[self.target].isna()]
        score = X[X[self.target].notnull()]
        print(no_score.shape)
        print(score.shape)
        # Stratification helper
        score['strat_cat'] = pd.cut(score[self.target],self.bins,self.labels)
        # Separate out the features and targets
        features = score.drop(columns=self.target)
        targets = pd.DataFrame(score[self.target])
        # Replace the inf and -inf with nan (required for later imputation)
        features = features.replace({np.inf: np.nan, -np.inf: np.nan})
        #features['strat_cat'].hist()
        # Split into training and testing set
        from sklearn.model_selection import train_test_split
        X, X_test, y, y_test = train_test_split(features, targets,
                                                test_size = self.test_size, random_state = self.random_state, stratify = features['strat_cat'])
        if self.verbose == True:
            def rp_cat_proportions(data):
                return data['strat_cat'].value_counts() / len(data)

            train_set, test_set = train_test_split(features, test_size=self.test_size, random_state=self.random_state)

            compare_props = pd.DataFrame({
                "Overall": rp_cat_proportions(features),
                "Stratified": rp_cat_proportions(X_test),
                "Random": rp_cat_proportions(test_set),
            }).sort_index()
            compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
            compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
            print(compare_props)

        X.drop(columns=['strat_cat'],inplace=True)
        X_test.drop(columns=['strat_cat'],inplace=True)
        return X, X_test, y, y_test

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns
