import numpy as np
import pandas as pd
 
def make_training_dataset(df, windows=[1], label='Numero carro',
                          drop=['Label', 'plate', 'plate_confidence', 'Talanquera']):
    """
    Makes the dataset for "different (or same) car classification". Concatenates 
    pairs of rows based on the elements (integers) in windows (list). 
    
    Also in the output dataframe goes our target column, 'different_cars', based on
    the column named label (string, name of labeled column in df). If label is None,
    does not create 'different_cars' column (useful for using trained classifier for
    prediction in deployment.)
    
    You can also specify which columns to drop (list of column names).

    Example:
    >>> df = pd.DataFrame({'a': ['a1', 'a2', 'a3', 'a4'],
    ...                    'b': ['b1', 'b2', 'b3', 'b4'],
    ...                    'c': ['c1', 'c2', 'c3', 'c4'],
    ...                    'l': ['l1', 'l1', 'l2', 'l2']})
    >>> df
        a   b   c   l
    0  a1  b1  c1  l1
    1  a2  b2  c2  l1
    2  a3  b3  c3  l2
    3  a4  b4  c4  l2
    >>> make_training_dataset(df, windows=[1,2], label='l', drop=['c'])
        ID a_1 b_1 a_2 b_2 different_cars
    0  0_1  a1  b1  a2  b2              0
    1  0_2  a1  b1  a3  b3              1
    2  1_2  a2  b2  a3  b3              1
    3  1_3  a2  b2  a4  b4              1
    >>> make_training_dataset(df, drop=['l'], label=None)
        ID a_1 b_1 c_1 a_2 b_2 c_2
    0  0_1  a1  b1  c1  a2  b2  c2
    1  1_2  a2  b2  c2  a3  b3  c3
    2  2_3  a3  b3  c3  a4  b4  c4
    """

    df = df.drop(columns=drop).dropna().reset_index(drop=True)
    
    if label:
        X = df.drop(columns=[label]).to_numpy()
        labels = list(df[label])
    else:
        X = df.to_numpy()

    X_cols = [col for col in df.columns if col != label]
   
    X1_cols = [col + '_1' for col in X_cols]
    X2_cols = [col + '_2' for col in X_cols]

    output = []
    if label:
        output_cols = ['ID'] + X1_cols + X2_cols + ['different_cars']
    else:
        output_cols = ['ID'] + X1_cols + X2_cols

    for i in range(X.shape[0]-max(windows)):
        for window in windows:
            ID = str(i) + '_' + str(i + window)

            X1 = X[i, :]
            X2 = X[i + window, :]

            if label:
                different_cars = int(labels[i] != labels[i + window])
                new_row = np.concatenate((np.array([ID]), X1, X2, 
                                          np.array([different_cars])))
            else:
                new_row = np.concatenate((np.array([ID]), X1, X2))

            output.append(new_row)

    output = np.array(output)

    return pd.DataFrame(output, columns=output_cols)