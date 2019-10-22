import pandas as pd
 
def make_training_dataset(csv_path, windows=[1], labels=['Label', 'Numero carro'],
                           drop=['plate', 'plate_confidence', 'Talanquera']):
    df = pd.read_csv(csv_path, index_col=0).drop(columns=drop).dropna(
            ).reset_index(drop=True)
    
    X_cols = [col for col in df.columns if col not in labels]

    
    X1_cols = [col + '_1' for col in X_cols]
    X2_cols = [col + '_2' for col in X_cols]

    output = []

    for i in df.index[:-max(windows)]:
        for window in windows:
            frames = str(df.loc[i, 'frame_number']) + '_' + str(df.loc[i + window, 'frame_number'])

            X1 = df.loc[i, X_cols]
            X2 = df.loc[i + window, X_cols]

            X1_dict = dict(zip(X1_cols, X1))
            X2_dict = dict(zip(X2_cols, X2))

            different_cars = int(df.loc[i, 'Numero carro'] != df.loc[i + window, 'Numero carro'])

            new_row = {'frames': frames, **X1_dict, **X2_dict, 'different_cars': different_cars}

            output.append(new_row)

    return pd.DataFrame(output)[['frames'] + X1_cols + X2_cols + ['different_cars']]

    