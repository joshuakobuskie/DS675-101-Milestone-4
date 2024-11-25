import numpy as np
import pandas

def train_test_split(dataframe, target, test_size=0.1, rand_state=None):

    # Control randomness for both np and sampling
    if rand_state is None:
        rand_state = np.random.randint(0, 1024)

    np.random.seed(rand_state)

    # Shuffle dataset
    dataframe = dataframe.sample(frac=1.0, random_state=rand_state).reset_index(drop=True)

    # Stratify
    dataframe_stratified = dataframe.groupby(target)

    train_indexes = []
    test_indexes = []

    # Split
    for index, group in dataframe_stratified:
        test_group_size = int(len(group) * test_size)

        # Extract and shuffle values by stratified group
        indexes = group.index.to_list()
        np.random.shuffle(indexes)

        # Save straitified indexes
        test_indexes.extend(indexes[:test_group_size])
        train_indexes.extend(indexes[test_group_size:])
    
    # Make and return dataframes
    train_dataframe = dataframe.loc[train_indexes].reset_index(drop=True)
    test_dataframe = dataframe.loc[test_indexes].reset_index(drop=True)
    return train_dataframe, test_dataframe

# The above is great and all, but sklearn also has these methods
# We can probably just get rid of this too