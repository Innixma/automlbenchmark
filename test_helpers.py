def ration_train_test(train_df, test_df, percent_test: float = 0.75):
    num_train = len(train_df)
    num_test = len(test_df)
    total_rows = num_train + num_test
    desired_num_test = int(percent_test * total_rows)

    test_num_diff = desired_num_test - num_test

    if test_num_diff > 0:
        sample_frac = test_num_diff / num_train
        more_test = train_df.sample(frac=sample_frac, random_state=1)
        train_df = train_df.drop(more_test.index)
        test_df.append(more_test)
    elif test_num_diff < 0:
        sample_frac = abs(test_num_diff) / num_test
        more_train = test_df.sample(frac=sample_frac, random_state=1)
        test_df = test_df.drop(more_train.index)
        train_df.append(more_train)

    return train_df, test_df