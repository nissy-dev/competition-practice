import pandas as pd


def read_data(BASE_PATH):
    print('Reading train.csv file....')
    train = pd.read_csv(BASE_PATH + 'train.csv')
    print('Training.csv file have {} rows and {} columns'.format(
        train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    print('Test.csv file have {} rows and {} columns'.format(
        test.shape[0], test.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv(BASE_PATH + 'sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(
        sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, sample_submission
