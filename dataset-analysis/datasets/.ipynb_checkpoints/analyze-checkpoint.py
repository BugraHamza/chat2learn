import argparse
from datasets import load_dataset

def analyze(dataset):
    trainset = load_dataset(dataset, split='train')
    frequencies = {i: 0 for i in trainset.column_names}

    # analyze the word frequency
    for i in trainset:
        for col in trainset.column_names:
            frequencies[col] += len(i[col].split(' '))

    # calculate the average length
    average_lengths = {col: [] for col in trainset.column_names}
    for i in trainset:
        for col in trainset.column_names:
            average_lengths[col].append(len(i[col]))

    return frequencies
        
def print_results(name, results):

    cols, freqs = results

    print('-------------------------------------------')
    print(f'     {name} Results')
    print('-------------------------------------------')
    print('Column\t\tAverage Length')
    print('-------------------------------------------')
    for col in cols:

    print('Column\t\t\tFrequency')
    print('-------------------------------------------')
    for col, freq in freqs.items():
        print(f'{col}\t\t\t{freq}')
    print('-------------------------------------------')
    print('Overall\t\t\t', sum(freqs.values()))
    print('-------------------------------------------')
    
    print()
    print('Done..')


def main():
    parser = argparse.ArgumentParser(description='Analyze a file')
    parser.add_argument('dataset', help='the dataset to analyze', default='daily_dialog')
    args = parser.parse_args()
    
    print('Analyzing..', args.dataset)
    
    analysis_results = analyze(args.dataset)
    print_results(args.dataset, analysis_results)


if __name__ == '__main__':
    main()

