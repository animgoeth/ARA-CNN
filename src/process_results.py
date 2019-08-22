import numpy as np
import pandas as pd
import argparse

from config import CLASS_DICT

REVERSED_CLASS_DICT = {}

def read_and_map_results(results_file):
    mapped_rows = []

    with open(results_file, 'r') as f:
        counter = 0
        for line in f:
            if counter == 0:
                counter += 1
                continue
            split_line = line.split(',')
            img_name = split_line[0]
            uncertainty = float(split_line[1])
            predicted_class = REVERSED_CLASS_DICT[np.argmax(split_line[2:])]

            mapped_rows.append((img_name, uncertainty, predicted_class))

    return mapped_rows

def find_most_uncertain_classes(results_file):
    mapped_rows = read_and_map_results(results_file)

    df = pd.DataFrame(mapped_rows, columns=['image', 'uncertainty', 'predicted_class'])
    grouped = df.groupby('predicted_class').mean()

    print('Most uncertain classes:')
    print(grouped.sort_values(["uncertainty"], ascending=False))


def find_least_uncertain_images(results_file, threshold):
    mapped_rows = read_and_map_results(results_file)
    df = pd.DataFrame(mapped_rows, columns=['image', 'uncertainty', 'predicted_class'])
    grouped = df.groupby('predicted_class').quantile(threshold)
    percentiles = grouped.to_dict()['uncertainty']

    chosen_images = []

    for row in mapped_rows:
        if row[1] < percentiles[row[2]]:
            chosen_images.append(row[0])

    print('Images with uncertainty below threshold q(%s):' % threshold)
    for img in chosen_images:
        print(img)


if __name__ == '__main__':
    for key, val in CLASS_DICT.items():
        REVERSED_CLASS_DICT[val] = key

    parser = argparse.ArgumentParser()
    parser.add_argument('--results-file', dest='results_file', help='Path to the results file')
    parser.add_argument('--threshold', dest='threshold', help='Uncertainty threshold', type=float)
    args = parser.parse_args()

    find_most_uncertain_classes(args.results_file)
    print()
    find_least_uncertain_images(args.results_file, args.threshold)
