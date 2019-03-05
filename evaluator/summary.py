import json
import glob
import re
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Evaluate output of one synthesizer.')

parser.add_argument('--result', type=str, default='output/__result__',
                    help='result dir')
parser.add_argument('--summary', type=str, default='output/__summary__',
                    help='result dir')


def coverage(datasets, results):
    ticks = []
    values = []

    for model, result in results:
        covered = set()
        for item in result:
            assert(item['dataset'] in datasets)
            covered.add(item['dataset'])

        ticks.append(model)
        values.append(len(covered) / len(datasets))

    plt.cla()
    plt.bar(list(range(len(values))), values, tick_label=ticks)
    plt.title("coverage")
    plt.ylim(0, 1)

    plt.savefig("{}/coverage.pdf".format(summary_dir), bbox_inches='tight')


def dataset_performance(dataset, results):
    synthesizer_metric_perform = {}

    for synthesizer, all_result in results:
        for one_result in all_result:
            if one_result['dataset'] != dataset:
                continue

            for model_metric_score in one_result['performance']:
                for metric, v in model_metric_score.items():
                    if metric == "name":
                        continue
                    else:
                        if one_result['step'] == 0:
                            synthesizer_name = synthesizer
                        else:
                            synthesizer_name = synthesizer + "_" + int(item['step'])

                        if not synthesizer_name in synthesizer_metric_perform:
                            synthesizer_metric_perform[synthesizer_name] = {}

                        if not metric in synthesizer_metric_perform[synthesizer_name]:
                            synthesizer_metric_perform[synthesizer_name][metric] = []

                        synthesizer_metric_perform[synthesizer_name][metric].append(v)

    if len(synthesizer_metric_perform) == 0:
        return

    plt.cla()

    barchart = []
    for synthesizer, metric_perform in synthesizer_metric_perform.items():
        for k, v in metric_perform.items():
            barchart.append((synthesizer, k, np.mean(v)))

    barchart = pd.DataFrame(barchart, columns=['synthesizer', 'metric', 'val'])
    barchart.pivot("metric", "synthesizer", "val").plot(kind='bar')
    plt.title(dataset)
    plt.xlabel(None)
    plt.legend(title=None)
    plt.savefig("{}/{}.pdf".format(summary_dir, dataset), bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()

    result_files = glob.glob("{}/*.json".format(args.result))
    summary_dir = args.summary

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    datasets = glob.glob("data/*/*.npz")
    datasets = [re.search('.*/([^/]*).npz', item).group(1) for item in datasets]

    results = []
    for result_file in result_files:
        model = re.search('.*/([^/]*).json', result_file).group(1)
        with open(result_file) as f:
            res = json.load(f)

        results.append((model, res))

    coverage(datasets, results)
    for dataset in datasets:
        dataset_performance(dataset, results)