import os

import pandas as pd
import seaborn as sns

from shenanigan.utils.utils import mkdir, remove_file


class MetricsLogger(object):
    """ Define an object to handle all metric logging """
    def __init__(self, path):
        if os.path.exists(path):
            os.remove(path)
        self.logger = open(path, 'a+')
        self.columns = None

    def __call__(self, metrics_dict):
        """ Log metrics to file """
        if self.columns is None:
            # First time, log the metric names
            self.columns = list(metrics_dict.keys())
            self.logger.write(','.join(self.columns) + '\n')

        text_line = ','.join([str(metrics_dict[metric]) for metric in self.columns])
        self.logger.write(f'{text_line}\n')

    def close(self):
        self.logger.close()

class LogPlotter(object):
    """ Generate plots from logs """
    def __init__(self, root_path):
        self.root_path = root_path
        self.plot_dir = os.path.join(self.root_path, 'plots')
        mkdir(self.plot_dir)

    def learning_curve(self):
        """ Save the learning curve plot """
        log_path = os.path.join(self.root_path, 'train.csv')
        metric_df = metric_df = pd.read_csv(log_path)
        metric_df = pd.melt(metric_df, 'epoch', value_name='loss')
        sns_plot = sns.lineplot(x='epoch', y='loss', hue='variable', data=metric_df)

        output_path = os.path.join(self.plot_dir, 'lc.png')
        remove_file(output_path)
        sns_plot.get_figure().savefig(output_path)
