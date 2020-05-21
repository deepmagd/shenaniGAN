import os

import pandas as pd
import seaborn as sns

from shenanigan.utils.utils import mkdir, remove_file


class MetricsLogger(object):
    """ Define an object to handle all metric logging """
    def __init__(self, path, continue_training=False):
        if os.path.exists(path) and not continue_training:
            os.remove(path)
        self.path = path
        self.continue_training = continue_training
        self.columns = None

    def __call__(self, metrics_dict):
        """ Log metrics to file """
        with open(self.path, "a+") as logger:
            if self.columns is None:
                # First time, log the metric names
                self.columns = list(metrics_dict.keys())
                if not self.continue_training:
                    logger.write(','.join(self.columns) + '\n')

            text_line = ','.join([str(metrics_dict[metric]) for metric in self.columns])
            logger.write(f'{text_line}\n')

class LogPlotter(object):
    """ Generate plots from logs """
    def __init__(self, root_path):
        self.root_path = root_path
        self.plot_dir = os.path.join(self.root_path, 'plots')
        mkdir(self.plot_dir)

    def _generate_learning_curve(self, method):
        log_path = os.path.join(self.root_path, f'{method}.csv')
        metric_df = metric_df = pd.read_csv(log_path)
        metric_df = pd.melt(metric_df, 'epoch', value_name='loss')
        sns_plot = sns.lineplot(x='epoch', y='loss', hue='variable', data=metric_df)

        output_path = os.path.join(self.plot_dir, f'lc_{method}.png')
        remove_file(output_path)
        sns_plot.get_figure().savefig(output_path)

    def learning_curve(self):
        """ Save the learning curve plot """
        for x in ['train', 'val']:
            self._generate_learning_curve(x)
