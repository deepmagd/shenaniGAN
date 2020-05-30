import os
import pandas as pd
import seaborn as sns
from shenanigan.utils.utils import mkdir, remove_file


class MetricsLogger(object):
    """ Define an object to handle all metric logging """
    def __init__(self, path, use_pretrained=False):
        if os.path.exists(path) and not use_pretrained:
            os.remove(path)
        self.path = path
        self.use_pretrained = use_pretrained
        # self.epoch_history = self._set_epoch_history()

    def __call__(self, metrics_dict):
        """ Log metrics to file """
        if not self.use_pretrained and metrics_dict['epoch'] == 1:
            # First time, create the metrics file
            new_metrics_df = pd.DataFrame(metrics_dict, index=[0])
            new_metrics_df.to_csv(self.path, index=False)
        else:
            # For all epochs after, check that we do not have duplicates and append
            metric_history_df = pd.read_csv(self.path)
            new_metrics_df = pd.DataFrame(metrics_dict, index=[metric_history_df.shape[0]])
            metric_history_df = metric_history_df[metric_history_df['epoch'] != metrics_dict['epoch']]
            metric_history_df = metric_history_df.append(new_metrics_df)
            remove_file(self.path)
            metric_history_df.to_csv(self.path, index=False)


class LogPlotter(object):
    """ Generate plots from logs """
    def __init__(self, root_path):
        self.root_path = root_path
        self.plot_dir = os.path.join(self.root_path, 'plots')
        mkdir(self.plot_dir)

    def _generate_learning_curve(self, method):
        log_path = os.path.join(self.root_path, f'{method}.csv')
        metric_df = pd.read_csv(log_path)
        metric_df = pd.melt(metric_df, 'epoch', value_name='loss')
        sns_plot = sns.lineplot(x='epoch', y='loss', hue='variable', data=metric_df)

        output_path = os.path.join(self.plot_dir, f'lc_{method}.png')
        remove_file(output_path)
        sns_plot.get_figure().savefig(output_path)

    def learning_curve(self):
        """ Save the learning curve plot """
        for x in ['train', 'val']:
            self._generate_learning_curve(x)
