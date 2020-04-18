import os

class MetricsLogger(object):
    """ Define an object to handle all metric logging """
    def __init__(self, path):
        if os.path.exists(path):
            os.remove(path)
        self.logger = open(path, 'a+')
        self.columns = None

    def __call__(self, metrics_dict):
        """ Log metrics to file """
        if self.columns == None:
            # First time, log the metric names
            self.columns = list(metrics_dict.keys())
            self.logger.write(','.join(self.columns) + '\n')

        text_line = ','.join([str(metrics_dict[metric]) for metric in self.columns])
        self.logger.write(f'{text_line}\n')
