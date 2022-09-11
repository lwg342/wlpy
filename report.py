import json
import torch
import shutil
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
# %%


class Report():
    def __init__(self, filename='report', file_format='md') -> None:
        self.file = filename
        self.format = file_format
        self.full_path = f'{self.file}.{self.format}'

    def timestamp(self):
        from datetime import datetime
        self.timestr = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.jot(self.timestr)

    def jot(self, input):
        with open(self.full_path, 'a') as f:
            f.write(f'{input}\n')

    def erase(self):
        with open(self.full_path, 'w') as f:
            f.write('')

    def savefig(self, fig, caption='', ask_erase=None, **kwargs):
        if ask_erase == None:
            ask_erase = input(
                'Do you want to erase the previous version? [y/n]')
        if ask_erase == 'y' or ask_erase:
            self.erase()

        if self.format == 'md':
            fig.savefig(self.file + '.png', format='png')
            self.jot(f'![{self.file}]({self.file}.png)')

        if self.format == 'tex':
            fig_format = 'eps'
            fig.savefig(f'{self.file}.{fig_format}', format=fig_format)
            self.jot(
                f'\\begin{{figure}}[htp]\n\includegraphics{{{self.file}.{fig_format}}}\n\\caption{{{caption}}}\n\\end{{figure}}')

    def savetable(self, df, caption='', ask_erase=None,escape = False,position = 'ht', bold_rows = True, **kwargs):
        if ask_erase == None:
            ask_erase = input(
                'Do you want to erase the previous version? [y/n]')
        if ask_erase == 'y' or ask_erase:
            self.erase()

        if self.format == 'tex':
            self.jot(df.to_latex(bold_rows=bold_rows, position=position, caption=caption, escape=escape,**kwargs)) 
# %% From CS230


class HyperParams():
    """Class that has the following functions:\n
    - loads hyperparameters from a json file.\n
    - save
    

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
        self.__json_path__ = json_path

    def save(self, json_path=None):
        if json_path == None:
            json_path = self.__json_path__
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def add(self):
        """Add new hyperparameters on the go"""
        pass

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

# %%


class DataReport(Report):
    """
    Write Reports for Data Cleaning jobs.
    """

    def __init__(self, filename, file_format='md'):
        super().__init__(filename, file_format)

    def df_head(self, DF, ncols=5, nrows=5):

        table = DF.iloc[:nrows, :ncols].to_markdown(
            tablefmt="github", floatfmt=".3f")
        self.jot(f'The header of the dataframe is\n\n{table}\n')

    def copy(self, file):
        with open(file, 'r') as f:
            content = f.read()
            self.jot(f'{content}\n')

# %%


class Timer():
    """
    Keep log of time
    """

    def __init__(self):
        pass

    def start(self):
        self.ctime = dt.now()

    def click(self):
        self.lapse = dt.now() - self.ctime
        print(self.lapse.total_seconds())


# %%
