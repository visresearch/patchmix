
import os
import time
import logging

from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, log_root='./', name='', logger_name=''):
        os.makedirs(log_root, exist_ok=True)
        if logger_name == '':
            date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.log_name = '{}_{}'.format(name, date)
            # self.log_name = date
            log_dir = os.path.join(log_root, self.log_name)
            super(Logger, self).__init__(log_dir, flush_secs=1)
        else:
            self.log_name = logger_name
            log_dir = os.path.join(log_root, self.log_name)
            super(Logger, self).__init__(log_dir, flush_secs=1)
        

def console_logger(log_root, logger_name) -> logging.Logger:
    log_file = logger_name + '.log'
    log_path = os.path.join(log_root, log_file)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger


if __name__ == '__main__':
    import math
    logger = Logger('./log/', 'test')

    nsamples = 100
    for i in range(nsamples):
        x = math.cos(2 * math.pi * i / nsamples)
        logger.add_scalar('x', x, i)
