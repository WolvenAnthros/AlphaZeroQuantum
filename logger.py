import logging


class CustomFormatter(logging.Formatter):
    colours = {
        'gray': "\u001b[37m",
        'yellow': "\u001b[33",
        'red': "\u001b[31m",
        'cyan': "\u001b[36m",
        'magenta': "\u001b[35m",
        'green': "\u001b[32m",
        'bold_green': "\u001b[32;1m",
        'bold_magenta': "\u001b[35;1m",
        'bold_yellow': "\u001b[33;1m",
        'bold_red': "\u001b[31;1m",
        'bold_cyan': "\u001b[36;1m",
        'bold_gray': "\u001b[37;1m"
    }
    reset = "\u001b[0m"
    format1 = "%(message)s"
    # format1 = "%(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # format = f'\33[1m\33[31m{"%(levelname)s: %(message)s"}\33[0m'

    FORMATS = {
        logging.DEBUG: colours['bold_yellow'] + format1 + reset,
        logging.INFO: colours['bold_gray'] + format1 + reset,
        logging.WARNING: colours['bold_magenta'] + format1 + reset,
        logging.ERROR: colours['bold_cyan'] + format1 + reset,
        logging.CRITICAL: colours['bold_red'] + format1 + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# create logger with 'spam_application'
logger = logging.getLogger("coloured-logger")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
