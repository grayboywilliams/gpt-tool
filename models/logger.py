
import os
import logging

# Custom log levels
SUMMARY = 25

# Filters to only SUMMARY logs
class SummaryFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == SUMMARY
    
# Set up the summary log file path
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, '../summary.log')

def configure_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the minimum log level for the logger

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Create a console handler for displaying logs in the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Create a file handler for saving logs to a file (summary.log)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SummaryFilter()) # Add the filter

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def reset_summary_log():
    open(file_path, 'w').close()

def save_summary_log(name):
    checkpoint_path = os.path.join(script_dir, '../checkpoints', name, 'summary.log')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    with open(file_path, "r") as source_file:
        source_contents = source_file.read()

    with open(checkpoint_path, "a") as destination_file:
        destination_file.write(source_contents)

    reset_summary_log()
