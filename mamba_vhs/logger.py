import os
import logging
from datetime import datetime

def setup_logger():
    """
    Set up the logger and create necessary directories for saving logs and models.
    Returns:
        logger (logging.Logger): Configured logger instance.
        folder_name (str): Name of the main folder created for saving logs and models.
        log_file_name (str): Path to the log file.
        loss_file_name (str): Path to the loss file.
        vhs_score_file (str): Path to the VHS score file.
        best_val_model (str): Path to save the best validation model.
        last_model (str): Path to save the last model.
    """
    # Get the current timestamp
    current_timestamp = datetime.now()
    folder_name = current_timestamp.strftime("%Y%m%d_%H%M%S")

    # Create main folder and subdirectories
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f"{folder_name}/predictions", exist_ok=True)
    os.makedirs(f"{folder_name}/models", exist_ok=True)

    # Define file paths
    log_file_name = os.path.join(folder_name, f"training_log.txt")
    loss_file_name = os.path.join(folder_name, f"loss.csv")
    vhs_score_file = os.path.join(folder_name, f"vhs_score.csv")
    best_val_model = os.path.join(folder_name, f"models/best_model.pth")
    last_model = os.path.join(folder_name, f"models/last_model.pth")

    # Print paths for confirmation
    print("Current folder:", folder_name)

    # Create the logger
    logger = logging.getLogger('dual_logger')
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Create custom filter for info logs
    class InfoOnlyFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.INFO

    # File handler for general info logs
    info_handler = logging.FileHandler(f'{folder_name}/info_log.txt')
    info_handler.setLevel(logging.INFO)
    info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(info_formatter)
    info_handler.addFilter(InfoOnlyFilter())

    # File handler for warnings/errors only
    warn_handler = logging.FileHandler(f'{folder_name}/warning_log.txt')
    warn_handler.setLevel(logging.WARNING)
    warn_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    warn_handler.setFormatter(warn_formatter)

    # Add handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(warn_handler)

    # Get script path and current date-time
    script_path = os.path.abspath(__file__)
    current_time = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()

    # Log script start
    logger.info(f"Script started: {script_path} at {current_time}, PID: {pid}")
    logger.warning(f"Script started: {script_path} at {current_time}, PID: {pid}")

    # Return the logger and file paths for further use
    return logger, folder_name, log_file_name, loss_file_name, vhs_score_file, best_val_model, last_model

