from utils.logger_util import CustomLogger


def log_args(args_dict):
    CustomLogger.info("-" * 10, "Arguments:", "-" * 10)
    for k, v in args_dict.items():
        CustomLogger.info(f"{k}: {v}")
    CustomLogger.info("-" * 10, "End of Arguments", "-" * 10)
