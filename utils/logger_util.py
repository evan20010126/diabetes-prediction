import os
import logging
from typing import Union
from torch.utils.tensorboard import SummaryWriter


class OnlyLevelFilter(logging.Filter):

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


class LevelBasedFormatter(logging.Formatter):

    def __init__(self,
                 fmt_info: str,
                 fmt_other: str,
                 datefmt: str = None) -> None:
        super().__init__(datefmt=datefmt)
        self.fmt_info = logging.Formatter(fmt_info, datefmt)
        self.fmt_other = logging.Formatter(fmt_other, datefmt)

    def format(self, record: logging.LogRecord) -> logging.Formatter:
        if record.levelno == logging.INFO:
            return self.fmt_info.format(record)
        else:
            return self.fmt_other.format(record)


class CustomLogger(object):

    GLOBAL_LOGGER: Union[logging.Logger, None] = None
    GLOBAL_TF_LOGGER: Union[SummaryWriter, None] = None
    ROOT: Union[str, None] = None

    def __init__(self) -> None:
        raise TypeError("This class cannot be instantiated.")

    @classmethod
    def configure(cls, log_root: str) -> None:
        cls.ROOT = log_root

        cls._configure_logger(log_root + "/logs")
        cls._configure_tf_logger(log_root + "/tf")

    @classmethod
    def get_root(cls) -> Union[str, None]:

        if cls.ROOT is None:
            raise RuntimeError(
                "Logger not configured. Call CustomLogger.configure() first.")

        return cls.ROOT

    @classmethod
    def log(cls, *args, level=logging.INFO) -> None:
        if cls.GLOBAL_LOGGER is None:
            raise RuntimeError(
                "Logger not configured. Call CustomLogger.configure() first.")
        msg = " ".join(str(a) for a in args)
        cls.GLOBAL_LOGGER.log(level, msg)

    @classmethod
    def debug(cls, *args) -> None:
        cls.log(*args, level=logging.DEBUG)

    @classmethod
    def info(cls, *args) -> None:
        cls.log(*args, level=logging.INFO)

    @classmethod
    def warn(cls, *args) -> None:
        cls.log(*args, level=logging.WARNING)

    @classmethod
    def error(cls, *args) -> None:
        cls.log(*args, level=logging.ERROR)

    @classmethod
    def add_scalar(cls, *args) -> None:
        if cls.GLOBAL_TF_LOGGER is None:
            raise RuntimeError(
                "TensorBoard logger not configured. Call CustomLogger.configure() first."
            )
        cls.GLOBAL_TF_LOGGER.add_scalar(*args)

    # @classmethod
    # def add_image(cls, *args) -> None:
    #     if dist.get_rank() != 0:
    #         return
    #     if cls.GLOBAL_TF_LOGGER is None:
    #         raise RuntimeError(
    #             "TensorBoard logger not configured. Call CustomLogger.configure() first."
    #         )
    #     cls.GLOBAL_TF_LOGGER.add_image(*args)

    @classmethod
    def _configure_logger(cls, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        cls.GLOBAL_LOGGER = logging.getLogger("app")
        cls.GLOBAL_LOGGER.setLevel(logging.DEBUG)

        formatter = LevelBasedFormatter(
            fmt_info="%(message)s",
            fmt_other="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        cls.GLOBAL_LOGGER.addHandler(console_handler)

        # info file handler
        info_log_path = os.path.join(log_dir, f"running.log")
        info_handler = logging.FileHandler(info_log_path)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        cls.GLOBAL_LOGGER.addHandler(info_handler)

        # debug only file handler
        debug_log_path = os.path.join(log_dir, f"debug.log")
        debug_handler = logging.FileHandler(debug_log_path)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.addFilter(OnlyLevelFilter(logging.DEBUG))
        debug_handler.setFormatter(formatter)
        cls.GLOBAL_LOGGER.addHandler(debug_handler)

    @classmethod
    def _configure_tf_logger(cls, log_dir: str) -> None:
        cls.GLOBAL_TF_LOGGER = SummaryWriter(log_dir)


if __name__ == "__main__":
    CustomLogger.configure("./logs")
    CustomLogger.debug("This is a debug message.")
    CustomLogger.info("This is an info message.")
    CustomLogger.warn("This is a warning message.")
    CustomLogger.error("This is an error message.")
