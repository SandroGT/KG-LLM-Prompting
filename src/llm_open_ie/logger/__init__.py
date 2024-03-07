LOGGER = None


def init():
    import logging
    from logging.config import fileConfig
    from pathlib import Path

    global LOGGER

    logger_config_file = Path(__file__).parent.joinpath('logging.ini').resolve()
    fileConfig(logger_config_file)
    LOGGER = logging.getLogger()
    LOGGER.debug(f'Loaded LOGGER from {logger_config_file}')


init()
