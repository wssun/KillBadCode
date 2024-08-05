import logging
import config

logger = logging.getLogger('curator')   # choose some name for the loggger

logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter('[%(levelname)s] %(asctime)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
