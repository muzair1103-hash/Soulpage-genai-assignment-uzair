import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
