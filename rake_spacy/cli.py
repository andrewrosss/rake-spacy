import argparse
import sys

from loguru import logger


def cli() -> None:
    parser = create_parser()
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    logger.debug("A debug level message")
    logger.info("An info level message")
    logger.warning("A warning level message")
    logger.error("An error level message")
    logger.critical("A critical level message")


def create_parser() -> argparse.ArgumentParser:
    description = "Extract keywords/keyphrases from text using the RAKE algorithm."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        action="store_const",
        default="WARNING",
        const="INFO",
        help="Be verbose.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const="DEBUG",
        help="A deluge of output.",
    )

    return parser
