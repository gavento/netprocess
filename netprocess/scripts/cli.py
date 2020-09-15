import logging

import click
import colorlog
import tqdm

from .. import utils

log = logging.getLogger(__name__)


class TQDMLogStream:
    def write(self, msg):
        tqdm.tqdm.write(msg, end="")
        return len(msg)

    def flush(self):
        tqdm.tqdm.write("", end="")


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--debug", "-d", is_flag=True, help="Increase logging verbosity.")
def cli(debug):
    handler = colorlog.StreamHandler(TQDMLogStream())
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(thin_white)s%(asctime)s%(reset)s [%(log_color)s%(levelname).1s%(reset)s] %(message)s"
        )
    )
    logger = colorlog.getLogger()
    logger.handlers = [handler]
    logger.setLevel(level=logging.DEBUG if debug else logging.INFO)
