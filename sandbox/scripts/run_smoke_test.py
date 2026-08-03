import logging
from pathlib import Path

from bankscope.config.settings import get_settings
from bankscope.observability.logging import configure_logging


def ensure_directories(directories: tuple[Path, ...]) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

        if not directory.is_dir():
            raise RuntimeError(f"Direktorijum nije dostupan: {directory}")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)

    required_directories = (
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.manifest_dir,
        Path("artifacts/logs"),
    )

    ensure_directories(required_directories)

    logger.info("Application settings loaded successfully")
    logger.info("Verified %d project directories", len(required_directories))
    logger.info("BankScope smoke test passed")


if __name__ == "__main__":
    main()
