from __future__ import annotations

from loguru import logger

logger.add("logs/newsensor_{time}.log", rotation="500 MB", level="INFO")