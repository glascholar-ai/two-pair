"""Telegram notifications. Silent no-op when credentials are absent."""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


class Notifier:
    """Sends messages to a Telegram chat; degrades to logging when unset."""

    def __init__(self, token: str, chat_id: str) -> None:
        self._token = token
        self._chat_id = chat_id

    @property
    def enabled(self) -> bool:
        """Whether real Telegram delivery is configured."""
        return bool(self._token and self._chat_id)

    def send(self, text: str) -> bool:
        """Sends a message; returns True on confirmed delivery.

        Failures are logged, never raised — notification must not be able to
        take down the trading loop.
        """
        if not self.enabled:
            logger.info("notify (telegram off): %s", text)
            return False
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        body = urllib.parse.urlencode(
            {"chat_id": self._chat_id, "text": text}).encode()
        try:
            with urllib.request.urlopen(url, data=body, timeout=10) as resp:
                return bool(json.load(resp).get("ok"))
        except Exception as err:  # noqa: BLE001 — deliberately broad
            logger.error("telegram send failed: %s", err)
            return False
