import errno
import socket
import traceback
import unittest
from urllib import error
from unittest.mock import patch

from moneygraph.assistant import AssistantError, http_json


class ConnectionErrorTests(unittest.TestCase):
    openai_url = "https://api.openai.com/v1/responses"
    secret = "private-test-token-should-never-appear"

    def failure(self, url, failure):
        with patch("urllib.request.OpenerDirector.open", side_effect=failure):
            try:
                http_json(url, {"input": "synthetic"}, {"Authorization": "Bearer " + self.secret})
            except AssistantError as exc:
                self.assertNotIn(self.secret, str(exc))
                self.assertNotIn(self.secret, traceback.format_exc())
                self.assertNotIn(url, str(exc))
                self.assertTrue(exc.__suppress_context__)
                return exc
        self.fail("A transport failure must raise AssistantError")

    def test_openai_connection_failure_never_suggests_ollama(self):
        exc = self.failure(self.openai_url, error.URLError(self.secret))
        self.assertEqual((exc.code, exc.status), ("unavailable", 503))
        self.assertIn("OpenAI", str(exc))
        self.assertNotIn("Ollama", str(exc))

    def test_local_connection_failures_identify_ollama(self):
        for host in ("localhost", "127.0.0.1", "[::1]"):
            with self.subTest(host=host):
                exc = self.failure("http://" + host + ":11434/api/chat", error.URLError(ConnectionRefusedError(self.secret)))
                self.assertEqual((exc.code, exc.status), ("unavailable", 503))
                self.assertIn("Ollama", str(exc))
                self.assertIn("OLLAMA_URL", str(exc))
                self.assertNotIn("OpenAI", str(exc))

    def test_direct_and_wrapped_permissions_are_distinguished(self):
        windows_denial = OSError(self.secret)
        windows_denial.winerror = 10013
        reasons = (PermissionError(errno.EACCES, self.secret),
                   OSError(errno.EPERM, self.secret), OSError(10013, self.secret), windows_denial)
        for reason in reasons:
            for failure in (reason, error.URLError(reason)):
                with self.subTest(reason=type(reason).__name__, wrapped=isinstance(failure, error.URLError)):
                    exc = self.failure(self.openai_url, failure)
                    self.assertEqual((exc.code, exc.status), ("network_denied", 503))
                    self.assertIn("операционной системой", str(exc))
                    self.assertIn("OpenAI", str(exc))
                    self.assertNotIn("Ollama", str(exc))

    def test_direct_and_wrapped_timeouts_are_distinguished(self):
        windows_timeout = OSError(self.secret)
        windows_timeout.winerror = 10060
        for reason in (TimeoutError(self.secret), socket.timeout(self.secret),
                       OSError(errno.ETIMEDOUT, self.secret), windows_timeout):
            for failure in (reason, error.URLError(reason)):
                with self.subTest(reason=type(reason).__name__, wrapped=isinstance(failure, error.URLError)):
                    exc = self.failure(self.openai_url, failure)
                    self.assertEqual((exc.code, exc.status), ("timeout", 504))
                    self.assertIn("время ожидания", str(exc))
                    self.assertIn("OpenAI", str(exc))
                    self.assertNotIn("Ollama", str(exc))

    def test_local_permission_and_timeout_messages_identify_ollama(self):
        for failure in (error.URLError(PermissionError(self.secret)), error.URLError(TimeoutError(self.secret))):
            with self.subTest(failure=type(failure.reason).__name__):
                exc = self.failure("http://127.0.0.1:11434/api/chat", failure)
                self.assertIn("локальной Ollama", str(exc))
                self.assertNotIn("OpenAI", str(exc))

    def test_response_read_connection_failure_is_sanitized(self):
        with patch("urllib.request.OpenerDirector.open") as opened:
            opened.return_value.__enter__.return_value.read.side_effect = ConnectionResetError(self.secret)
            try:
                http_json(self.openai_url)
            except AssistantError as exc:
                self.assertEqual((exc.code, exc.status), ("unavailable", 503))
                self.assertNotIn(self.secret, traceback.format_exc())
            else:
                self.fail("A response read failure must raise AssistantError")


if __name__ == "__main__":
    unittest.main()
