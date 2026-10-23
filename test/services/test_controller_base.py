import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from app.config import config
from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException


class TestControllerAuthentication(unittest.TestCase):
    generated_task_id = UUID("00000000-0000-4000-8000-000000000001")

    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    @staticmethod
    def _request(headers=None):
        return SimpleNamespace(
            headers=headers or {},
            url="http://localhost/api/v1/tasks",
        )

    def test_normalize_task_id_preserves_printable_values_up_to_limit(self):
        task_ids = (
            "request-123",
            "trace/01HZX_abc.def:456",
            "请求-123",
            "x" * base.MAX_TASK_ID_LENGTH,
        )

        for task_id in task_ids:
            with self.subTest(task_id=task_id):
                self.assertEqual(base.normalize_task_id(task_id), task_id)

    def test_normalize_task_id_replaces_unsafe_or_malformed_values(self):
        unsafe_values = (
            None,
            "",
            123,
            b"request-123",
            object(),
            "line\nforged",
            "line\rforged",
            "column\tforged",
            "ansi\x1b[31m",
            "unicode\u2028separator",
            "x" * (base.MAX_TASK_ID_LENGTH + 1),
        )

        with patch.object(base, "uuid4", return_value=self.generated_task_id):
            for value in unsafe_values:
                with self.subTest(value=value):
                    self.assertEqual(
                        base.normalize_task_id(value), str(self.generated_task_id)
                    )

    def test_get_task_id_reuses_safe_header_or_generates_uuid(self):
        """
        When the client provides a request ID, preserve it as-is; when absent, generate a UUID
        recordable in logs and error responses, ensuring both entry paths have traceable identifiers.
        """
        self.assertEqual(
            base.get_task_id(self._request({"x-task-id": "request-123"})),
            "request-123",
        )

        with patch.object(base, "uuid4", return_value=self.generated_task_id):
            generated = base.get_task_id(self._request())

        self.assertEqual(generated, str(self.generated_task_id))

    def test_verify_token_never_exposes_unsafe_task_id(self):
        config.app["api_key"] = "secret"
        malicious_task_id = "attacker\nforged-log-entry"

        with (
            patch.object(base, "uuid4", return_value=self.generated_task_id),
            patch("app.models.exception.logger.warning") as log_warning,
        ):
            with self.assertRaises(HttpException):
                base.verify_token(
                    self._request(
                        {
                            "x-api-key": "wrong",
                            "x-task-id": malicious_task_id,
                        }
                    )
                )

        logged_warning = log_warning.call_args.args[0]
        self.assertIn(str(self.generated_task_id), logged_warning)
        self.assertNotIn(malicious_task_id, logged_warning)
        self.assertNotIn("forged-log-entry", logged_warning)

    def test_verify_token_accepts_matching_key(self):
        """When API Key is configured, identical request headers must pass authentication normally."""
        config.app["api_key"] = "secret"

        result = base.verify_token(self._request({"x-api-key": "secret"}))

        self.assertIsNone(result)

    def test_verify_token_allows_requests_when_key_is_not_configured(self):
        """When Key is unconfigured, retain legacy unauthenticated behavior to prevent breakage after local upgrade."""

        config.app.pop("api_key", None)
        self.assertIsNone(base.verify_token(self._request()))

        for configured_key in (None, ""):
            with self.subTest(configured_key=configured_key):
                config.app["api_key"] = configured_key
                self.assertIsNone(base.verify_token(self._request()))

    def test_verify_token_rejects_missing_or_wrong_key(self):
        """
        Both missing and incorrect API Keys must return 401 while preserving the client request ID,
        preventing auth failures from becoming uncorrelatable with caller requests in logs.
        """
        config.app["api_key"] = "secret"

        for provided_key in (None, "wrong"):
            with self.subTest(provided_key=provided_key):
                headers = {"x-task-id": "auth-request"}
                if provided_key is not None:
                    headers["x-api-key"] = provided_key

                with self.assertRaises(HttpException) as raised:
                    base.verify_token(self._request(headers))

                self.assertEqual(raised.exception.status_code, 401)
                self.assertEqual(raised.exception.message, "invalid API key")

    def test_verify_token_rejects_non_string_configuration(self):
        """Non-string configuration should raise an explicit error without exposing config content in the error."""

        config.app["api_key"] = ["unexpected", "value"]

        with self.assertRaises(HttpException) as raised:
            base.verify_token(self._request())

        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(
            raised.exception.message,
            "API authentication is misconfigured",
        )

    def test_verify_token_handles_unicode_without_server_error(self):
        """Non-ASCII headers must not trigger compare_digest TypeError or return 500."""

        config.app["api_key"] = "密钥-é"
        self.assertIsNone(base.verify_token(self._request({"x-api-key": "密钥-é"})))

        with self.assertRaises(HttpException) as raised:
            base.verify_token(self._request({"x-api-key": "错误-é"}))

        self.assertEqual(raised.exception.status_code, 401)

    def test_new_router_preserves_common_prefix_and_dependencies(self):
        """All V1 routers should reuse a common prefix and only set auth dependencies when provided."""
        dependency = object()

        plain_router = new_router()
        protected_router = new_router(dependencies=[dependency])

        self.assertEqual(plain_router.prefix, "/api/v1")
        self.assertEqual(plain_router.tags, ["V1"])
        self.assertEqual(protected_router.dependencies, [dependency])


if __name__ == "__main__":
    unittest.main()
