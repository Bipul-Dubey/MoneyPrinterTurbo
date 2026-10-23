import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient

from app import asgi


class TestASGICORS(unittest.TestCase):
    """Verify browser cross-origin defaults and explicit compatibility configurations to avoid reintroducing open CORS."""

    @staticmethod
    def _create_client(allowed_origins: list[str]) -> TestClient:
        """Construct an application containing only probe routes to isolate business tasks and external API calls."""

        application = FastAPI()

        @application.get("/probe")
        def probe():
            return {"status": "ok"}

        asgi.configure_browser_access(application, allowed_origins)
        return TestClient(application)

    def test_origin_parser_trims_values_and_ignores_empty_items(self):
        """Whitespace and trailing commas in environment variables should not break valid origin matching."""

        origins = asgi.parse_cors_allowed_origins(
            " https://a.example,https://b.example, ,"
        )

        self.assertEqual(
            origins,
            ["https://a.example", "https://b.example"],
        )
        self.assertEqual(asgi.parse_cors_allowed_origins(""), [])
        self.assertEqual(asgi.parse_cors_allowed_origins(None), [])

    def test_origin_parser_normalizes_every_item_to_origin_header_shape(self):
        """Config items must be normalized to canonical Origin request header form, otherwise the allowlist silently fails."""

        origins = asgi.parse_cors_allowed_origins(
            "https://frontend.example/,"
            "HTTPS://Admin.Example,"
            "https://frontend.example/,"
            "localhost:3000,"
            "ftp://files.example,"
            "*"
        )

        # Trailing slashes, case differences, and duplicates are normalized to the same origin; hostnames without scheme
        # and non-http/https schemes can never appear in an Origin header and are thus discarded.
        self.assertEqual(
            origins,
            ["https://frontend.example", "https://admin.example", "*"],
        )

    def test_malformed_entry_is_dropped_without_losing_valid_origins(self):
        """Malformed entries must only drop themselves and log a warning without interrupting the parsing of valid origins."""

        with patch.object(asgi, "logger") as mocked_logger:
            origins = asgi.parse_cors_allowed_origins(
                "https://valid.example,"
                "https://[::1,"
                "https://frontend.example:bad,"
                "https://frontend.example:99999,"
                "https://second.example"
            )

        # Malformed IPv6 literals and invalid ports can never appear in an Origin header. Parsing exceptions
        # must be caught here: this function is called during module import; uncaught exceptions would prevent API startup.
        self.assertEqual(
            origins,
            ["https://valid.example", "https://second.example"],
        )
        self.assertEqual(mocked_logger.warning.call_count, 3)

    def test_explicit_default_port_is_folded_into_the_origin(self):
        """Explicitly written default ports must be folded into the actual Origin sent by browsers."""

        origins = asgi.parse_cors_allowed_origins(
            "https://secure.example:443,"
            "http://plain.example:80,"
            "https://padded.example:0443,"
            "https://custom.example:8443,"
            "https://[::1]:3000"
        )

        # :443 / :0443 / :80 equal protocol default ports, which browsers omit when serializing Origin;
        # non-default ports must be retained, and IPv6 literals retained with square brackets.
        self.assertEqual(
            origins,
            [
                "https://secure.example",
                "http://plain.example",
                "https://padded.example",
                "https://custom.example:8443",
                "https://[::1]:3000",
            ],
        )

    def test_address_bar_style_origin_admits_the_trusted_frontend(self):
        """Trailing-slash origins copied from address bars must yield the same frontend access result as canonical origins."""

        trusted_origin = "https://frontend.example"
        client = self._create_client(
            asgi.parse_cors_allowed_origins("https://frontend.example/")
        )

        response = client.get("/probe", headers={"Origin": trusted_origin})
        preflight = client.options(
            "/probe",
            headers={
                "Origin": trusted_origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"], trusted_origin
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers["access-control-allow-origin"], trusted_origin
        )

    def test_default_port_configuration_admits_the_trusted_frontend(self):
        """When default port is configured, browser origins without explicit ports must still be admitted."""

        trusted_origin = "https://frontend.example"
        client = self._create_client(
            asgi.parse_cors_allowed_origins("https://frontend.example:443")
        )

        response = client.get("/probe", headers={"Origin": trusted_origin})
        preflight = client.options(
            "/probe",
            headers={
                "Origin": trusted_origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"], trusted_origin
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers["access-control-allow-origin"], trusted_origin
        )

    def test_empty_configuration_keeps_browser_same_origin_policy(self):
        """When allowlist is unconfigured, third-party web pages cannot read responses or pass preflight."""

        client = self._create_client([])
        origin = "https://evil.attacker.example"

        response = client.get("/probe", headers={"Origin": origin})
        preflight = client.options(
            "/probe",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertNotIn("access-control-allow-credentials", response.headers)
        self.assertEqual(preflight.status_code, 403)
        self.assertNotIn("access-control-allow-origin", preflight.headers)

    def test_same_origin_and_server_clients_remain_compatible(self):
        """Same-origin browsers and server-side clients that do not send Origin must continue accessing normally."""

        client = self._create_client([])

        same_origin = client.get(
            "/probe",
            headers={"Origin": "http://testserver"},
        )
        server_client = client.get("/probe")

        self.assertEqual(same_origin.status_code, 200)
        self.assertEqual(server_client.status_code, 200)

    def test_explicit_origin_allows_only_the_trusted_frontend(self):
        """After explicit configuration, independent web frontend can access while other origins are rejected."""

        trusted_origin = "https://frontend.example"
        untrusted_origin = "https://evil.attacker.example"
        client = self._create_client([trusted_origin])

        trusted = client.get("/probe", headers={"Origin": trusted_origin})
        untrusted = client.get("/probe", headers={"Origin": untrusted_origin})
        trusted_preflight = client.options(
            "/probe",
            headers={
                "Origin": trusted_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        untrusted_preflight = client.options(
            "/probe",
            headers={
                "Origin": untrusted_origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(trusted.headers["access-control-allow-origin"], trusted_origin)
        self.assertEqual(trusted.headers["access-control-allow-credentials"], "true")
        self.assertNotIn("access-control-allow-origin", untrusted.headers)
        self.assertEqual(trusted_preflight.status_code, 200)
        self.assertEqual(untrusted_preflight.status_code, 400)

    def test_trusted_origin_can_request_private_network_access(self):
        """Exact allowlist should support additional preflight for remote web pages accessing localhost or LAN APIs."""

        trusted_origin = "https://frontend.example"
        client = self._create_client([trusted_origin])

        preflight = client.options(
            "/probe",
            headers={
                "Origin": trusted_origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Private-Network": "true",
            },
        )

        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers["access-control-allow-private-network"],
            "true",
        )

    def test_explicit_wildcard_does_not_enable_credentials(self):
        """Explicit wildcard preserves compatibility capability, but must not re-enable reflected Origin combinations."""

        client = self._create_client(["*"])
        origin = "https://frontend.example"

        response = client.get("/probe", headers={"Origin": origin})
        preflight = client.options(
            "/probe",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.headers["access-control-allow-origin"], "*")
        self.assertNotIn("access-control-allow-credentials", response.headers)
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers["access-control-allow-origin"], "*")
        self.assertNotIn("access-control-allow-credentials", preflight.headers)

    def test_untrusted_multipart_request_is_rejected_before_side_effect(self):
        """Preflight-free multipart requests must also return 403 before reaching upload handler."""

        application = FastAPI()
        save_upload = Mock(return_value="stored.mp3")

        @application.post("/upload")
        def upload(file: UploadFile = File(...)):
            return {"file": save_upload(file.filename)}

        asgi.configure_browser_access(application, [])
        client = TestClient(application)

        response = client.post(
            "/upload",
            headers={"Origin": "https://evil.attacker.example"},
            files={
                "file": (
                    "attack.mp3",
                    b"attacker-controlled",
                    "audio/mpeg",
                )
            },
        )

        self.assertEqual(response.status_code, 403)
        save_upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
