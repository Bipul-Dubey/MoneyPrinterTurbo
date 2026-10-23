# -*- coding: utf-8 -*-
import base64
import io
import os
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import requests
from PIL import Image

from app.config import config
from app.services import material


def _png_bytes(width=64, height=96, color=(120, 40, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_response(payload, status_code=200):
    return SimpleNamespace(json=lambda: payload, status_code=status_code)


def _download_response(content, status_code=200):
    return SimpleNamespace(
        status_code=status_code,
        content=content,
        headers={"Content-Length": str(len(content))},
        iter_content=lambda chunk_size: iter((content,)),
        close=lambda: None,
    )


class TestOpenAIImageProvider(unittest.TestCase):
    """
    OpenAI-compatible text-to-image material provider. Consistent with other material tests,
    all use unittest.mock to replace requests and time.sleep so CI does not depend on real
    networks, real API keys, or real billing.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)
        # Assertions must run after the generation call returns; temporary directory
        # cannot be prematurely cleaned up with a with-block, so use mkdtemp + addCleanup.
        self.save_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.save_dir, ignore_errors=True)
        config.app["openai_image_base_url"] = "https://img.example.com/v1"
        config.app["openai_image_api_keys"] = ["sk-test-key"]
        config.app["openai_image_model"] = "test-image-model"
        # Prompt template and custom size default to disabled; test cases that need overrides
        # configure them individually to prevent settings in local config.toml from affecting default tests.
        config.app.pop("openai_image_prompt_template", None)
        config.app.pop("openai_image_size", None)
        config.app.pop("tls_verify", None)
        config.proxy.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    @staticmethod
    def _generated_item(term, image_path, duration=5):
        item = material.MaterialInfo()
        item.provider = "openai_image"
        item.url = image_path
        item.duration = duration
        item.source_info = {
            "provider": "openai_image",
            "search_term": term,
            "rendition": {"id": None, "width": 736, "height": 1312},
        }
        return item

    # ------------------------------------------------------------------
    # Success path
    # ------------------------------------------------------------------

    def test_generate_images_openai_with_b64_json_response(self):
        """
        b64_json response must be decoded and saved as a valid PNG, and recorded in rendition
        with real image dimensions (compatible with relay services returning sizes different from request).
        duration records target segment duration.
        """
        image_data = _png_bytes(width=736, height=1312)
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            results = material.generate_images_openai(
                "sunrise over mountains",
                minimum_duration=5,
                video_aspect=material.VideoAspect.portrait,
                save_dir=self.save_dir,
            )

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.provider, "openai_image")
        self.assertEqual(item.duration, 5)
        # Request size uses OpenAI official compatible size per aspect ratio, not video resolution directly
        self.assertEqual(
            post.call_args.args[0],
            "https://img.example.com/v1/images/generations",
        )
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "model": "test-image-model",
                "prompt": "sunrise over mountains",
                "n": 1,
                "size": "1024x1536",
            },
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer sk-test-key",
        )
        # Saved file is a decodable PNG
        self.assertTrue(item.url.endswith(".png"))
        self.assertTrue(os.path.isfile(item.url))
        with Image.open(item.url) as saved:
            self.assertEqual(saved.size, (736, 1312))
        # rendition records true image dimensions, independent of request parameters
        self.assertEqual(
            item.source_info["rendition"],
            {"id": None, "width": 736, "height": 1312},
        )
        self.assertEqual(item.source_info["search_term"], "sunrise over mountains")

    def test_generate_images_openai_with_url_response(self):
        """url response must immediately download temporary URL and save to disk."""
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/abc.png?sig=1"}]}
        )
        download = _download_response(_png_bytes(width=200, height=300))

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.requests.get", return_value=download) as get,
        ):
            results = material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(os.path.isfile(results[0].url))
        # Temporary URL downloaded verbatim; signed query parameters must not be stripped
        self.assertEqual(
            get.call_args.args[0],
            "https://cdn.example.com/generated/abc.png?sig=1",
        )

    def test_generate_images_openai_skips_b64_json_with_invalid_image(self):
        """
        When compatibility layer returns 200 but body is not a decodable image (e.g., HTML
        error page disguised as JSON), it must return an empty list per material provider
        convention so upper layer skips the keyword, rather than crashing the whole task.
        """
        fake_content = b"<html><body>gateway degraded</body></html>"
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(fake_content).decode("ascii")}]}
        )

        with patch("app.services.material.requests.post", return_value=response):
            results = material.generate_images_openai(
                "sunrise over mountains", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_skips_url_download_with_invalid_content(self):
        """When temporary URL downloads 200 non-image content, it also follows skip path."""
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/abc.png?sig=1"}]}
        )
        download = _download_response(b"\x89PNG\r\n\x1a\nnot-really-a-png")

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.requests.get", return_value=download),
        ):
            results = material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_propagates_image_write_failure(self):
        """
        When image decodes successfully but PNG writing fails, the task must halt. Such
        failures typically persist across subsequent keywords; treating it as a single-image
        anomaly would produce unsavable paid requests.
        """
        response = _image_response(
            {
                "data": [
                    {"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}
                ]
            }
        )

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch.object(
                Image.Image,
                "save",
                side_effect=OSError("no space left on device"),
            ),
            self.assertRaisesRegex(OSError, "no space left on device"),
        ):
            material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(os.listdir(self.save_dir), [])

    # ------------------------------------------------------------------
    # Backoff retry and key rotation
    # ------------------------------------------------------------------

    def test_generate_images_openai_retries_429_with_backoff(self):
        """429 indicates temporary rate limiting; must back off and retry rather than failing the task."""
        image_data = _png_bytes()
        responses = [
            _image_response({"error": {"message": "rate limited"}}, status_code=429),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "ocean waves", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)
        # Must wait linear backoff before first retry; cannot immediately hammer remote endpoint
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(
            sleep.call_args.args[0],
            material.OPENAI_IMAGE_RETRY_BACKOFF_SECONDS[0],
        )

    def test_generate_images_openai_rotates_key_on_401(self):
        """
        401 indicates current key was rejected. When multiple keys are configured, retry must
        rotate to the next key using get_api_key rather than retrying the rejected key.
        """
        config.app["openai_image_api_keys"] = ["sk-bad-key", "sk-good-key"]
        image_data = _png_bytes()

        responses = [
            _image_response({"error": {"message": "unauthorized"}}, status_code=401),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "forest fog", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)
        used_keys = [
            call.kwargs["headers"]["Authorization"] for call in post.call_args_list
        ]
        # Consecutive requests must use different keys, both from configured list
        self.assertNotEqual(used_keys[0], used_keys[1])
        for auth in used_keys:
            self.assertIn(auth.replace("Bearer ", ""), ["sk-bad-key", "sk-good-key"])

    def test_generate_images_openai_fails_fast_on_401_with_single_key(self):
        """With only a single key, retrying on 401 is meaningless; must fail fast and return empty result."""
        response = _image_response(
            {"error": {"message": "unauthorized"}}, status_code=401
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "desert dunes", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_generate_images_openai_returns_empty_after_retries_exhausted(self):
        """
        After exhausting retries, return empty list per provider convention for upper layer to skip;
        must not leave leftover files on disk.
        """
        response = _image_response(
            {"error": {"message": "rate limited"}}, status_code=429
        )

        with (
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "storm clouds", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, material.OPENAI_IMAGE_MAX_ATTEMPTS)
        self.assertEqual(sleep.call_count, material.OPENAI_IMAGE_MAX_ATTEMPTS - 1)
        # No leftover files created
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_redacts_api_key_in_failure_detail(self):
        """Failure details must not write API key in plaintext into logs."""
        config.app["openai_image_api_keys"] = ["sk-secret-123"]
        response = _image_response(
            {"error": {"message": "invalid key sk-secret-123 provided"}},
            status_code=401,
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.logger") as logger,
        ):
            results = material.generate_images_openai(
                "redacted term", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        logged = [str(call) for call in logger.error.call_args_list]
        self.assertTrue(logged)
        for message in logged:
            self.assertNotIn("sk-secret-123", message)

    def test_generate_images_openai_retries_generated_image_download(self):
        """
        Images are billed per item; download jitter must retry the same URL, not fall back to
        regenerating the same image causing duplicate billing.
        """
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/x.png"}]}
        )
        downloads = [
            _download_response(b"", status_code=502),
            _download_response(_png_bytes()),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.requests.get", side_effect=downloads) as get,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "aurora", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        # Download retry hits the same address without triggering a second paid generation
        self.assertEqual(post.call_count, 1)
        self.assertEqual(get.call_count, 2)
        for call in get.call_args_list:
            self.assertEqual(call.args[0], "https://cdn.example.com/generated/x.png")

    def test_generated_image_download_rejects_oversize_body(self):
        """A generated image URL must not buffer an unbounded provider body."""
        response = _download_response(b"too-large")
        response.headers = {}
        closed = []
        response.close = lambda: closed.append(True)
        with (
            patch("app.services.material.requests.get", return_value=response) as get,
            patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5, create=True),
            patch("app.services.material.time.sleep"),
        ):
            image_bytes, error = material._openai_image_download_bytes(
                "https://cdn.example.com/generated/x.png", "test-key"
            )

        self.assertIsNone(image_bytes)
        self.assertIn("limit", error)
        self.assertTrue(all(call.kwargs.get("stream") is True for call in get.call_args_list))
        self.assertEqual(closed, [True])

    def test_generated_image_download_rejects_declared_oversize_body(self):
        """An advertised oversize image should be rejected before reading bytes."""
        response = _download_response(b"small")
        response.headers = {"Content-Length": "100"}
        response.iter_content = lambda chunk_size: self.fail("body should not be read")
        with (
            patch("app.services.material.requests.get", return_value=response),
            patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5),
        ):
            image_bytes, error = material._openai_image_download_bytes(
                "https://cdn.example.com/generated/x.png", "test-key"
            )

        self.assertIsNone(image_bytes)
        self.assertIn("limit", error)

    def test_generated_image_response_rejects_oversize_base64(self):
        """Do not decode an unbounded inline image from a provider response."""
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(b"too-large").decode()}]}
        )
        with patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5):
            image_bytes, error = material._parse_openai_image_response(response, "test-key")

        self.assertIsNone(image_bytes)
        self.assertIn("limit", error)

    def test_generated_image_decode_rejects_pillow_bomb_warning(self):
        """Small compressed bytes can still expand to too many image pixels."""
        image_bytes = _png_bytes(width=12, height=12)
        with (
            patch.object(Image, "MAX_IMAGE_PIXELS", 100),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with self.assertRaises(material._OpenAIImageDecodeError):
                material._save_openai_image_file(image_bytes, self.save_dir)

        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generated_image_download_failure_does_not_buy_another_image(self):
        """A confirmed paid result remains a task failure if its URL cannot download."""
        signed_url = "https://cdn.example.com/generated/x.png?token=private"
        response = _image_response(
            {"data": [{"url": signed_url}]}
        )
        with (
            patch("app.services.material.requests.post", return_value=response) as post,
            patch(
                "app.services.material.requests.get",
                side_effect=requests.exceptions.ConnectionError(
                    f"download failed for {signed_url}"
                ),
            ) as get,
            patch("app.services.material.time.sleep"),
            patch("app.services.material.logger") as logger,
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be downloaded"):
                material.download_videos(
                    task_id="test-openai-image-download-failure",
                    search_terms=["first", "second"],
                    source="openai_image",
                    audio_duration=10,
                    max_clip_duration=5,
                )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(get.call_count, material.OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS)
        self.assertNotIn("token=private", str(logger.warning.call_args_list))

    def test_generate_images_openai_returns_empty_on_rejected_request(self):
        """Business rejection (e.g. content policy) returns empty result without backoff retry."""
        response = _image_response(
            {"error": {"message": "content policy violation"}}, status_code=400
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "blocked term", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    # ------------------------------------------------------------------
    # Configuration switch
    # ------------------------------------------------------------------

    def test_is_openai_image_enabled_requires_full_configuration(self):
        """
        Both base_url and model are required; API key is allowed to be empty—purely local
        ComfyUI/SD gateways usually do not require authentication.
        """
        self.assertTrue(material.is_openai_image_enabled())

        config.app["openai_image_base_url"] = ""
        self.assertFalse(material.is_openai_image_enabled())
        config.app["openai_image_base_url"] = "https://img.example.com/v1"

        # Local unauthenticated gateway: considered enabled even without key
        config.app["openai_image_api_keys"] = []
        self.assertTrue(material.is_openai_image_enabled())
        config.app["openai_image_api_keys"] = ["sk-test-key"]

        config.app["openai_image_model"] = ""
        self.assertFalse(material.is_openai_image_enabled())

    def test_generate_images_openai_sends_no_authorization_without_key(self):
        """
        When API key is not configured, generation proceeds normally without Authorization header,
        suitable for unauthenticated local ComfyUI/SD gateways.
        """
        config.app["openai_image_api_keys"] = []
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with (
            patch("app.services.material.requests.post", return_value=response) as post,
        ):
            results = material.generate_images_openai(
                "local gateway", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])

    def test_generate_images_openai_retries_connect_timeout(self):
        """
        Connection timeout indicates request did not reach the server; no billing task could
        have been created, so backoff retry is allowed.
        """
        image_data = _png_bytes()
        responses = [
            requests.exceptions.ConnectTimeout("connect timed out"),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "connect timeout term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)

    def test_generate_images_openai_does_not_retry_unconfirmed_errors(self):
        """
        Read timeout/connection drop is an "unconfirmed" state: server may have generated and billed,
        but response was lost. Automatic resubmission causes duplicate generation and billing; must
        fail and abort task immediately without submitting paid requests for subsequent keywords.
        """
        for error in (
            requests.exceptions.ReadTimeout("read timed out"),
            requests.exceptions.ConnectionError("connection dropped"),
        ):
            with self.subTest(error=type(error).__name__):
                with (
                    patch(
                        "app.services.material.requests.post", side_effect=error
                    ) as post,
                    patch("app.services.material.time.sleep") as sleep,
                ):
                    with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                        material.generate_images_openai(
                            "unconfirmed term",
                            minimum_duration=5,
                            save_dir=self.save_dir,
                        )

                self.assertEqual(post.call_count, 1)
                sleep.assert_not_called()

    def test_download_videos_openai_image_stops_after_unconfirmed_paid_request(self):
        """Earlier images must not hide a later ambiguous paid submission."""
        image_response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )
        config.app["material_directory"] = self.save_dir

        with (
            patch(
                "app.services.material.requests.post",
                side_effect=[
                    image_response,
                    requests.exceptions.ReadTimeout("response lost"),
                    image_response,
                ],
            ) as post,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered.mp4",
            ),
            patch("app.services.material._persist_material_sources"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                material.download_videos(
                    task_id="test-openai-image-unconfirmed",
                    search_terms=["first", "uncertain", "third"],
                    source="openai_image",
                    audio_duration=20,
                    max_clip_duration=5,
                )

        self.assertEqual(post.call_count, 2)

    def test_generate_images_openai_size_defaults_and_override(self):
        """
        Default size uses OpenAI official compatible size per aspect ratio (portrait 1024x1536 /
        landscape 1536x1024); setting openai_image_size completely overrides it for local
        gateways supporting arbitrary resolutions.
        """
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as default_post:
            material.generate_images_openai(
                "landscape term",
                minimum_duration=5,
                video_aspect=material.VideoAspect.landscape,
                save_dir=self.save_dir,
            )
        # Landscape defaults to OpenAI official compatible size
        self.assertEqual(default_post.call_args.kwargs["json"]["size"], "1536x1024")

        config.app["openai_image_size"] = "1080x1920"
        self.addCleanup(config.app.pop, "openai_image_size", None)

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "custom size term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["size"],
            "1080x1920",
        )

    def test_generate_images_openai_raises_without_base_url(self):
        """When called directly without base_url configured, must raise error with config guidance."""
        config.app["openai_image_base_url"] = ""
        with self.assertRaises(ValueError):
            material.generate_images_openai("term", minimum_duration=5)

    # ------------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------------

    def test_generate_images_openai_applies_prompt_template(self):
        """
        When template containing {term} placeholder is configured, request prompt must be result
        of template substitution, uniformly attaching style modifiers to improve text-image matching.
        """
        config.app["openai_image_prompt_template"] = (
            "cinematic photo of {term}, photorealistic, high detail"
        )
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "晨光中的玻璃杯", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["prompt"],
            "cinematic photo of 晨光中的玻璃杯, photorealistic, high detail",
        )

    def test_generate_images_openai_sends_raw_term_without_template(self):
        """When no template is configured, prompt must be original search term, preserving legacy behavior."""
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "raw term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(post.call_args.kwargs["json"]["prompt"], "raw term")

    def test_generate_images_openai_falls_back_when_template_lacks_placeholder(self):
        """When template lacks {term} placeholder, cannot inject keyword; must fall back to original text."""
        config.app["openai_image_prompt_template"] = "no placeholder here"
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "fallback term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(post.call_args.kwargs["json"]["prompt"], "fallback term")

    # ------------------------------------------------------------------
    # download_videos dispatch and on-demand generation
    # ------------------------------------------------------------------

    def test_download_videos_openai_image_generates_on_demand_and_stops(self):
        """
        Text-to-image is billed per image and cannot generate for all keywords upfront. Materials
        must be generated on demand per image; once cumulative valid duration (capped per segment)
        reaches required audio duration, subsequent keywords must not trigger paid requests.
        """
        generated = {
            "term-1": [self._generated_item("term-1", "/tmp/img-1.png")],
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        def fake_render(image_path, clip_duration, image_motion="zoom"):
            return f"{image_path}.mp4"

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                side_effect=fake_render,
            ) as render,
        ):
            result = material.download_videos(
                task_id="test-openai-image-lazy",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=8,
                max_clip_duration=5,
            )

        # 5s + 5s > 8s; third keyword must not trigger paid generation request
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            [call.kwargs["search_term"] for call in generate.call_args_list],
            ["term-1", "term-2"],
        )
        # Each image is counted towards duration only after rendering into mp4 clip
        self.assertEqual(render.call_count, 2)
        self.assertEqual(result, ["/tmp/img-1.png.mp4", "/tmp/img-2.png.mp4"])

    def test_download_videos_openai_image_continues_after_invalid_image(self):
        """
        When first compatible API response cannot be decoded, skip that keyword; subsequent valid
        image can still be saved and rendered, verifying fix covers the on-demand call chain.
        """
        invalid_response = _image_response(
            {
                "data": [
                    {
                        "b64_json": base64.b64encode(
                            b"<html>gateway degraded</html>"
                        ).decode("ascii")
                    }
                ]
            }
        )
        valid_response = _image_response(
            {
                "data": [
                    {"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}
                ]
            }
        )
        config.app["material_directory"] = self.save_dir

        with (
            patch(
                "app.services.material.requests.post",
                side_effect=[invalid_response, valid_response],
            ) as post,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered-openai-image.mp4",
            ) as render,
            patch("app.services.material._persist_material_sources"),
        ):
            result = material.download_videos(
                task_id="test-openai-image-invalid-then-valid",
                search_terms=["invalid term", "valid term"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(render.call_count, 1)
        self.assertEqual(result, ["/tmp/rendered-openai-image.mp4"])

    def test_download_videos_openai_image_stops_when_duration_exactly_covered(self):
        """Boundary regression: exactly covering required duration is sufficient; stop condition must be >= not >."""
        generated = {
            "term-1": [self._generated_item("term-1", "/tmp/img-1.png")],
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered.mp4",
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-exact",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=10,
                max_clip_duration=5,
            )

        # 5s + 5s == 10s, exact coverage; 3rd segment must not be generated
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(result), 2)

    def test_download_videos_openai_image_bypasses_search_cache(self):
        """
        Generated results are one-off image files and do not participate in 24-hour search cache—
        caching causes different tasks to repeatedly get the same image. download_videos must
        directly enter on-demand generation branch.
        """
        with (
            patch(
                "app.services.material.generate_images_openai",
                return_value=[self._generated_item("sunrise", "/tmp/img-1.png")],
            ) as generate,
            patch("app.services.material._search_videos_with_cache") as cached_search,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/img-1.png.mp4",
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-cache-bypass",
                search_terms=["sunrise"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(generate.call_count, 1)
        cached_search.assert_not_called()
        self.assertEqual(result, ["/tmp/img-1.png.mp4"])

    def test_download_videos_openai_image_skips_rejected_segment(self):
        """
        When explicitly rejected generation result is empty, skip that keyword and continue to next segment.
        """
        generated = {
            "term-1": [],  # Generation failed
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        def fake_render(image_path, clip_duration, image_motion="zoom"):
            return f"{image_path}.mp4"

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                side_effect=fake_render,
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-skip",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result, ["/tmp/img-2.png.mp4"])

    def test_download_videos_openai_image_stops_after_render_failure(self):
        """A paid image saved locally should not trigger another purchase if render fails."""
        generated_item = self._generated_item("first", "/tmp/img-1.png")
        with (
            patch(
                "app.services.material.generate_images_openai",
                return_value=[generated_item],
            ) as generate,
            patch("app.services.material._render_openai_image_video", return_value=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be rendered"):
                material.download_videos(
                    task_id="test-openai-image-render-failure",
                    search_terms=["first", "second"],
                    source="openai_image",
                    audio_duration=10,
                    max_clip_duration=5,
                )

        self.assertEqual(generate.call_count, 1)

    def test_download_videos_openai_image_skips_generation_without_audio(self):
        """When audio duration is non-positive, return empty immediately; do not pay per image for unfulfillable tasks."""
        with patch("app.services.material.generate_images_openai") as generate:
            result = material.download_videos(
                task_id="test-openai-image-no-audio",
                search_terms=["term-1"],
                source="openai_image",
                audio_duration=0,
                max_clip_duration=5,
            )

        generate.assert_not_called()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
