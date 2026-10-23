import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.models.schema import MaterialInfo, VideoAspect
from app.services import material, material_cache


class TestMaterialSearchCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir_patch = patch(
            "app.services.material_cache.utils.storage_dir",
            return_value=self.temp_dir.name,
        )
        self.cache_dir_patch.start()

    def tearDown(self):
        self.cache_dir_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _item(url: str = "https://example.com/video.mp4") -> MaterialInfo:
        return MaterialInfo(
            provider="pixabay",
            url=url,
            duration=12,
            source_info={
                "provider": "pixabay",
                "search_term": "nature",
                "asset_id": "123",
                "source_page": "https://pixabay.com/videos/example-123/",
                "creator": {
                    "id": "456",
                    "name": "Creator",
                    "profile_page": "https://pixabay.com/users/creator-456/",
                },
                "rendition": {
                    "id": "large",
                    "width": 1080,
                    "height": 1920,
                },
            },
        )

    def _cache_path(self) -> Path:
        return material_cache._cache_path(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

    def test_cache_round_trip_preserves_material_fields(self):
        """
        Disk cache must be able to restore all fields required by MaterialInfo across processes;
        it cannot merely cache the URL and lose provider or duration, which would alter subsequent download and duration calculations.
        """
        saved = material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertTrue(saved)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].provider, "pixabay")
        self.assertEqual(loaded[0].url, "https://example.com/video.mp4")
        self.assertEqual(loaded[0].duration, 12)
        self.assertEqual(loaded[0].source_info["search_term"], "nature")
        self.assertEqual(loaded[0].source_info["asset_id"], "123")
        self.assertEqual(
            loaded[0].source_info["source_page"],
            "https://pixabay.com/videos/example-123/",
        )
        self.assertEqual(
            loaded[0].source_info["creator"]["profile_page"],
            "https://pixabay.com/users/creator-456/",
        )

    def test_expired_cache_is_removed_and_treated_as_miss(self):
        """
        Pixabay requires search results to be reused for at most 24 hours. Expired files must immediately become invalid and deleted,
        preventing stale material URLs from indefinite reuse and preventing the cache directory from accumulating invalid JSON.
        """
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        cache_path = self._cache_path()
        now = 2_000_000_000.0
        expired_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(cache_path, (expired_mtime, expired_mtime))

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            now=now,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_future_dated_cache_is_removed_and_treated_as_miss(self):
        """System clock abnormalities must not allow future timestamps to bypass the 24-hour expiration window."""
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[self._item()],
        )
        cache_path = self._cache_path()
        now = 2_000_000_000.0
        future_mtime = now + 60
        os.utime(cache_path, (future_mtime, future_mtime))

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            now=now,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_corrupted_cache_is_removed_without_breaking_search(self):
        """
        Process aborts, disk corruption, or manual edits can leave broken files. Read failures should fall back
        to remote search and remove the corrupted file, rather than allowing a single cache entry to permanently block material generation.
        """
        cache_path = self._cache_path()
        cache_path.write_text("{invalid-json", encoding="utf-8")

        with patch("app.services.material_cache.logger.warning") as warning:
            loaded = material_cache.load_material_search_cache(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())
        self.assertTrue(warning.called)

    def test_empty_results_are_not_cached(self):
        """
        Current provider interface uses [] to represent both empty results and request failures. Caching empty lists would
        freeze Cloudflare blocks or transient network issues for 24 hours; thus, only non-empty successful results can be cached.
        """
        saved = material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[],
        )

        self.assertFalse(saved)
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_cache_file_does_not_contain_search_parameters_or_credentials(self):
        """
        Cache filename uses a digest and content only stores material fields. Even if users share the storage directory,
        keywords, API keys, or other request configurations must never appear in the file.
        """
        item = self._item()
        item.source_info["source_page"] += "?token=drop"
        item.source_info["creator"]["profile_page"] += "?key=drop"
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="private search term",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[item],
        )
        cache_files = list(Path(self.temp_dir.name).glob("*.json"))

        self.assertEqual(len(cache_files), 1)
        self.assertNotIn("private search term", cache_files[0].name)
        raw_payload = cache_files[0].read_text(encoding="utf-8")
        payload = json.loads(raw_payload)
        self.assertEqual(set(payload), {"version", "items"})
        self.assertNotIn("private search term", raw_payload)
        self.assertNotIn("token=drop", raw_payload)

    def test_coverr_signed_urls_are_never_cached(self):
        """Coverr download URLs include signed JWTs and must not enter the long-term disk cache."""
        item = self._item(
            "https://storage.coverr.co/video/download?token=signed-jwt"
        )
        item.provider = "coverr"
        item.source_info["provider"] = "coverr"

        saved = material_cache.save_material_search_cache(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[item],
        )

        self.assertFalse(saved)
        self.assertEqual(list(Path(self.temp_dir.name).glob("*.json")), [])

    def test_coverr_cache_load_removes_legacy_signed_url(self):
        """Accessing Coverr should clean up any legacy signed download URL caches left by older versions."""
        cache_path = material_cache._cache_path(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        cache_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "coverr",
                            "url": "https://storage.coverr.co/video?token=signed-jwt",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        loaded = material_cache.load_material_search_cache(
            provider="coverr",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_version_one_cache_is_invalidated(self):
        """Version 1 cache lacks provenance info; after upgrading it must be requeried rather than generating incomplete task records."""
        cache_path = self._cache_path()
        cache_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "pixabay",
                            "url": "https://example.com/old.mp4",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        loaded = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertIsNone(loaded)
        self.assertFalse(cache_path.exists())

    def test_cache_key_separates_provider_duration_and_aspect(self):
        """
        Material provider, minimum duration, and aspect ratio all affect remote search results. Any change in these
        parameters must use an isolated cache to prevent returning materials incompatible with the current task.
        """
        base_path = material_cache._cache_path(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        paths = {
            base_path,
            material_cache._cache_path(
                provider="pexels",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            ),
            material_cache._cache_path(
                provider="pixabay",
                search_term="nature",
                minimum_duration=10,
                video_aspect=VideoAspect.portrait,
            ),
            material_cache._cache_path(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.landscape,
            ),
        }

        self.assertEqual(len(paths), 4)

    def test_search_wrapper_reuses_cache_across_calls(self):
        """
        First call performs remote search and saves to cache; second call with identical parameters must directly reuse disk results.
        This is core behavior to reduce Pixabay API calls and minimize triggering Cloudflare anti-bot mitigation.
        """
        remote_search = Mock(return_value=[self._item()])

        first = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        second = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertEqual(remote_search.call_count, 1)
        self.assertEqual(first, second)

    def test_search_wrapper_refreshes_mixed_orientation_cache(self):
        """
        Caches saved prior to upgrade might have mixed orientations. Returning only the filtered subset decreases
        material diversity, so any orientation mismatch must trigger a re-query that replaces the candidate set.
        """
        portrait_item = self._item("https://example.com/old-portrait.mp4")
        landscape_item = self._item("https://example.com/old-landscape.mp4")
        landscape_item.source_info["rendition"] = {
            "id": "large",
            "width": 1920,
            "height": 1080,
        }
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
            items=[portrait_item, landscape_item],
        )

        refreshed_item = self._item("https://example.com/refreshed-portrait.mp4")
        remote_search = Mock(return_value=[refreshed_item])
        results = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertEqual(remote_search.call_count, 1)
        self.assertEqual(
            [item.url for item in results],
            ["https://example.com/refreshed-portrait.mp4"],
        )
        cached_items = material_cache.load_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )
        self.assertEqual(
            [item.url for item in cached_items],
            ["https://example.com/refreshed-portrait.mp4"],
        )

    def test_square_search_reuses_crop_compatible_cache(self):
        """Square video tasks should continue reusing crop-compatible cached materials without repeated remote queries for differing original orientation."""
        landscape_item = self._item("https://example.com/landscape.mp4")
        landscape_item.source_info["rendition"] = {
            "id": "large",
            "width": 1920,
            "height": 1080,
        }
        material_cache.save_material_search_cache(
            provider="pixabay",
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.square,
            items=[landscape_item],
        )
        remote_search = Mock(return_value=[])

        results = material._search_videos_with_cache(
            provider="pixabay",
            search_videos=remote_search,
            search_term="nature",
            minimum_duration=5,
            video_aspect=VideoAspect.square,
        )

        self.assertEqual(remote_search.call_count, 0)
        self.assertEqual(
            [item.url for item in results],
            ["https://example.com/landscape.mp4"],
        )

    def test_search_wrapper_retries_after_empty_result(self):
        """Empty results are not cached; subsequent calls still contact remote to automatically retry once transient issues clear."""
        remote_search = Mock(return_value=[])

        for _ in range(2):
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )
            self.assertEqual(results, [])

        self.assertEqual(remote_search.call_count, 2)

    def test_cache_read_failure_falls_back_to_remote_search(self):
        """Cache read exceptions must only degrade to cache misses without blocking remote material search."""
        remote_items = [self._item()]
        remote_search = Mock(return_value=remote_items)

        with patch.object(
            material_cache,
            "load_material_search_cache",
            side_effect=RuntimeError("cache read failed"),
        ), patch.object(material_cache.logger, "warning") as warning:
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, remote_items)
        self.assertEqual(remote_search.call_count, 1)
        self.assertTrue(warning.called)

    def test_cache_write_failure_keeps_remote_results(self):
        """After remote search succeeds, usable materials must still be returned even if writing to cache fails."""
        remote_items = [self._item()]
        remote_search = Mock(return_value=remote_items)

        with patch.object(
            material_cache,
            "load_material_search_cache",
            return_value=None,
        ), patch.object(
            material_cache,
            "save_material_search_cache",
            side_effect=RuntimeError("cache write failed"),
        ), patch.object(material_cache.logger, "warning") as warning:
            results = material._search_videos_with_cache(
                provider="pixabay",
                search_videos=remote_search,
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, remote_items)
        self.assertEqual(remote_search.call_count, 1)
        self.assertTrue(warning.called)

    def test_invalid_cache_item_does_not_raise(self):
        """Malformed material objects must not allow optional cache writes to break caller flow."""
        with patch.object(material_cache.logger, "warning") as warning:
            saved = material_cache.save_material_search_cache(
                provider="pixabay",
                search_term="nature",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
                items=[None],
            )

        self.assertFalse(saved)
        self.assertTrue(warning.called)

    def test_concurrent_identical_searches_share_remote_request(self):
        """
        API service allows concurrent tasks. On initial search under identical criteria, subsequent threads should
        wait for the first thread to write to cache rather than redundantly consuming third-party quota.
        """
        remote_started = threading.Event()
        allow_remote_finish = threading.Event()
        remote_call_lock = threading.Lock()
        remote_call_count = 0
        results = []

        def remote_search(**_kwargs):
            nonlocal remote_call_count
            with remote_call_lock:
                remote_call_count += 1
            remote_started.set()
            self.assertTrue(allow_remote_finish.wait(timeout=2))
            return [self._item()]

        def run_search():
            results.append(
                material._search_videos_with_cache(
                    provider="pixabay",
                    search_videos=remote_search,
                    search_term="shared nature",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
            )

        first_thread = threading.Thread(target=run_search)
        second_thread = threading.Thread(target=run_search)
        first_thread.start()
        self.assertTrue(remote_started.wait(timeout=2))
        second_thread.start()
        # Give second thread time to enter cache lock wait area to ensure test covers real concurrent miss.
        time.sleep(0.05)
        allow_remote_finish.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(remote_call_count, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])

    def test_cleanup_removes_expired_entries_only(self):
        """Low-frequency cleanup deletes only expired caches without affecting active caches or user files."""
        stale_path = self._cache_path()
        stale_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "items": [
                        {
                            "provider": "pixabay",
                            "url": "https://example.com/stale.mp4",
                            "duration": 12,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        fresh_path = material_cache._cache_path(
            provider="pexels",
            search_term="fresh",
            minimum_duration=5,
            video_aspect=VideoAspect.landscape,
        )
        fresh_path.write_text("{}", encoding="utf-8")
        unrelated_path = Path(self.temp_dir.name) / "notes.json"
        unrelated_path.write_text("keep", encoding="utf-8")

        now = 2_000_000_000.0
        stale_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(stale_path, (stale_mtime, stale_mtime))
        os.utime(fresh_path, (now - 60, now - 60))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 1)
        self.assertFalse(stale_path.exists())
        self.assertTrue(fresh_path.exists())
        self.assertTrue(unrelated_path.exists())

    def test_cleanup_removes_orphaned_temp_files(self):
        """
        When a process is abruptly terminated, intermediate files created by NamedTemporaryFile(delete=False)
        have no chance to be consumed by os.replace, nor will exception cleanups fire. They must be collected along with
        expired cache; otherwise every crash leaves a permanent orphan file in the cache directory.
        """
        orphan_path = Path(self.temp_dir.name) / (
            f".{self._cache_path().stem}-m429jzwe.tmp"
        )
        orphan_path.write_text('{"version":2,"items":[]}', encoding="utf-8")
        unrelated_path = Path(self.temp_dir.name) / "notes.tmp"
        unrelated_path.write_text("keep", encoding="utf-8")

        now = 2_000_000_000.0
        stale_mtime = now - material_cache.MATERIAL_SEARCH_CACHE_TTL_SECONDS - 1
        os.utime(orphan_path, (stale_mtime, stale_mtime))
        os.utime(unrelated_path, (stale_mtime, stale_mtime))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 1)
        self.assertFalse(orphan_path.exists())
        # Other files with mismatched prefixes belong to the user and must not be deleted by cleanup logic.
        self.assertTrue(unrelated_path.exists())

    def test_cleanup_keeps_recent_temp_files(self):
        """
        During concurrent searches another process might be writing a temp file. Cleanup must reuse cache expiration logic
        to keep in-flight temp files intact, avoiding deletion of files waiting on os.replace.
        """
        in_flight_path = Path(self.temp_dir.name) / (
            f".{self._cache_path().stem}-m429jzwe.tmp"
        )
        in_flight_path.write_text('{"version":2,"items":', encoding="utf-8")

        now = 2_000_000_000.0
        os.utime(in_flight_path, (now - 1, now - 1))

        deleted = material_cache.cleanup_expired_material_search_cache(
            now=now,
            force=True,
        )

        self.assertEqual(deleted, 0)
        self.assertTrue(in_flight_path.exists())


if __name__ == "__main__":
    unittest.main()
