import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from app.controllers.manager.base_manager import TaskQueueFullError
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.controllers.manager.redis_manager import RedisTaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import task as task_service


def _queued_payload(func: str, **kwargs) -> str:
    """Construct a queue entry using the on-disk format of RedisTaskManager.enqueue."""
    return json.dumps({"func": func, "args": [], "kwargs": kwargs})


def _video_params() -> dict:
    """Return a serialized VideoParams object that passes current validation."""
    return VideoParams(video_subject="Tea").model_dump(warnings=False)


class TestInMemoryTaskManager(unittest.TestCase):
    def test_queue_operations_preserve_task_payload(self):
        """In-memory queue must preserve function, positional args, and keyword args without mutating task content."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
        task = {"func": len, "args": ([1, 2],), "kwargs": {}}

        manager.enqueue(task)

        self.assertFalse(manager.is_queue_empty())
        self.assertEqual(manager.queue_size(), 1)
        self.assertEqual(manager.dequeue(), task)
        self.assertTrue(manager.is_queue_empty())

    def test_add_task_rejects_only_after_queue_limit(self):
        """Allows queuing up to limit when concurrency slots are exhausted; only raises error when exceeding limit."""
        manager = InMemoryTaskManager(max_concurrent_tasks=0, max_queued_tasks=1)

        manager.add_task(len, [1])

        with self.assertRaises(TaskQueueFullError):
            manager.add_task(len, [2])

    def test_add_task_reserves_slot_before_background_thread_runs(self):
        """
        Concurrency slot must be reserved before thread startup; even if mock thread has not entered run_task yet,
        the second request must enter the queue without exceeding max_concurrent_tasks.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)

        with patch.object(manager, "execute_task") as execute_task:
            manager.add_task(len, [1])
            manager.add_task(len, [2])

        self.assertEqual(manager.current_tasks, 1)
        execute_task.assert_called_once_with(len, [1])
        self.assertEqual(manager.queue_size(), 1)

    def test_add_task_rolls_back_slot_when_thread_cannot_start(self):
        """Thread startup failure must not permanently occupy concurrency slot; exception should propagate to caller."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1)

        with patch.object(
            manager,
            "execute_task",
            side_effect=RuntimeError("thread unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                manager.add_task(len, [1])

        self.assertEqual(manager.current_tasks, 0)

    def test_task_done_starts_next_queued_task(self):
        """When current task completes, release concurrency slot and immediately schedule next task in queue."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
        manager.current_tasks = 1
        manager.enqueue({"func": len, "args": ([1, 2],), "kwargs": {}})

        with patch.object(manager, "execute_task") as execute_task:
            manager.task_done()

        self.assertEqual(manager.current_tasks, 1)
        execute_task.assert_called_once_with(len, [1, 2])
        self.assertTrue(manager.is_queue_empty())

    def test_new_request_cannot_take_slot_before_waiting_task(self):
        """A completion must reserve its freed slot for the oldest queued task."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
        manager.current_tasks = 1

        def queued_task():
            pass

        def incoming_task():
            pass

        manager.enqueue({"func": queued_task, "args": (), "kwargs": {}})
        released = threading.Event()
        resume = threading.Event()

        class PausingLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.worker = None
                self.paused = False

            def __enter__(self):
                self.lock.acquire()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.lock.release()
                if threading.current_thread() is self.worker and not self.paused:
                    self.paused = True
                    released.set()
                    resume.wait(timeout=2)

        manager.lock = PausingLock()
        started = []
        with patch.object(manager, "execute_task", side_effect=lambda func: started.append(func)):
            worker = threading.Thread(target=manager.task_done)
            manager.lock.worker = worker
            worker.start()
            try:
                self.assertTrue(released.wait(timeout=2))
                manager.add_task(incoming_task)
            finally:
                resume.set()
                worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(started, [queued_task])
        self.assertEqual(manager.dequeue()["func"], incoming_task)

    def test_task_done_requeues_task_when_thread_cannot_start(self):
        """If thread startup fails after dequeue, roll back slot and put task back into queue to avoid losing tasks."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)
        manager.current_tasks = 1
        queued_task = {"func": len, "args": ([1, 2],), "kwargs": {}}
        manager.enqueue(queued_task)

        with patch.object(
            manager,
            "execute_task",
            side_effect=RuntimeError("thread unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                manager.task_done()

        self.assertEqual(manager.current_tasks, 0)
        self.assertEqual(manager.dequeue(), queued_task)

    def test_run_task_releases_slot_after_failure(self):
        """When task function raises exception, finally must still release slot to prevent deadlock in queue."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1)
        manager.current_tasks = 1

        with patch.object(manager, "task_done") as task_done:
            with self.assertRaisesRegex(RuntimeError, "task failed"):
                manager.run_task(MagicMock(side_effect=RuntimeError("task failed")))

        self.assertEqual(manager.current_tasks, 1)
        task_done.assert_called_once_with()

    def test_check_queue_handles_dequeue_returning_none(self):
        """
        dequeue() may return None after internally skipping all tasks that no longer satisfy current validation,
        even if is_queue_empty() was False before check_queue was called. check_queue must not assume dequeue always
        yields a usable task, which would crash on task_info["func"].
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)

        with patch.object(manager, "is_queue_empty", return_value=False), patch.object(
            manager, "dequeue", return_value=None
        ), patch.object(manager, "execute_task") as execute_task:
            manager.check_queue()

        execute_task.assert_not_called()
        self.assertEqual(manager.current_tasks, 0)

    def test_execute_task_starts_background_thread(self):
        """Task execution entry point must start thread and forward function arguments completely to run_task."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1)
        fake_thread = MagicMock()

        with patch(
            "app.controllers.manager.base_manager.threading.Thread",
            return_value=fake_thread,
        ) as thread:
            manager.execute_task(len, [1, 2])

        thread.assert_called_once_with(
            target=manager.run_task,
            args=(len, [1, 2]),
            kwargs={},
        )
        fake_thread.start.assert_called_once_with()

    def test_limits_accept_quoted_toml_integers(self):
        """
        When limits are quoted as strings in TOML, they must be compared as integers;
        before the fix, concurrency comparisons raised TypeError, and queue limits only triggered after slots emptied.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks="1", max_queued_tasks="1")

        with patch.object(manager, "execute_task") as execute_task:
            manager.add_task(len, [1])
            manager.add_task(len, [2])
            with self.assertRaises(TaskQueueFullError):
                manager.add_task(len, [3])

        self.assertEqual(manager.max_concurrent_tasks, 1)
        self.assertEqual(manager.max_queued_tasks, 1)
        execute_task.assert_called_once_with(len, [1])
        self.assertEqual(manager.queue_size(), 1)

    def test_invalid_limits_raise_a_named_error(self):
        """Unparseable limits must immediately identify the config key name rather than deferring to anonymous exceptions at runtime."""
        with self.assertRaisesRegex(ValueError, "max_concurrent_tasks"):
            InMemoryTaskManager(max_concurrent_tasks="abc")

        with self.assertRaisesRegex(ValueError, "max_queued_tasks"):
            InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks="abc")

        with self.assertRaisesRegex(ValueError, "max_queued_tasks"):
            InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=True)

    def test_non_integral_limits_are_rejected_by_name(self):
        """Before fix, `0.5` was truncated to 0 by `int()` (tasks queued without workers executing), while `inf` leaked OverflowError."""
        cases = (
            (0.5, "an integer"),
            (1.5, "an integer"),
            ("0.5", "an integer"),
            (float("nan"), "a finite"),
            (float("inf"), "a finite"),
            (float("-inf"), "a finite"),
        )
        for field in ("max_concurrent_tasks", "max_queued_tasks"):
            for value, msg in cases:
                with self.subTest(field=field, value=value):
                    limits = {"max_concurrent_tasks": 1, "max_queued_tasks": 1}
                    limits[field] = value
                    pattern = f"{field} must be {msg}"
                    with self.assertRaisesRegex(ValueError, pattern):
                        InMemoryTaskManager(**limits)

        # Integral float values without truncation risks remain valid (TOML allows `max_queued_tasks = 2.0`).
        integral = {"max_concurrent_tasks": 1, "max_queued_tasks": 2.0}
        self.assertEqual(InMemoryTaskManager(**integral).max_queued_tasks, 2)

    def test_zero_concurrency_keeps_queueing_without_executing(self):
        """0 and negative values remain valid: queue only, do not execute; existing tests depend on this semantic."""
        manager = InMemoryTaskManager(max_concurrent_tasks=0, max_queued_tasks=1)

        with patch.object(manager, "execute_task") as execute_task:
            manager.add_task(len, [1])

        execute_task.assert_not_called()
        self.assertEqual(manager.queue_size(), 1)


class TestRedisTaskManager(unittest.TestCase):
    def setUp(self):
        self.redis_client = MagicMock()
        patcher = patch(
            "app.controllers.manager.redis_manager.redis.Redis.from_url",
            return_value=self.redis_client,
        )
        self.addCleanup(patcher.stop)
        from_url = patcher.start()
        self.manager = RedisTaskManager(
            max_concurrent_tasks=1,
            redis_url="redis://localhost:6379/0",
            max_queued_tasks=3,
        )
        from_url.assert_called_once_with("redis://localhost:6379/0")

    def test_resume_queued_tasks_fills_available_slots_after_restart(self):
        """Persisted Redis entries should run without waiting for a new API request."""
        self.manager.max_concurrent_tasks = 2
        self.redis_client.llen.return_value = 2
        self.redis_client.lpop.side_effect = [
            _queued_payload("start", task_id="first", params=_video_params()),
            _queued_payload("start", task_id="second", params=_video_params()),
        ]

        with patch.object(self.manager, "execute_task") as execute_task:
            self.manager.resume_queued_tasks()

        self.assertEqual(execute_task.call_count, 2)
        self.assertEqual(self.manager.current_tasks, 2)
        self.assertEqual(self.redis_client.lpop.call_count, 2)

    def test_enqueue_serializes_video_params_without_mutating_task(self):
        """
        Redis only stores JSON; VideoParams should be converted to dict, but the original task must retain the model,
        avoiding serialization side-effects impacting logging, retry, or subsequent caller reads.
        """
        params = VideoParams(video_subject="Coffee")
        task = {
            "func": task_service.start,
            "args": (),
            "kwargs": {"task_id": "task-1", "params": params},
        }

        self.manager.enqueue(task)

        self.assertIs(task["kwargs"]["params"], params)
        queue_name, payload = self.redis_client.rpush.call_args.args
        decoded = json.loads(payload)
        self.assertEqual(queue_name, "task_queue")
        self.assertEqual(decoded["func"], "start")
        self.assertEqual(decoded["kwargs"]["task_id"], "task-1")
        self.assertEqual(decoded["kwargs"]["params"]["video_subject"], "Coffee")

    def test_dequeue_restores_function_and_video_params(self):
        """Tasks dequeued from Redis should restore callable functions and VideoParams model."""
        payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-1",
                "params": VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ),
            },
        }
        self.redis_client.lpop.return_value = json.dumps(payload)

        task = self.manager.dequeue()

        self.redis_client.lpop.assert_called_once_with("task_queue")
        self.assertIs(task["func"], task_service.start)
        self.assertIsInstance(task["kwargs"]["params"], VideoParams)
        self.assertEqual(task["kwargs"]["params"].video_subject, "Coffee")

    def test_empty_queue_and_size_use_redis_length(self):
        """Queue emptiness check and size must directly reflect Redis current list length."""
        self.redis_client.lpop.return_value = None
        self.redis_client.llen.side_effect = [0, 2]

        self.assertIsNone(self.manager.dequeue())
        self.assertTrue(self.manager.is_queue_empty())
        self.assertEqual(self.manager.queue_size(), 2)

    def test_dequeue_skips_task_that_fails_current_validation(self):
        """
        A task may have been queued before validation rules were tightened (e.g. video_count used to permit 0).
        lpop is destructive; when reconstructing VideoParams fails, this task is permanently removed from Redis
        and cannot be treated as present; dequeue should not propagate validation errors to caller
        (which would crash the locked caller and drop the task unlogged), but rather skip it and continue trying
        the next item until a valid task is found or the queue is truly empty.
        """
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        valid_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-valid",
                "params": VideoParams(video_subject="Tea").model_dump(
                    warnings=False
                ),
            },
        }
        self.redis_client.lpop.side_effect = [
            json.dumps(stale_payload),
            json.dumps(valid_payload),
        ]

        task = self.manager.dequeue()

        self.assertEqual(self.redis_client.lpop.call_count, 2)
        self.assertEqual(task["kwargs"]["task_id"], "task-valid")
        self.assertIsInstance(task["kwargs"]["params"], VideoParams)
        self.assertEqual(task["kwargs"]["params"].video_subject, "Tea")

    def test_dequeue_returns_none_when_every_queued_task_is_stale(self):
        """When all remaining tasks are discarded due to current validation rules, return None rather than raising exception."""
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": -1},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        self.assertIsNone(self.manager.dequeue())
        self.assertEqual(self.redis_client.lpop.call_count, 2)

    def test_dequeue_marks_stale_task_failed_instead_of_leaving_it_processing(self):
        """
        Task state record is created prior to queuing with processing by default. Merely skipping and discarding
        this queue entry without updating state record would leave it permanently showing as running in API/WebUI.
        patch_task (rather than update_task) must mark it failed, so if task was deleted by user, its state is not resurrected.
        """
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = True
            result = self.manager.dequeue()

        self.assertIsNone(result)
        state.patch_task.assert_called_once()
        call_args = state.patch_task.call_args
        self.assertEqual(call_args.args[0], "task-stale")
        self.assertEqual(call_args.kwargs["state"], const.TASK_STATE_FAILED)
        self.assertEqual(call_args.kwargs["failed_stage"], "dequeue")
        self.assertIn("video_count", call_args.kwargs["error"])

    def test_dequeue_does_not_recreate_state_for_already_deleted_task(self):
        """patch_task returns False when task was already deleted; dequeue should not treat this as an error."""
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-deleted",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = False
            result = self.manager.dequeue()

        self.assertIsNone(result)
        state.patch_task.assert_called_once()
        state.update_task.assert_not_called()

    def test_dequeue_skips_task_with_unknown_function_name(self):
        """
        Members of FUNC_MAP vary with deployments (commented-out entries exist), and legacy entry points may linger in queue.
        Directly indexing names not in FUNC_MAP raises KeyError, bypassing dequeue's discard strategy, so it must be skipped
        identically to the validation failure path.
        """
        self.redis_client.lpop.side_effect = [
            _queued_payload("start_test", task_id="task-unknown-func"),
            _queued_payload("start", task_id="task-valid", params=_video_params()),
        ]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = True
            task = self.manager.dequeue()

        self.assertEqual(self.redis_client.lpop.call_count, 2)
        self.assertIs(task["func"], task_service.start)
        self.assertEqual(task["kwargs"]["task_id"], "task-valid")
        state.patch_task.assert_called_once()
        call_args = state.patch_task.call_args
        self.assertEqual(call_args.args[0], "task-unknown-func")
        self.assertEqual(call_args.kwargs["state"], const.TASK_STATE_FAILED)
        self.assertIn("start_test", call_args.kwargs["error"])

    def test_dequeue_skips_every_unusable_payload_shape(self):
        """
        Three types of unusable items may linger in queue: truncated JSON, non-object JSON, and tasks referencing
        removed functions (commented out in FUNC_MAP). None of them may bubble out of dequeue, or later usable tasks
        would never be scheduled; entries where task_id is recoverable must transition the departed task to failure.
        """
        self.redis_client.lpop.side_effect = [
            '{"func": "start", "args": [',
            json.dumps(["not", "a", "mapping"]),
            _queued_payload("retired-entry", task_id="task-retired"),
            _queued_payload("start", task_id="task-valid", params=_video_params()),
        ]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = True
            task = self.manager.dequeue()

        self.assertEqual(self.redis_client.lpop.call_count, 4)
        self.assertIs(task["func"], task_service.start)
        self.assertEqual(task["kwargs"]["task_id"], "task-valid")
        state.patch_task.assert_called_once()
        self.assertEqual(state.patch_task.call_args.args[0], "task-retired")
        self.assertEqual(
            state.patch_task.call_args.kwargs["state"], const.TASK_STATE_FAILED
        )

    def test_dequeue_returns_none_when_only_unusable_entries_remain(self):
        """When all remaining entries are unusable, return None instead of raising an exception."""
        self.redis_client.lpop.side_effect = [
            _queued_payload("retired-entry"),
            None,
        ]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            result = self.manager.dequeue()

        self.assertIsNone(result)
        self.assertEqual(self.redis_client.lpop.call_count, 2)
        state.patch_task.assert_not_called()

    def test_task_done_keeps_draining_queue_when_entry_is_unusable(self):
        """
        After finishing tasks release slots, task_done triggers check_queue to schedule the next task.
        If an unusable entry causes check_queue to raise, the exception would crash the worker thread via run_task's finally;
        subsequently no tasks would run and none would call check_queue, leaving trailing tasks permanently stuck in processing.
        Therefore task_done must continue scheduling the next usable task rather than stranding the entire queue.
        """
        self.manager.current_tasks = 1
        self.redis_client.lpop.side_effect = [
            _queued_payload("retired-entry"),
            _queued_payload("start", task_id="task-next", params=_video_params()),
        ]

        with patch.object(self.manager, "execute_task") as execute_task:
            self.manager.task_done()

        self.assertEqual(self.manager.current_tasks, 1)
        self.assertEqual(execute_task.call_args.kwargs["task_id"], "task-next")
        self.assertIs(execute_task.call_args.args[0], task_service.start)

    def test_dequeue_discards_payloads_that_cannot_be_dispatched(self):
        """
        Three types of entries parse successfully yet cannot be dispatched: lpop returning falsy, args defined as null / mapping /
        number / string. None of these may cause dequeue to raise or be treated as "queue empty" (which would requeue or exit early),
        leaving queued usable tasks starved.
        """
        cases = (
            ("empty bytes", b"", None),
            ("empty string", "", None),
            ("args null", {"args": None}, "t-1"),
            ("args mapping", {"args": {}}, "t-2"),
            ("args number", {"args": 5}, "t-3"),
            ("args string", {"args": "x"}, "t-4"),
        )

        for case, broken, expected_task_id in cases:
            with self.subTest(case=case):
                if isinstance(broken, dict):
                    broken = json.dumps(
                        {
                            "func": "start",
                            "kwargs": {"task_id": expected_task_id},
                            **broken,
                        }
                    )
                self.redis_client.reset_mock()
                self.redis_client.lpop.side_effect = [
                    broken,
                    _queued_payload(
                        "start", task_id="task-valid", params=_video_params()
                    ),
                ]

                with patch("app.controllers.manager.redis_manager.sm.state") as state:
                    state.patch_task.return_value = True
                    task = self.manager.dequeue()

                self.assertEqual(self.redis_client.lpop.call_count, 2)
                self.assertEqual(task["kwargs"]["task_id"], "task-valid")
                self.assertIs(task["func"], task_service.start)
                if expected_task_id is None:
                    state.patch_task.assert_not_called()
                else:
                    self.assertEqual(
                        state.patch_task.call_args.args[0], expected_task_id
                    )
                    self.assertIn(
                        "positional arguments",
                        state.patch_task.call_args.kwargs["error"],
                    )

    def test_dequeue_dispatches_payload_without_args_field(self):
        """Complete omission of the args field is an existing format (check_queue defaults to empty tuple) and must not be discarded."""
        self.redis_client.lpop.return_value = json.dumps(
            {
                "func": "start",
                "kwargs": {"task_id": "task-no-args", "params": _video_params()},
            }
        )

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            task = self.manager.dequeue()

        self.assertEqual(task["kwargs"]["task_id"], "task-no-args")
        self.assertIs(task["func"], task_service.start)
        state.patch_task.assert_not_called()

    def test_dequeue_ignores_non_string_task_id_in_every_discard_path(self):
        """
        When task_id is a non-string such as a JSON list or dict, passing it to patch_task causes redis to raise
        DataError, interrupting the discard loop on a single corrupt entry and blocking trailing valid tasks.
        Both discard paths must skip state persistence and simply discard the entry itself.
        """
        cases = (
            ("unknown function", _queued_payload("retired-entry", task_id=["task-1"])),
            (
                "stale params",
                _queued_payload(
                    "start",
                    task_id={"nested": "task-2"},
                    params={**_video_params(), "video_count": 0},
                ),
            ),
        )

        for case, payload in cases:
            with self.subTest(case=case):
                self.redis_client.reset_mock()
                self.redis_client.lpop.side_effect = [
                    payload,
                    _queued_payload(
                        "start", task_id="task-valid", params=_video_params()
                    ),
                ]

                with patch("app.controllers.manager.redis_manager.sm.state") as state:
                    task = self.manager.dequeue()

                self.assertEqual(self.redis_client.lpop.call_count, 2)
                self.assertEqual(task["kwargs"]["task_id"], "task-valid")
                state.patch_task.assert_not_called()
                state.update_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
