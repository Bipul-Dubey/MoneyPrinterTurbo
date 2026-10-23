import inspect
import json
from typing import Dict

import redis
from loguru import logger
from pydantic import ValidationError

from app.controllers.manager.base_manager import TaskManager, TaskQueueFullError
from app.models import const
from app.models.schema import AudioRequest, SubtitleRequest, VideoParams
from app.services import state as sm
from app.services import task as tm

FUNC_MAP = {
    "start": tm.start,
    # 'start_test': tm.start_test
}

_ADMIT_QUEUED_TASK_SCRIPT = """
if redis.call("LLEN", KEYS[1]) >= tonumber(ARGV[1]) then
    return 0
end
redis.call("RPUSH", KEYS[1], ARGV[2])
return 1
"""


class RedisTaskManager(TaskManager):
    def __init__(
        self,
        max_concurrent_tasks: int,
        redis_url: str,
        max_queued_tasks: int = 100,
    ):
        self.redis_client = redis.Redis.from_url(redis_url)
        super().__init__(max_concurrent_tasks, max_queued_tasks=max_queued_tasks)

    def create_queue(self):
        return "task_queue"

    def resume_queued_tasks(self):
        """Dispatch persisted queue entries when a new API process starts.

        Queued Redis items survive a restart, but no worker calls check_queue
        until an active task finishes. Fill each available local worker slot so
        an idle process does not leave old requests stuck in processing.
        """
        for _ in range(max(0, self.max_concurrent_tasks - self.current_tasks)):
            self.check_queue()

    def enqueue(self, task: Dict):
        self.redis_client.rpush(self.queue, self._serialize_task(task))

    def enqueue_new_task(self, task: Dict):
        # Manager locks are process-local. Keep capacity validation and admission
        # in one Redis operation so other API workers cannot claim the same slot.
        admitted = self.redis_client.eval(
            _ADMIT_QUEUED_TASK_SCRIPT,
            1,
            self.queue,
            self.max_queued_tasks,
            self._serialize_task(task),
        )
        if not admitted:
            raise TaskQueueFullError("task queue is full, please try again later")

    @staticmethod
    def _serialize_task(task: Dict) -> str:
        task_with_serializable_params = task.copy()
        # task.copy() only shallow-copies the outer dict. Deep-copy kwargs separately
        # so converting objects to dicts does not mutate the caller's VideoParams instance.
        task_kwargs = task.get("kwargs", {})
        task_with_serializable_params["kwargs"] = task_kwargs.copy()

        if "params" in task_kwargs and isinstance(
            task_kwargs["params"], (VideoParams, AudioRequest, SubtitleRequest)
        ):
            task_with_serializable_params["kwargs"]["params"] = task_kwargs[
                "params"
            ].model_dump(warnings=False)

        # Convert function object to its name
        task_with_serializable_params["func"] = task["func"].__name__
        return json.dumps(task_with_serializable_params)

    def dequeue(self):
        # Loop rather than popping once: queued tasks may have met validation rules that changed across deployments
        # (e.g. VideoParams schema additions, removed functions). Old unparseable items would otherwise crash workers
        # or leave workers idle with valid items remaining. Discard invalid items and continue popping until valid or empty.
        while True:
            task_json = self.redis_client.lpop(self.queue)
            # Only when lpop returns None is the queue genuinely empty. Empty strings/bytes are invalid items
            # that must follow the discard path rather than signalling queue end.
            if task_json is None:
                return None

            task_info = None
            try:
                task_info = json.loads(task_json)
                # Map function name back to callable. Name missing or absent from FUNC_MAP cannot be indexed directly;
                # KeyError must be caught to follow the discard policy.
                task_info["func"] = FUNC_MAP[task_info["func"]]
                task_kwargs = task_info["kwargs"]
                if not isinstance(task_kwargs, dict):
                    raise ValueError("queued task has no keyword argument mapping")
                # When args is missing, use default empty list; non-list structures would cause TypeError on *args.
                if not isinstance(task_info.get("args", []), list):
                    raise ValueError("queued task positional arguments are not a list")
                # A persisted request can outlive a callable's signature. Fail
                # it before dispatch: Python argument-binding errors happen
                # before start() enters its pipeline failure handler.
                inspect.signature(task_info["func"]).bind(
                    *task_info.get("args", []), **task_kwargs
                )
            except (TypeError, ValueError, KeyError) as e:
                logger.error(f"dropping unusable queued task: {e}")
                # Consistent with params validation failure: if task_id can be read, mark the popped
                # task as failed so API/WebUI does not display it as processing forever. If payload cannot
                # be parsed or task_id is non-string, discard without calling patch_task to prevent DataError.
                stale_kwargs = (
                    task_info.get("kwargs") if isinstance(task_info, dict) else None
                )
                task_id = (
                    stale_kwargs.get("task_id")
                    if isinstance(stale_kwargs, dict)
                    else None
                )
                if isinstance(task_id, str) and task_id:
                    sm.state.patch_task(
                        task_id,
                        state=const.TASK_STATE_FAILED,
                        failed_stage="dequeue",
                        error=f"discarded stale queued task: {e}",
                    )
                continue

            if "params" in task_kwargs and isinstance(task_kwargs["params"], dict):
                try:
                    params_model = VideoParams
                    # /audio and /subtitle use short-form request models without
                    # video_subject. Preserve those schemas at the queue boundary;
                    # full VideoParams can also stop at either of these stages.
                    if "video_subject" not in task_kwargs["params"]:
                        if task_kwargs.get("stop_at") == "audio":
                            params_model = AudioRequest
                        elif task_kwargs.get("stop_at") == "subtitle":
                            params_model = SubtitleRequest
                    task_kwargs["params"] = params_model(**task_kwargs["params"])
                except ValidationError as e:
                    logger.error(
                        "dropping queued task with params that fail current "
                        f"request model validation (queued under an older, more "
                        f"permissive schema, or corrupted): {e}"
                    )
                    # Task state is created before queuing and defaults to processing.
                    # Mark invalid items as failed via patch_task rather than update_task so deleted tasks
                    # are not resurrected. If task_id is invalid, skip update.
                    task_id = task_kwargs.get("task_id")
                    if isinstance(task_id, str) and task_id:
                        sm.state.patch_task(
                            task_id,
                            state=const.TASK_STATE_FAILED,
                            failed_stage="dequeue",
                            error=f"discarded stale queued task: {e}",
                        )
                    continue

            return task_info

    def is_queue_empty(self):
        return self.redis_client.llen(self.queue) == 0

    def queue_size(self):
        return self.redis_client.llen(self.queue)
