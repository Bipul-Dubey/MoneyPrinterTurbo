import traceback
from typing import Any

from loguru import logger


class HttpException(Exception):
    def __init__(
        self, task_id: str, status_code: int, message: str = "", data: Any = None
    ):
        self.message = message
        self.status_code = status_code
        self.data = data
        # Retrieve the exception stack trace information.
        tb_str = traceback.format_exc().strip()
        if not tb_str or tb_str == "NoneType: None":
            msg = f"HttpException: {status_code}, {task_id}, {message}"
        else:
            msg = f"HttpException: {status_code}, {task_id}, {message}\n{tb_str}"

        # 400/401 are expected client input issues. Especially with authentication enabled,
        # internet scanners may send invalid keys; WARNING keeps diagnostic info without polluting
        # ERROR alerts. Server configuration errors and other exceptions remain ERROR.
        if status_code in (400, 401):
            logger.warning(msg)
        else:
            logger.error(msg)


class FileNotFoundException(Exception):
    pass
