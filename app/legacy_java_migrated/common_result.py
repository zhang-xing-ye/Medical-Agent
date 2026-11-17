"""
对应原 Java:
- com.dao.c.backend.common.CommonResult
- com.dao.c.backend.common.StatusCode
- com.dao.c.backend.common.MessageCode
保持与原 Vue 前端 request.js 拦截器 100% 兼容的响应格式：{ "state": 1, "message": "...", "data": ... }
"""
from typing import Any, Optional


class StatusCode:
    SUCCESS = 1          # 业务成功
    ERROR = 0            # 业务失败
    NOT_LOGIN = 401      # 未登录或 Token 过期
    NOT_REGISTER = 402   # 用户未注册
    LOGOUT = 403         # 用户已注销

    DATA_NOT_EXIST = 0
    DATA_EXIST = 1


class MessageCode:
    SELECT = "查询成功"
    SELECTFAIL = "查询失败"
    UPDATE = "更新成功"
    UPDATEFAIL = "更新失败"
    DELETE = "删除成功"
    DELETEFAIL = "删除失败"


def common_result(state: int, message: str, data: Optional[Any] = None) -> dict:
    """构造与原 Java CommonResult<T> 完全一致的 JSON 字典"""
    return {
        "state": state,
        "message": message,
        "data": data,
    }
