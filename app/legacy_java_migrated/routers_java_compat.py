"""
================================================================================
原 Java 后端 (dao_c_backend) 控制器与业务逻辑 1:1 迁移路由
================================================================================
覆盖原 Spring Boot 的全部 4 个 Controller：
1. UserController     -> /user/login, /user/sign
2. CategoryController -> /category/list/tree, /category/list/all
3. ModelController    -> /model/list/mid
4. SessionController  -> /session/list, /session/clearAll, /session/delete

注：原前端 axios (request.js) 默认使用 application/x-www-form-urlencoded 发送参数，
本路由通过 _extract_params() 同时兼容 Form 表单、Query 参数与 JSON 请求体。
"""
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.auth_and_ratelimit import (
    get_cached_category_tree,
    set_cached_category_tree,
)
from app.cache.redis_client import redis_manager
from app.legacy_java_migrated.common_result import MessageCode, StatusCode, common_result
from app.legacy_java_migrated.models_db import (
    CategoryEntity,
    ModelEntity,
    SessionEntity,
    UserEntity,
    get_db,
)
from app.legacy_java_migrated.security_utils import (
    JWTUtil,
    mask_phone_and_email,
    revoke_jwt_token,
)

router = APIRouter(tags=["Java-Backend-Migrated (兼容原8081接口)"])


async def _extract_params(request: Request) -> Dict[str, Any]:
    """统一提取 Form 表单、JSON 或 Query 参数，兼容原 Spring MVC 自动参数绑定行为"""
    params: Dict[str, Any] = dict(request.query_params)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                params.update(body)
        except Exception:
            pass
    elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        try:
            form = await request.form()
            params.update(dict(form))
        except Exception:
            pass
    return params


# ============================================================================
# 1. 对应 UserController.java & UserServiceImpl.java
# ============================================================================
@router.api_route("/user/login", methods=["POST", "GET"])
async def user_login(request: Request, db: AsyncSession = Depends(get_db)):
    params = await _extract_params(request)
    username = params.get("username")
    password = params.get("password")

    stmt = select(UserEntity).where(UserEntity.username == username)
    result = await db.execute(stmt)
    user = result.scalars().first()

    if user is None:
        return common_result(StatusCode.NOT_REGISTER, "用户未注册，请前往注册", None)
    if user.status == 0:
        return common_result(StatusCode.ERROR, "用户状态异常，请联系管理员", None)
    if user.status == -1:
        return common_result(StatusCode.LOGOUT, "用户已注销", {"id": user.id})
    if password is not None and password != user.password:
        return common_result(StatusCode.ERROR, "用户密码账号不匹配", None)

    token = JWTUtil.build_token({"username": user.username or ""})
    masked_user = mask_phone_and_email(user.to_dict())
    return common_result(StatusCode.SUCCESS, "登录成功", {"user": masked_user, "token": token})


@router.api_route("/user/sign", methods=["POST", "GET"])
async def user_sign(request: Request):
    return common_result(StatusCode.SUCCESS, "注册接口预留", None)


@router.post("/user/logout")
async def user_logout(authorization: str = Header(default="")):
    """主动退出登录：将当前 JWT 加入 Redis 黑名单，防止在有效期内被继续使用"""
    if authorization:
        await revoke_jwt_token(authorization)
    return common_result(StatusCode.SUCCESS, "已安全退出登录", None)


# ============================================================================
# 2. 对应 CategoryController.java & CategoryServiceImpl.java
# ============================================================================
def _set_category_children(
    parent: dict,
    all_categories: List[dict],
    models_by_cat_id: Dict[int, List[dict]],
) -> None:
    """递归构建三级分类与模型树（与 CategoryServiceImpl.setChildren 完全一致）"""
    parent["children"] = []
    parent_level = parent["categoryLevel"]
    parent_url = parent.get("categoryUrl") or ""

    child_cats = [
        c
        for c in all_categories
        if c["categoryLevel"] == parent_level + 1
        and c.get("categoryUrl")
        and c["categoryUrl"].startswith(parent_url)
    ]
    for child in child_cats:
        parent["children"].append(child)
        _set_category_children(child, all_categories, models_by_cat_id)

    # 二级分类下挂载对应的模型节点
    if parent_level == 2:
        related_models = models_by_cat_id.get(parent["id"], [])
        parent["children"].extend(related_models)


@router.api_route("/category/list/tree", methods=["POST", "GET"])
async def category_list_tree(db: AsyncSession = Depends(get_db)):
    # 1. Cache-Aside 先查 Redis 旁路缓存
    cached_tree = await get_cached_category_tree()
    if cached_tree is not None:
        return common_result(StatusCode.SUCCESS, MessageCode.SELECT, cached_tree)

    # 2. 缓存未命中时使用分布式互斥锁防击穿，随后回源 MySQL 并回填 Redis
    async with redis_manager.distributed_lock("category_tree_rebuild", ttl_seconds=10):
        # Double-check：拿到锁后再次检查缓存是否已被前序请求填好
        cached_tree = await get_cached_category_tree()
        if cached_tree is not None:
            return common_result(StatusCode.SUCCESS, MessageCode.SELECT, cached_tree)

        cat_res = await db.execute(
            select(CategoryEntity).where(CategoryEntity.status == StatusCode.DATA_EXIST)
        )
        mod_res = await db.execute(
            select(ModelEntity).where(ModelEntity.status == StatusCode.DATA_EXIST)
        )
        categories = [c.to_camel_dict() for c in cat_res.scalars().all()]
        models = [m.to_public_camel_dict() for m in mod_res.scalars().all()]

        if not categories or not models:
            return common_result(StatusCode.ERROR, MessageCode.SELECTFAIL, None)

        models_by_cat_id: Dict[int, List[dict]] = defaultdict(list)
        for m in models:
            models_by_cat_id[m["categoryId"]].append(m)

        root_categories = [c for c in categories if c["categoryLevel"] == 1]
        for root in root_categories:
            _set_category_children(root, categories, models_by_cat_id)

        await set_cached_category_tree(root_categories)
        return common_result(StatusCode.SUCCESS, MessageCode.SELECT, root_categories)


@router.api_route("/category/list/all", methods=["POST", "GET"])
async def category_list_all(db: AsyncSession = Depends(get_db)):
    cat_res = await db.execute(
        select(CategoryEntity).where(CategoryEntity.status == StatusCode.DATA_EXIST)
    )
    categories = [c.to_camel_dict() for c in cat_res.scalars().all()]
    return common_result(StatusCode.SUCCESS, MessageCode.SELECT, categories)


# ============================================================================
# 3. 对应 ModelController.java & ModelServiceImpl.java
# ============================================================================
@router.api_route("/model/list/mid", methods=["POST", "GET"])
async def model_list_mid(request: Request, db: AsyncSession = Depends(get_db)):
    params = await _extract_params(request)
    mid = int(params.get("id", 0))

    stmt = select(ModelEntity).where(
        ModelEntity.status == StatusCode.DATA_EXIST,
        ModelEntity.id == mid,
    )
    res = await db.execute(stmt)
    db_model = res.scalars().first()
    if db_model is None:
        return common_result(StatusCode.ERROR, MessageCode.SELECTFAIL, None)
    return common_result(StatusCode.SUCCESS, MessageCode.SELECT, db_model.to_public_camel_dict())


# ============================================================================
# 4. 对应 SessionController.java & SessionServiceImpl.java
# ============================================================================
@router.api_route("/session/list", methods=["POST", "GET"])
async def session_list_all(request: Request, db: AsyncSession = Depends(get_db)):
    params = await _extract_params(request)
    uid = int(params.get("id", 0))

    stmt = (
        select(SessionEntity)
        .where(SessionEntity.uid == uid, SessionEntity.status == 1)
        .order_by(desc(SessionEntity.createtime))
    )
    res = await db.execute(stmt)
    sessions = res.scalars().all()
    if not sessions:
        return common_result(StatusCode.SUCCESS, MessageCode.SELECT, None)

    model_ids = {s.mid for s in sessions if s.mid is not None}
    models_map: Dict[int, ModelEntity] = {}
    if model_ids:
        m_res = await db.execute(select(ModelEntity).where(ModelEntity.id.in_(model_ids)))
        models_map = {m.id: m for m in m_res.scalars().all()}

    session_vos = []
    for s in sessions:
        m = models_map.get(s.mid)
        session_vos.append(
            {
                "id": s.id,
                "uid": s.uid,
                "mid": s.mid,
                "title": s.title,
                "content": s.content,
                "createtime": s.createtime.isoformat() if s.createtime else None,
                "updatetime": s.updatetime.isoformat() if s.updatetime else None,
                "modelAvatar": m.avatar if m else None,
                "modelName": m.modelname if m else None,
            }
        )
    return common_result(StatusCode.SUCCESS, MessageCode.SELECT, session_vos)


@router.api_route("/session/clearAll", methods=["POST", "GET"])
async def session_clear_all(request: Request, db: AsyncSession = Depends(get_db)):
    params = await _extract_params(request)
    uid = int(params.get("uid", 0))

    stmt = (
        update(SessionEntity)
        .where(SessionEntity.uid == uid)
        .values(status=0, updatetime=datetime.now())
    )
    res = await db.execute(stmt)
    await db.commit()
    if res.rowcount <= 0:
        return common_result(StatusCode.ERROR, MessageCode.UPDATEFAIL, None)
    return common_result(StatusCode.SUCCESS, MessageCode.UPDATE, res.rowcount)


@router.api_route("/session/delete", methods=["POST", "GET"])
async def session_delete(request: Request, db: AsyncSession = Depends(get_db)):
    params = await _extract_params(request)
    sid = int(params.get("id", 0))

    stmt = (
        update(SessionEntity)
        .where(SessionEntity.id == sid)
        .values(status=0, updatetime=datetime.now())
    )
    res = await db.execute(stmt)
    await db.commit()
    if res.rowcount <= 0:
        return common_result(StatusCode.ERROR, MessageCode.UPDATEFAIL, None)
    return common_result(StatusCode.SUCCESS, MessageCode.UPDATE, res.rowcount)
