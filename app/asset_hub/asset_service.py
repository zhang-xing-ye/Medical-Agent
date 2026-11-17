"""
================================================================================
多模态资产中心 (AssetHub) —— Claim-Check 凭据指针模式 + Redis 元数据共享缓存
================================================================================
核心作用：
解决 `.nii.gz` / `.npy` 等数十 MB 的医学影像文件无法放进 DeepSeek 上下文或 MCP JSON-RPC 的问题。
1. 用户上传文件 -> 存入 SHARED_ASSET_DIR -> 生成短凭据 `asset_id` (如 "ast_9a8f1b")；
2. 凭据元数据写入 Redis (`dao_c:asset:meta:{asset_id}`)，支持多 FastAPI Worker 与远程服务共享；
3. 总控 DeepSeek 调用 `api_new_mcp` 工具时仅传递 `asset_id` 字符串。
"""
import json
import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.cache.redis_client import build_key, redis_manager
from app.config import settings

router = APIRouter(prefix="/api/v1/assets", tags=["AssetHub (大文件凭据指针中心)"])


def _asset_meta_key(asset_id: str) -> str:
    return build_key("asset", "meta", asset_id)


@router.post("/upload")
async def upload_medical_asset(file: UploadFile = File(...)) -> dict:
    """
    上传医学影像 (.nii, .nii.gz, .npy) 或生信数据文件，返回轻量凭据 asset_id，
    并将元数据写入 Redis 缓存供跨 Worker / MCP 调度读取。
    """
    orig_name = file.filename or "unknown.dat"
    asset_id = f"ast_{uuid.uuid4().hex[:8]}"
    safe_filename = f"{asset_id}__{os.path.basename(orig_name)}"
    save_path = os.path.join(settings.SHARED_ASSET_DIR, safe_filename)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    lower_name = orig_name.lower()
    if lower_name.endswith(".nii") or lower_name.endswith(".nii.gz"):
        modality = "nifti_brain_mri"
    elif lower_name.endswith(".npy"):
        modality = "numpy_brain_slices"
    else:
        modality = "general_medical_file"

    meta = {
        "asset_id": asset_id,
        "filename": orig_name,
        "modality": modality,
        "size_mb": round(len(content) / 1024 / 1024, 3),
        "storage_path": save_path,
    }

    client = await redis_manager.get_client()
    await client.setex(
        _asset_meta_key(asset_id),
        settings.REDIS_ASSET_TTL_SECONDS,
        json.dumps(meta, ensure_ascii=False),
    )
    return meta


@router.get("/{asset_id}/download")
async def download_medical_asset(asset_id: str):
    """供跨机器部署的远程 FastMCP 服务或前端画布流式拉取文件"""
    meta = await get_asset_metadata(asset_id)
    storage_path = meta.get("storage_path")
    if storage_path and os.path.exists(storage_path):
        return FileResponse(storage_path)

    for fname in os.listdir(settings.SHARED_ASSET_DIR):
        if fname.startswith(asset_id):
            return FileResponse(os.path.join(settings.SHARED_ASSET_DIR, fname))

    raise HTTPException(status_code=404, detail=f"未找到资产凭据: {asset_id}")


async def get_asset_metadata(asset_id: str) -> dict:
    """
    Read-Through 模式获取资产轻量元数据：
    优先命中 Redis 缓存；未命中时扫描本地存储目录并自动回填 Redis。
    """
    client = await redis_manager.get_client()
    raw = await client.get(_asset_meta_key(asset_id))
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    if os.path.exists(settings.SHARED_ASSET_DIR):
        for fname in os.listdir(settings.SHARED_ASSET_DIR):
            if fname.startswith(asset_id):
                orig = fname.split("__", 1)[-1] if "__" in fname else fname
                full_path = os.path.join(settings.SHARED_ASSET_DIR, fname)
                meta = {
                    "asset_id": asset_id,
                    "filename": orig,
                    "modality": "nifti_brain_mri"
                    if orig.lower().endswith((".nii", ".nii.gz"))
                    else "numpy_brain_slices",
                    "storage_path": full_path,
                }
                await client.setex(
                    _asset_meta_key(asset_id),
                    settings.REDIS_ASSET_TTL_SECONDS,
                    json.dumps(meta, ensure_ascii=False),
                )
                return meta

    return {"asset_id": asset_id, "filename": asset_id, "modality": "unknown"}
