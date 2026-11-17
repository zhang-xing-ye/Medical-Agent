"""
对应原 Java:
- com.dao.c.backend.utils.JWTUtil (HMAC256 Token 签发与校验)
- com.dao.c.backend.utils.AesUtil (AES/GCM/NoPadding 128位加解密与手机号/邮箱掩码)
"""
import base64
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import jwt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Header, HTTPException

from app.cache.auth_and_ratelimit import blacklist_jwt_token, is_jwt_blacklisted
from app.config import settings


class JWTUtil:
    """1:1 兼容原 Java com.auth0.jwt 的 HMAC256 签名逻辑"""

    @staticmethod
    def build_token(payload: Dict[str, str]) -> str:
        tz_gmt8 = timezone(timedelta(hours=8))
        exp = datetime.now(tz_gmt8) + timedelta(days=settings.JWT_EXPIRE_DAYS)
        claims = {**payload, "exp": exp}
        return jwt.encode(claims, settings.JWT_SIGNATURE, algorithm="HS256")

    @staticmethod
    def verify_and_decode(token: str) -> Dict[str, str]:
        if not token or not token.strip():
            raise ValueError("Token is null or empty")
        # 兼容前端直接传裸 Token 或 Bearer Token
        clean_token = token.replace("Bearer ", "").strip()
        return jwt.decode(clean_token, settings.JWT_SIGNATURE, algorithms=["HS256"])


class AesUtil:
    """
    1:1 兼容原 Java AesUtil:
    算法: AES/GCM/NoPadding, IV 长度: 12 字节, Tag 长度: 128 位, 密钥: 16 字节 (AES-128)
    """

    IV_LENGTH = 12

    def __init__(self, base64_key: str = settings.AES_SECRET_KEY_BASE64):
        decoded_key = base64.b64decode(base64_key)
        if len(decoded_key) != 16:
            raise ValueError("密钥必须是 16 字节（AES-128）")
        self.aesgcm = AESGCM(decoded_key)

    def encrypt(self, plaintext: str) -> str:
        iv = os.urandom(self.IV_LENGTH)
        ciphertext_with_tag = self.aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
        combined = iv + ciphertext_with_tag
        return base64.b64encode(combined).decode("utf-8")

    def decrypt(self, encrypted_text: str) -> str:
        combined = base64.b64decode(encrypted_text)
        iv = combined[: self.IV_LENGTH]
        ciphertext_with_tag = combined[self.IV_LENGTH :]
        decrypted = self.aesgcm.decrypt(iv, ciphertext_with_tag, None)
        return decrypted.decode("utf-8")


aes_util = AesUtil()


def mask_phone_and_email(user_dict: dict) -> dict:
    """
    对应原 Java UserServiceImpl.phoneAndEmailMask(User user)
    对 AES-GCM 加密存储的手机号与邮箱解密后执行星号脱敏。
    """
    phone = user_dict.get("phone")
    if phone:
        try:
            raw_phone = aes_util.decrypt(phone)
            if len(raw_phone) >= 7:
                user_dict["phone"] = raw_phone[:3] + "****" + raw_phone[7:]
        except Exception:
            pass

    email = user_dict.get("email")
    if email:
        try:
            raw_email = aes_util.decrypt(email)
            if "@" in raw_email:
                local_part, domain = raw_email.split("@", 1)
                if len(local_part) > 4:
                    masked_local = local_part[:2] + "****" + local_part[-2:]
                else:
                    masked_local = "****"
                user_dict["email"] = f"{masked_local}@{domain}"
        except Exception:
            pass

    user_dict["openid"] = None
    user_dict["password"] = None
    return user_dict


async def verify_jwt_header(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """对应原 Java JWTInterceptor，校验请求头中的 Authorization，并检查 Redis 吊销黑名单"""
    if not authorization:
        return None
    if await is_jwt_blacklisted(authorization):
        raise HTTPException(status_code=401, detail="Token 已注销，请重新登录")
    try:
        return JWTUtil.verify_and_decode(authorization)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token 无效或已过期: {e}")


async def revoke_jwt_token(authorization: str) -> None:
    """退出登录时解析 Token 过期时间并写入 Redis 黑名单"""
    try:
        claims = JWTUtil.verify_and_decode(authorization)
        exp_ts = int(claims["exp"]) if "exp" in claims else None
    except Exception:
        exp_ts = None
    await blacklist_jwt_token(authorization, exp_timestamp=exp_ts)
