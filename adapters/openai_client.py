# -*- coding: utf-8 -*-
"""
adapters/openai_client.py — 统一模型调用（仅 Chat Completions）
- 兼容 OpenAI 官方与“OpenAI 兼容端点”（火山/通义/千帆等），通过 base_url + api_key_env 切换
- 支持多模态图片（本地路径→base64 data URL）
- 允许模型完整推导，要求末尾输出 <ANS>...</ANS>
- 更健壮：连接/超时/协议错误均重试；指数退避 + 抖动；可配置重试次数/退避基数
"""

from __future__ import annotations
import os, json, time, base64, mimetypes, random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import httpx
from openai import OpenAI
from openai import (
    APIConnectionError, APITimeoutError, APIStatusError,
    RateLimitError, BadRequestError
)


def _to_data_url(path: str) -> str:
    """读本地图片为 data URL（自动推断 MIME，默认 image/png）"""
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


@dataclass
class ClientConfig:
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    timeout_s: int = 60
    temperature: float = 0.0
    max_output_tokens: int = 1024
    max_retries: int = 3
    retry_backoff: float = 0.75
    extra_headers: Optional[Dict[str, str]] = None
    log_root: Optional[str] = None


class OpenAIClient:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"[OpenAIClient] 环境变量 {cfg.api_key_env} 未设置；请 export/设置后再试。"
            )

        # 显式超时 & 禁用 HTTP/2
        req_timeout = httpx.Timeout(
            connect=10.0,
            read=float(cfg.timeout_s or 60),
            write=30.0,
            pool=None,
        )
        http_client = httpx.Client(http2=False, timeout=req_timeout)

        # -------- 关键改动：仅在千帆时自动加 Bearer --------
        self.is_qianfan = bool(cfg.base_url and "qianfan.baidubce.com" in cfg.base_url)

        headers: Dict[str, str] = {}
        if cfg.extra_headers:
            headers.update(cfg.extra_headers)
        if self.is_qianfan:
            # 千帆要求：Authorization: Bearer <bce-v3/...>
            headers["Authorization"] = f"Bearer {api_key}"

        # 初始化 OpenAI 兼容 SDK 客户端
        self.client = OpenAI(
            api_key=api_key,              # OpenAI/兼容端点会用到；千帆留着也不冲突
            base_url=cfg.base_url,        # 千帆需指向 https://qianfan.baidubce.com/v2
            timeout=req_timeout,
            http_client=http_client,
            default_headers=headers,      # 千帆时附加 Bearer；其他为空
        )

        if cfg.log_root:
            _ensure_dir(cfg.log_root)

        # 启动日志：便于你确认当前通道
        provider = "qianfan" if self.is_qianfan else "openai/other"
        print(f"[OpenAIClient] provider={provider}, base_url={cfg.base_url}, api_key_env={cfg.api_key_env}")

    # ---------------- Public ----------------
    def generate(self,
                 prompt: str,
                 images: Optional[List[str]] = None,
                 log_tag: Optional[str] = None) -> str:
        sys_msg = {
            "role": "system",
            "content": (
                "Solve the problem carefully and show full reasoning if needed.\n"
                "At the very END, repeat the final answer on ONE LINE exactly as:\n"
                "<ANS> ... </ANS>\n"
                "Do not output anything after the closing tag."
            )
        }

        # 组装多模态内容：千帆与 OpenAI 的 image_url 字段差异
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            for p in images:
                data_url = _to_data_url(p)
                if self.is_qianfan:
                    # 千帆要求：image_url 是字符串
                    content.append({
                        "type": "image_url",
                        "image_url": data_url
                    })
                else:
                    # OpenAI / 其他兼容：image_url 是 {"url": ...}
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })

        messages = [sys_msg, {"role": "user", "content": content}]
        req = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }

        RETRYABLE_EXC = (
            APIConnectionError, APITimeoutError, RateLimitError, APIStatusError,
            httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
            httpx.WriteError, httpx.ReadError,
            httpx.RemoteProtocolError, httpx.LocalProtocolError
        )

        last_err = None
        for attempt in range(1, int(self.cfg.max_retries) + 2):
            try:
                resp = self.client.chat.completions.create(**req)
                text = (resp.choices[0].message.content or "").strip()

                if self.cfg.log_root and log_tag:
                    self._write_log(log_tag, ok=True, req=req, resp_text=text)

                return text

            except BadRequestError as e:
                last_err = e
                self._maybe_log_error(log_tag, req, e, attempt, final=True)
                break
            except APIStatusError as e:
                last_err = e
                code = getattr(e, "status_code", None)
                if code in (429, 500, 502, 503, 504):
                    self._maybe_log_error(log_tag, req, e, attempt)
                    self._backoff_sleep(attempt)
                    continue
                else:
                    self._maybe_log_error(log_tag, req, e, attempt, final=True)
                    break
            except RETRYABLE_EXC as e:
                last_err = e
                self._maybe_log_error(log_tag, req, e, attempt)
                self._backoff_sleep(attempt, jitter=True)
                continue
            except Exception as e:
                last_err = e
                self._maybe_log_error(log_tag, req, e, attempt, final=(attempt == int(self.cfg.max_retries) + 1))
                if attempt < int(self.cfg.max_retries) + 1:
                    self._backoff_sleep(attempt, jitter=True)
                    continue
                break

        raise RuntimeError(f"[OpenAIClient] 调用失败（{self.cfg.model}）：{last_err}")

    # ---------------- Helpers ----------------
    def _backoff_sleep(self, attempt: int, jitter: bool = False) -> None:
        base = float(self.cfg.retry_backoff or 0.75)
        t = (2 ** (attempt - 1)) * base
        if jitter:
            t *= (1.0 + 0.2 * random.random())
        time.sleep(t)

    def _write_log(self, tag: str, ok: bool, req: Dict[str, Any],
                   resp_text: Optional[str] = None, err: Optional[str] = None) -> None:
        if not self.cfg.log_root:
            return
        rec = {
            "ok": ok,
            "model": self.cfg.model,
            "base_url": self.cfg.base_url,
            "request": {
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_output_tokens,
                "has_images": any(
                    (isinstance(m.get("content"), list) and any(c.get("type") == "image_url" for c in m["content"]))
                    for m in req.get("messages", [])
                ),
            },
            "response_text": resp_text,
            "error": err,
        }
        path = os.path.join(self.cfg.log_root, f"{tag}.json")
        _ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    def _maybe_log_error(self, tag: Optional[str], req: Dict[str, Any], e: Exception,
                         attempt: int, final: bool = False) -> None:
        if self.cfg.log_root and tag:
            self._write_log(
                tag=tag,
                ok=False,
                req=req,
                resp_text=None,
                err=f"{type(e).__name__} @attempt={attempt}: {str(e)} | final={final}"
            )
