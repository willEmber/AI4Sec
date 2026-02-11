"""
Sci-Hub PDF downloader module.

This module provides functionality to download PDFs from Sci-Hub mirror sites
as a last resort fallback when other OA sources fail.
"""

from __future__ import annotations

import os
import re
import urllib3
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.parse import urljoin

import requests

# 关闭 urllib3 的 InsecureRequestWarning 警告（Sci-Hub 镜像站可能证书无效）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Sci-Hub 镜像站列表
SCIHUB_MIRRORS = [
    "https://www.sci-hub.ru/",
    "https://www.sci-hub.se/",
    "https://www.sci-hub.st/",
    "https://sci-hub.box/",
    "https://sci-hub.red/",
    "https://sci-hub.al/",
    "https://www.sci-hub.ee/",
    "https://www.sci-hub.ren/",
    "https://sci-hub.shop/",
    "https://sci-hub.vg/",
]

# Sci-Hub cookies 目录（相对于本模块）
SCIHUB_COOKIES_DIR = Path(__file__).parent / "cookies"


def looks_like_pdf_bytes(data: bytes) -> bool:
    """检查数据是否看起来像 PDF 文件。"""
    return data.lstrip().startswith(b"%PDF")


def extract_pdf_url(site_url: str, html_content: str) -> str | None:
    """
    从 Sci-Hub 返回的 HTML 中提取真实的 PDF 下载链接。
    
    Args:
        site_url: Sci-Hub 镜像站 URL
        html_content: HTML 页面内容
        
    Returns:
        PDF 下载链接，如果未找到则返回 None
    """
    # 正则1：匹配 download 节点中的 PDF 链接
    download_pattern = r'<div class = "download">\s*<a href = "(.*?\.pdf)"></a>\s*</div>'
    # 正则2：匹配 object 节点中的 PDF 预览链接
    object_pattern = r'<object type = "application/pdf" data = "(.*?\.pdf)#[^"]*"></object>'

    download_matches = re.findall(download_pattern, html_content, re.DOTALL)
    if download_matches:
        pdf_url = download_matches[0].strip()
        return urljoin(site_url, pdf_url)

    object_matches = re.findall(object_pattern, html_content, re.DOTALL)
    if object_matches:
        pdf_url = object_matches[0].strip()
        return urljoin(site_url, pdf_url)

    return None


def load_cookies(session: requests.Session, site_url: str) -> None:
    """
    根据 Sci-Hub 站点 URL 加载对应的 cookies 文件。
    
    Args:
        session: requests Session 对象
        site_url: Sci-Hub 镜像站 URL
    """
    if not SCIHUB_COOKIES_DIR.exists():
        return
    # 提取站点域名（去掉协议和 www.）
    domain = site_url.split("://")[-1].strip("/").replace("www.", "")
    for cookie_file in SCIHUB_COOKIES_DIR.iterdir():
        if domain in cookie_file.name:
            try:
                cookie_jar = MozillaCookieJar()
                cookie_jar.load(str(cookie_file), ignore_discard=True, ignore_expires=True)
                session.cookies.update(cookie_jar)
                return
            except Exception:
                pass


def download(
    session: requests.Session,
    doi: str,
    dest_path: Path,
    *,
    timeout: int = 30,
    mirrors: list[str] | None = None,
) -> tuple[bool, str | None, str | None]:
    """
    尝试从 Sci-Hub 镜像站下载 PDF。
    
    Args:
        session: requests Session 对象
        doi: 论文 DOI
        dest_path: PDF 保存路径
        timeout: 请求超时时间（秒）
        mirrors: Sci-Hub 镜像站列表，默认使用内置列表
        
    Returns:
        (成功标志, 错误信息, 使用的镜像站 URL)
    """
    mirrors = mirrors or SCIHUB_MIRRORS
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    last_err: str | None = None
    used_mirror: str | None = None

    for site_url in mirrors:
        site_url = site_url.rstrip("/") + "/"
        sci_hub_url = urljoin(site_url, doi)
        used_mirror = site_url

        # 加载对应站点的 cookies
        load_cookies(session, site_url)

        try:
            # 第一步：请求初始 URL，判断响应类型
            resp = session.get(
                sci_hub_url,
                timeout=timeout,
                allow_redirects=True,
                verify=False,
                stream=False,
            )
            resp.raise_for_status()

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/pdf" in content_type:
                # 直接是 PDF 数据流
                real_pdf_url = sci_hub_url
            else:
                # 是 HTML 页面，提取真实 PDF 链接
                real_pdf_url = extract_pdf_url(site_url, resp.text)
                if not real_pdf_url:
                    last_err = "No PDF link found in Sci-Hub page"
                    continue

            # 第二步：下载真实 PDF 文件
            with session.get(
                real_pdf_url,
                timeout=timeout,
                allow_redirects=True,
                verify=False,
                stream=True,
            ) as pdf_resp:
                pdf_resp.raise_for_status()

                pdf_content_type = (pdf_resp.headers.get("Content-Type") or "").lower()
                if "application/pdf" not in pdf_content_type:
                    # 额外检查：是否以 %PDF 开头
                    first_chunk = next(pdf_resp.iter_content(chunk_size=1024), b"")
                    if not looks_like_pdf_bytes(first_chunk):
                        last_err = f"Not a PDF (Content-Type={pdf_content_type!r})"
                        continue
                    # 是 PDF，继续下载
                    with open(tmp_path, "wb") as f:
                        f.write(first_chunk)
                        for chunk in pdf_resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                else:
                    with open(tmp_path, "wb") as f:
                        for chunk in pdf_resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

            os.replace(tmp_path, dest_path)
            return True, None, used_mirror

        except requests.RequestException as e:
            last_err = str(e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            continue
        except Exception as e:
            last_err = str(e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            continue

    return False, last_err, used_mirror
