"""Read-only availability checks: a passing build is not a running application."""

import argparse
import json
import re
import sys
from uuid import UUID

import httpx


def check_services(client, *, frontend="http://127.0.0.1:5173", backend="http://127.0.0.1:8010", project_id=None):
    if project_id:
        project_id = str(UUID(project_id))
    results = []

    def check(name, url, validate):
        try:
            response = client.get(url)
            response.raise_for_status()
            detail = validate(response)
            results.append({"check": name, "ok": True, "status": response.status_code, "detail": detail})
            return response
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            results.append({"check": name, "ok": False, "error": type(exc).__name__, "detail": str(exc)[:300]})
            return None

    def health(response):
        if response.json().get("status") != "ok":
            raise ValueError("服务未通过健康检查")
        return "ok"

    def html(response):
        if "text/html" not in response.headers.get("content-type", "") or 'id="app"' not in response.text:
            raise ValueError("没有返回前端应用入口")
        return "应用入口已返回；浏览器渲染仍需另外验收"

    check("backend_health", backend.rstrip("/") + "/api/health", health)
    entry = check("frontend_page", frontend.rstrip("/") + (
        f"/projects/{project_id}" if project_id else "/"), html)
    if entry is not None:
        scripts = re.findall(r'<script\b[^>]*\bsrc=[\"\']([^\"\']+)[\"\']', entry.text, re.I)
        scripts = [src for src in scripts if src.startswith("/") and not src.startswith("//") and not src.startswith("/@vite")]
        if scripts:
            def javascript(response):
                if "javascript" not in response.headers.get("content-type", ""):
                    raise ValueError("前端主脚本缺失或返回了 HTML 错误页")
                return "主脚本可获取"
            check("frontend_script", frontend.rstrip("/") + scripts[-1], javascript)
        else:
            results.append({"check": "frontend_script", "ok": False, "detail": "应用入口缺少本地主脚本"})
    check("frontend_api_proxy", frontend.rstrip("/") + "/api/health", health)
    if project_id:
        def project(response):
            if response.json().get("id") != project_id:
                raise ValueError("项目接口未返回所选项目")
            return "项目记录可读取"
        check("project_api", frontend.rstrip("/") + f"/api/projects/{project_id}", project)
    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", default="http://127.0.0.1:5173")
    parser.add_argument("--backend", default="http://127.0.0.1:8010")
    parser.add_argument("--project-id", type=lambda value: str(UUID(value)))
    args = parser.parse_args()
    with httpx.Client(trust_env=False, timeout=8) as client:
        results = check_services(client, frontend=args.frontend, backend=args.backend, project_id=args.project_id)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(row["ok"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
