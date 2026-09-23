"""Minimal live connection test; never loads bank data or prints credentials."""
import json
from pathlib import Path
import sys
from urllib import request, error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moneygraph.assistant import Settings, NoRedirect


def main():
    settings = Settings.load(Path(__file__).resolve().parents[1] / ".env")
    if settings.provider != "openai" or not settings.api_key:
        print(json.dumps({"ok": False, "reason": "openai_not_configured"}))
        return 1
    payload = {"model": settings.model, "input": "Connection test. Reply exactly OK.",
               "max_output_tokens": 32, "store": False}
    req = request.Request("https://api.openai.com/v1/responses",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer " + settings.api_key})
    try:
        with request.build_opener(NoRedirect()).open(req, timeout=40) as response:
            result = json.loads(response.read(100000))
        text = "".join(part.get("text", "") for item in result.get("output", [])
                       if item.get("type") == "message" for part in item.get("content", [])
                       if part.get("type") == "output_text").strip()
        ok = result.get("status") == "completed" and text == "OK"
        print(json.dumps({"ok": ok, "http": 200, "completed": result.get("status") == "completed",
                          "expected_reply_received": text == "OK", "bank_data_sent": False}))
        return 0 if ok else 1
    except error.HTTPError as exc:
        try:
            body = json.loads(exc.read(100000)).get("error", {})
        except (ValueError, AttributeError):
            body = {}
        code = body.get("code")
        allowed = {"invalid_api_key", "insufficient_quota", "rate_limit_exceeded", "model_not_found",
                   "permission_denied", "unsupported_country_region_territory"}
        print(json.dumps({"ok": False, "http": exc.code,
                          "error_code": code if code in allowed else "provider_error",
                          "bank_data_sent": False}))
        return 1
    except (error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        print(json.dumps({"ok": False, "reason": "network_error", "error_type": type(reason).__name__,
                          "errno": getattr(reason, "errno", None), "winerror": getattr(reason, "winerror", None)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
