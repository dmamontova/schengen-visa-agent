"""Safe three-stage GigaChat check. Does not reveal secrets or response bodies.

Run: python -m schengen_agent.diagnostics
"""
from __future__ import annotations

import os

from .config import Settings
from .providers import GigaChatProvider
from .schemas import tool_specifications


def describe_failure(exc: BaseException) -> str:
    """Only safe metadata: never print str(exc), repr(exc), URLs or headers."""
    name = type(exc).__name__
    modules = [type(exc).__module__]
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)

    cause = exc.__cause__ or exc.__context__
    for _ in range(3):
        if cause is None:
            break
        modules.append(type(cause).__module__)
        if status is None:
            status = getattr(cause, "status_code", None)
        cause = cause.__cause__ or cause.__context__
    label = f"{name} (HTTP {status})" if isinstance(status, int) else name
    names = " ".join([name, *modules]).lower()
    if status == 401:
        hint = "Проверьте новый Authorization Key и срок действия авторизации."
    elif status == 403:
        hint = "Проверьте доступ аккаунта к API и выбранной модели."
    elif status == 429:
        hint = "Достигнут лимит запросов/квоты; проверьте кабинет."
    elif status in (400, 422):
        hint = "Сервер отклонил параметры запроса или схему функций."
    elif "ssl" in names or "certificate" in names or "connect" in names:
        hint = "Проверьте HTTPS-подключение и сертификаты Минцифры."
    elif "validation" in names:
        hint = "SDK отклонил формат запроса/ответа; проверьте совместимость версии."
    elif status is not None and status >= 500:
        hint = "Сервер временно недоступен; попробуйте позже."
    else:
        hint = "Проверьте версию SDK, модель и конфигурацию API."
    return f"{label}. {hint}"


def main() -> None:
    try:
        settings = Settings.from_env()
    except (ValueError, KeyError) as exc:
        print(f"Настройки: ошибка ({type(exc).__name__}), проверьте .env.")
        return
    if settings.provider != "gigachat":
        print("Сейчас AGENT_PROVIDER не gigachat. Исправьте .env.")
        return
    if not os.getenv("GIGACHAT_CREDENTIALS", "").strip():
        print("GIGACHAT_CREDENTIALS не задан. Укажите новый ключ в .env.")
        return
    print(f"Провайдер: {settings.provider}, модель: {settings.model}")
    try:
        import importlib.metadata as metadata
        print("Версия gigachat SDK:", metadata.version("gigachat"))
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole
    except Exception as exc:
        print("Импорт SDK: ОШИБКА —", describe_failure(exc))
        return

    try:
        with GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=settings.model,
            timeout=settings.timeout_sec,
            verify_ssl_certs=True,
        ) as client:
            print("1/3 Авторизация и список моделей...", flush=True)
            models = client.get_models()
            available = [m.id_ for m in models.data]
            print("   OK: доступны модели:", ", ".join(map(str, available[:10])))
            print("2/3 Простой текстовый запрос...", flush=True)
            prompt = Chat(
                model=settings.model,
                messages=[Messages(role=MessagesRole.USER, content="Ответь одним словом: тест.")],
            )
            client.chat(prompt)
            print("   OK: текстовый запрос обработан")
    except Exception as exc:
        print("   ОШИБКА —", describe_failure(exc))
        return

    try:
        print("3/3 Вызов модели со схемами инструментов...", flush=True)
        provider = GigaChatProvider(settings.model, settings.timeout_sec)
        reply = provider.complete(
            [
                {"role": "system", "content": "Ты агент. Используй доступные инструменты."},
                {"role": "user", "content": "Какие документы нужны для визы во Францию?"},
            ],
            tool_specifications(),
        )
        if reply.calls:
            print("   OK: модель запросила инструмент", ", ".join(c.name for c in reply.calls))
        else:
            print("   OK: модель ответила без инструментов (это допустимо при auto).")
    except Exception as exc:
        print("   ОШИБКА —", describe_failure(exc))
        print("   Первые два этапа успешны: ошибка связана с вызовом функций или адаптером.")


if __name__ == "__main__":
    main()
