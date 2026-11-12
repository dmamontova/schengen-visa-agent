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
    certificate_chain = []
    cause = exc
    for _ in range(5):
        if cause is None:
            break
        certificate_chain.append(type(cause).__name__.lower())
        cause = cause.__cause__ or cause.__context__
    is_ssl = any("ssl" in v or "certificat" in v for v in certificate_chain)
    if status == 401:
        hint = "Проверьте новый Authorization Key и срок действия авторизации."
    elif status == 403:
        hint = "Проверьте доступ аккаунта к API и выбранной модели."
    elif status == 429:
        hint = "Достигнут лимит запросов/квоты; проверьте кабинет."
    elif status in (400, 422):
        hint = "Сервер отклонил параметры запроса или схему функций."
    elif is_ssl:
        hint = "Похоже на ошибку TLS. Настройте корневой сертификат Минцифры через GIGACHAT_CA_BUNDLE_FILE."
    elif "connect" in names or "connect" in " ".join(certificate_chain):
        hint = "Соединение не установлено. Проверьте доступ к API и OAuth-хосту, DNS, VPN и TLS."
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
    credentials = os.getenv("GIGACHAT_CREDENTIALS", "")
    if not credentials.strip():
        print("GIGACHAT_CREDENTIALS не задан. Укажите новый ключ в .env.")
        return
    # Do not print credential values or full exception messages: both can leak secrets.
    for name, value in (
        ("GIGACHAT_CREDENTIALS", credentials),
        ("GIGACHAT_SCOPE", os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")),
        ("AGENT_MODEL", settings.model),
    ):
        if not value.isascii():
            print(f"Конфигурация: {name} содержит не-ASCII символы "
                  "(например, русскую заглушку). Замените значение в .env.")
            return
        if value != value.strip() or any(ch.isspace() for ch in value):
            print(f"Конфигурация: {name} содержит лишние пробелы. "
                  "Скопируйте значение заново.")
            return
    if credentials in ("СЮДА_НОВЫЙ_КЛЮЧ", "твой_ключ", "твой_настоящий_ключ"):
        print("Конфигурация: вместо ключа указана заглушка.")
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
            ca_bundle_file=os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None,
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
        print("3/3 Проверка пользовательских функций...", flush=True)
        # Minimal function: if even this is rejected, the model/account may
        # not support functions. No credentials or raw response is printed.
        with GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=settings.model,
            timeout=settings.timeout_sec,
            verify_ssl_certs=True,
            ca_bundle_file=os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None,
        ) as client:
            client.chat(Chat(
                model=settings.model,
                messages=[Messages(
                    role=MessagesRole.USER, content="Проверь страну Франция."
                )],
                functions=[{
                    "name": "check_country",
                    "description": "Определи страну пользователя",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "description": "Страна",
                            },
                        },
                        "required": ["country"],
                    },
                }],
                function_call="auto",
            ))
        print("   OK: минимальная функция принята сервером")
    except Exception as exc:
        print("   ОШИБКА минимальной функции —", describe_failure(exc))
        print(
            "   Проблема не в наших Pydantic-схемах: даже простой "
            "пример не работает. Возможна неподдерживаемая модель или "
            "ограничение доступа к функциям. Проверьте поддержку в Sber Studio."
        )
        return
    try:
        print("   Проверяем три инструмента проекта...", flush=True)
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
            print("   OK: модель ответила без инструментов (допустимо при auto).")
    except Exception as exc:
        print("   ОШИБКА схем проекта —", describe_failure(exc))
        print("   Минимальная функция работает, проверьте адаптер и контракты.")


if __name__ == "__main__":
    main()
