from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _load_providers() -> dict[str, object]:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        return {}
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    return providers if isinstance(providers, dict) else {}


def _provider_entry_by_name(providers: dict[str, object], name: str) -> dict[str, object] | None:
    """Return providers[name], matching normalized custom names when needed."""
    if not name:
        return None
    entry = providers.get(name)
    if isinstance(entry, dict):
        return entry
    # Normalized match (find_custom_provider_identity returns normalized names)
    try:
        from hermes_cli.runtime_provider import _normalize_custom_provider_name

        target = _normalize_custom_provider_name(name)
    except Exception:
        target = name.strip().lower().replace(" ", "-")
    for ep_name, ep in providers.items():
        if not isinstance(ep, dict):
            continue
        try:
            from hermes_cli.runtime_provider import _normalize_custom_provider_name

            if _normalize_custom_provider_name(str(ep_name)) == target:
                return ep
        except Exception:
            if str(ep_name).strip().lower() == target:
                return ep
    return None


def _entry_has_timeout_config(entry: dict[str, object]) -> bool:
    if entry.get("request_timeout_seconds") is not None:
        return True
    if entry.get("stale_timeout_seconds") is not None:
        return True
    models = entry.get("models")
    if isinstance(models, dict):
        for model_cfg in models.values():
            if not isinstance(model_cfg, dict):
                continue
            if model_cfg.get("timeout_seconds") is not None:
                return True
            if model_cfg.get("stale_timeout_seconds") is not None:
                return True
    return False


def _entry_matches_model(entry: dict[str, object], model: str) -> bool:
    if entry.get("default_model") == model:
        return True
    models = entry.get("models")
    return isinstance(models, dict) and model in models


def _resolve_provider_config(
    provider_id: str,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, object] | None:
    """Resolve the providers: entry that owns timeout config for this runtime.

    Named custom endpoints resolve to the billing class ``\"custom\"`` at runtime,
    while timeout knobs live under the *named* key (e.g. ``dspark-deepseek``).
    This helper recovers that named entry via:

    1. Direct provider_id lookup (including ``custom:<name>``)
    2. base_url reverse-lookup via find_custom_provider_identity
    3. Model-name scan across providers (prefer entries with timeout config)
    """
    if not provider_id:
        return None

    providers = _load_providers()
    if not providers:
        return None

    pid = provider_id.strip()
    pid_lower = pid.lower()

    # 1a. custom:<name> identity → providers.<name>
    if pid_lower.startswith("custom:"):
        named = pid.split(":", 1)[1].strip()
        entry = _provider_entry_by_name(providers, named)
        if entry is not None:
            return entry

    # 1b. Direct named provider (not bare "custom")
    if pid_lower != "custom":
        entry = _provider_entry_by_name(providers, pid)
        if entry is not None:
            return entry

    # 2. base_url reverse-lookup (custom billing class + known endpoint)
    if base_url:
        try:
            from hermes_cli.runtime_provider import find_custom_provider_identity

            identity = find_custom_provider_identity(base_url)
        except Exception:
            identity = None
        if isinstance(identity, str) and identity.lower().startswith("custom:"):
            named = identity.split(":", 1)[1].strip()
            entry = _provider_entry_by_name(providers, named)
            if entry is not None:
                return entry

    # 3. Model-name scan — recovers named entry when provider is bare "custom"
    #    and call sites did not pass base_url.
    if model:
        matches: list[dict[str, object]] = []
        for entry in providers.values():
            if isinstance(entry, dict) and _entry_matches_model(entry, model):
                matches.append(entry)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            for entry in matches:
                if _entry_has_timeout_config(entry):
                    return entry
            return matches[0]

    return None


def get_provider_request_timeout(
    provider_id: str,
    model: str | None = None,
    base_url: str | None = None,
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    if not provider_id:
        return None

    provider_config = _resolve_provider_config(provider_id, model=model, base_url=base_url)
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))


def get_provider_stale_timeout(
    provider_id: str,
    model: str | None = None,
    base_url: str | None = None,
) -> float | None:
    """Return a configured non-stream / stream stale timeout in seconds, if any."""
    if not provider_id:
        return None

    provider_config = _resolve_provider_config(provider_id, model=model, base_url=base_url)
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
