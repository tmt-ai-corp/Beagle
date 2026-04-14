import os


if os.environ.get("SPECFORGE_DISABLE_SGLANG_BITA_PATCH", "").lower() not in {
    "1",
    "true",
    "yes",
}:
    try:
        from specforge.integrations.sglang_bita_patch import apply_sglang_bita_patch
    except Exception:
        apply_sglang_bita_patch = None

    if apply_sglang_bita_patch is not None:
        try:
            apply_sglang_bita_patch()
        except Exception:
            pass
