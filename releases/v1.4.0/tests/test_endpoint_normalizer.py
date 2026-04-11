from controller.core.endpoint_normalizer import normalize_provider_base_url, build_url


def test_normalize_provider_base_url_strips_known_suffixes():
    assert normalize_provider_base_url("https://api.example.com/v1/chat/completions") == "https://api.example.com"
    assert normalize_provider_base_url("https://api.example.com/v1/models") == "https://api.example.com"
    assert normalize_provider_base_url("https://api.example.com/responses") == "https://api.example.com"


def test_build_url_uses_normalized_base():
    assert build_url("https://api.example.com/v1/chat/completions", "/chat/completions") == "https://api.example.com/chat/completions"
