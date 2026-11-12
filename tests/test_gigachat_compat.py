from schengen_agent.schemas import tool_specifications
from schengen_agent.gigachat_compat import to_gigachat_functions


def test_gigachat_functions_have_supported_plain_schema():
    functions = to_gigachat_functions(tool_specifications())
    assert len(functions) == 3
    assert {f["name"] for f in functions} == {
        "get_visa_statistics", "get_visa_requirements", "search_visa_cases",
    }
    for fun in functions:
        schema = fun["parameters"]
        assert schema["type"] == "object"
        for key, field in schema["properties"].items():
            assert field["type"] in {"string", "integer", "boolean"}
            assert "anyOf" not in field
            assert "default" not in field
            assert "title" not in field
        assert schema["required"] == ["country"]


def test_optional_fields_not_required_for_gigachat():
    specs = to_gigachat_functions(tool_specifications())
    stats = next(f for f in specs if f["name"] == "get_visa_statistics")
    assert stats["parameters"]["properties"]["period_days"]["type"] == "integer"
    assert "city" not in stats["parameters"]["required"]
