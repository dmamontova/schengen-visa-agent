"""GigaChat function schema adapter.

GigaChat's documented function parameters are a narrower JSON Schema subset
than Pydantic's full model_json_schema(). Pydantic remains the authority for
runtime validation; this module only changes what the model sees.
"""
from __future__ import annotations


def to_gigachat_parameters(schema: dict) -> dict:
    properties = {}
    for name, original in schema.get("properties", {}).items():
        field = dict(original)
        if "anyOf" in field:
            # Optional[T] in Pydantic is often anyOf[T, null]; in the function
            # description the field is simply optional and can be omitted.
            options = [choice for choice in field["anyOf"] if choice.get("type") != "null"]
            if len(options) != 1:
                raise ValueError(f"Unsupported union for {name}")
            field = {**options[0], "description": original.get("description", "")}
        result = {"type": field["type"]}
        if field.get("description"):
            result["description"] = field["description"]
        if "enum" in field:
            result["enum"] = field["enum"]
        lower, upper = original.get("minimum"), original.get("maximum")
        if lower is not None or upper is not None:
            bounds = "Allowed range: " + " to ".join(
                str(v) for v in (lower, upper) if v is not None
            )
            result["description"] = (result.get("description", "") + ". " + bounds).strip(". ")
        properties[name] = result
    return {
        "type": "object",
        "properties": properties,
        "required": schema.get("required", []),
    }


def to_gigachat_functions(specifications: list[dict]) -> list[dict]:
    return [
        {
            "name": item["function"]["name"],
            "description": item["function"]["description"],
            "parameters": to_gigachat_parameters(item["function"]["parameters"]),
        }
        for item in specifications
    ]
