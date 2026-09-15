from scenesmith.utils.openai_strict_schema import make_openai_strict_json_schema
def test_strict_schema_requires_all_properties_without_losing_nullability():
    source = {
        "type": "object",
        "properties": {
            "required_value": {"type": "string"},
            "optional_value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "default_value": {"type": "string", "default": "fallback"},
        },
        "required": ["required_value"],
        "$defs": {
            "Nested": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rank": {"type": "integer"},
                },
                "required": ["name"],
            }
        },
    }

    result = make_openai_strict_json_schema(source)

    assert result["required"] == ["required_value", "optional_value", "default_value"]
    assert result["$defs"]["Nested"]["required"] == ["name", "rank"]
    assert result["properties"]["optional_value"] == source["properties"]["optional_value"]
    assert source["required"] == ["required_value"]
