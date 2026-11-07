"""Recover only top-level, extras-only invocation failures (mrl-infra #138)."""

import copy
import logging

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from openhands.sdk.tool import Action
from openhands.sdk.tool.builtins.think import ThinkAction, ThinkTool


class BoundaryAliasAction(Action):
    thought: str = Field(
        validation_alias=AliasChoices(
            "thought", "reasoning", AliasPath("request", "text")
        )
    )


class BoundaryNestedValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class BoundaryNestedAction(Action):
    details: BoundaryNestedValue


class BoundaryDictionaryAction(Action):
    data: dict[str, object]


class BoundaryPermissiveAction(Action):
    model_config = ConfigDict(extra="allow", frozen=True)
    thought: str


class BoundaryResult(BaseModel):
    success: bool


@pytest.fixture
def think_tool():
    return ThinkTool.create()[0]


@pytest.mark.parametrize(
    "extras", [{"timeout": 45}, {"timeout": 45, "other": "private"}]
)
def test_extras_only_are_removed_without_mutating_input(think_tool, extras, caplog):
    arguments = {"thought": 'two words\n"quoted" && unchanged', **extras}
    original = copy.deepcopy(arguments)
    with caplog.at_level(logging.WARNING):
        action = think_tool.action_from_arguments(arguments)
    assert action.thought == arguments["thought"]
    assert arguments == original
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "think" in message
    assert all(key in message for key in extras)
    assert "private" not in message
    assert arguments["thought"] not in message


def test_valid_call_has_no_warning(think_tool, caplog):
    action = think_tool.action_from_arguments({"thought": "unchanged"})
    assert action.thought == "unchanged"
    assert not caplog.records


def test_direct_model_validation_stays_strict():
    with pytest.raises(ValidationError):
        ThinkAction.model_validate({"thought": "unchanged", "timeout": 45})


@pytest.mark.parametrize(
    "arguments", [{"timeout": 45}, {"thought": 123, "timeout": 45}]
)
def test_other_validation_errors_are_not_repaired(think_tool, arguments, caplog):
    with pytest.raises(ValidationError) as error:
        think_tool.action_from_arguments(arguments)
    assert any(e["type"] != "extra_forbidden" for e in error.value.errors())
    assert not caplog.records


@pytest.mark.parametrize(
    "arguments",
    [
        {"thought": "unchanged", "timeout": 45},
        {"reasoning": "unchanged", "timeout": 45},
        {"request": {"text": "unchanged"}, "timeout": 45},
    ],
)
def test_validation_aliases_are_preserved(think_tool, arguments):
    tool = think_tool.model_copy(update={"action_type": BoundaryAliasAction})
    original = copy.deepcopy(arguments)
    action = tool.action_from_arguments(arguments)
    assert action.thought == "unchanged"
    assert arguments == original


@pytest.mark.parametrize("extras", [{}, {"timeout": 45}])
def test_nested_errors_are_not_filtered(think_tool, extras, caplog):
    tool = think_tool.model_copy(update={"action_type": BoundaryNestedAction})
    arguments = {"details": {"text": "unchanged", "unexpected": "private"}, **extras}
    original = copy.deepcopy(arguments)
    with pytest.raises(ValidationError) as error:
        tool.action_from_arguments(arguments)
    assert any(e["loc"] == ("details", "unexpected") for e in error.value.errors())
    assert arguments == original
    assert not caplog.records


def test_declared_dictionary_values_are_untouched(think_tool):
    tool = think_tool.model_copy(update={"action_type": BoundaryDictionaryAction})
    data = {"timeout": 45, "nested": {"arbitrary": ["two words", "line\nbreak"]}}
    action = tool.action_from_arguments({"data": data, "unexpected": True})
    assert action.data == data


def test_permissive_action_keeps_its_extras(think_tool, caplog):
    tool = think_tool.model_copy(update={"action_type": BoundaryPermissiveAction})
    action = tool.action_from_arguments({"thought": "unchanged", "timeout": 45})
    assert action.model_extra == {"timeout": 45}
    assert not caplog.records


def test_structured_output_is_preserved(think_tool):
    tool = think_tool.set_response_schema(BoundaryResult)
    action = tool.action_from_arguments(
        {"thought": "unchanged", "success": True, "timeout": 45}
    )
    assert action.thought == "unchanged"
    assert action.structured_output == {"success": True}


def test_missing_structured_output_is_not_repaired(think_tool, caplog):
    tool = think_tool.set_response_schema(BoundaryResult)
    with pytest.raises(ValidationError):
        tool.action_from_arguments({"thought": "unchanged", "timeout": 45})
    assert not caplog.records
