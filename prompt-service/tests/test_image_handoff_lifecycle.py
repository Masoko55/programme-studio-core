"""The prompt job mirrors the top-level image-workflow lifecycle."""

import asyncio
from unittest.mock import AsyncMock, call, patch

import pytest

with patch("fastapi.dependencies.utils.ensure_multipart_is_installed"):
    from app.main import hand_off_to_image_service


@pytest.mark.parametrize(
    ("result", "status", "stage", "error"),
    [
        ({"status": "processing", "current_stage": "generation_pending",
          "error": "3/9 background candidates completed; 0 failed; 6 remain pending."},
         "processing", "generation_pending",
         "3/9 background candidates completed; 0 failed; 6 remain pending."),
        ({"status": "processing", "current_stage": None, "error": None},
         "processing", "image_workflow_running", None),
        ({"status": "awaiting_selection", "current_stage": "awaiting_selection", "error": None},
         "awaiting_selection", "awaiting_selection", None),
        ({"status": "awaiting_selection", "current_stage": "backgrounds_complete_with_failures",
          "error": "3/9 background candidates completed; 6 failed; 0 remain pending."},
         "awaiting_selection", "backgrounds_complete_with_failures",
         "3/9 background candidates completed; 6 failed; 0 remain pending."),
        ({"status": "awaiting_selection", "current_stage": "", "error": None},
         "awaiting_selection", "awaiting_selection", None),
        ({"status": "failed", "current_stage": "failed",
          "error": "No valid background candidates were produced."},
         "failed", "failed", "No valid background candidates were produced."),
        ({"status": "failed", "current_stage": None, "error": "Renderer failed."},
         "failed", "image_workflow_failed", "Renderer failed."),
        ({"status": "failed", "current_stage": None, "error": None},
         "failed", "image_workflow_failed",
         "Image workflow reported failure without an error."),
    ],
)
def test_successful_handoff_persists_returned_lifecycle(result, status, stage, error):
    result["image_job"] = {"status": "failed", "current_stage": "nested_stage"}
    with patch("app.main.run_image_workflow", new_callable=AsyncMock, return_value=result) as workflow, \
         patch("app.main.update_job_status") as update:
        asyncio.run(hand_off_to_image_service("ABCDEF-123456"))
    workflow.assert_awaited_once_with("ABCDEF-123456")
    assert update.call_args_list == [
        call("ABCDEF-123456", status="processing", current_stage="image_workflow_running", error=None),
        call("ABCDEF-123456", status=status, current_stage=stage, error=error),
    ]


@pytest.mark.parametrize("result", [
    {"status": "mystery_state", "current_stage": "something", "error": None},
    {"current_stage": "generation_pending", "error": None},
    {"status": [], "current_stage": "generation_pending", "error": None},
])
def test_invalid_top_level_status_fails_safely(result):
    with patch("app.main.run_image_workflow", new_callable=AsyncMock, return_value=result), \
         patch("app.main.update_job_status") as update:
        with pytest.raises(RuntimeError, match="Unexpected image workflow status"):
            asyncio.run(hand_off_to_image_service("ABCDEF-123456"))
    assert update.call_args.kwargs["status"] == "failed"
    assert update.call_args.kwargs["current_stage"] == "image_workflow_invalid_state"
    assert len(update.call_args.kwargs["error"]) <= 200


def test_transport_failure_keeps_existing_failure_path():
    with patch("app.main.run_image_workflow", new_callable=AsyncMock,
               side_effect=RuntimeError("handoff failed")), \
         patch("app.main.update_job_status") as update:
        with pytest.raises(RuntimeError, match="handoff failed"):
            asyncio.run(hand_off_to_image_service("ABCDEF-123456"))
    assert update.call_args.kwargs == {
        "status": "failed", "current_stage": "image_workflow_failed", "error": "handoff failed",
    }
