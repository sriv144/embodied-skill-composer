from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


RunStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "interrupted",
    "resuming",
    "completed",
    "failed",
    "cancelled",
]


class _StrictEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RunCreatedEvent(_StrictEvent):
    event: Literal["run_created"]
    status: RunStatus
    matrix_id: str | None = None
    run_key: str | None = None

    @model_validator(mode="after")
    def validate_matrix_identity(self) -> RunCreatedEvent:
        if (self.matrix_id is None) != (self.run_key is None):
            raise ValueError("matrix_id and run_key must be provided together")
        return self


class RunClaimedEvent(_StrictEvent):
    event: Literal["run_claimed"]
    attempt: int = Field(ge=1)


class RunInterruptedEvent(_StrictEvent):
    event: Literal["run_interrupted"]
    reason: Literal[
        "stale_worker",
        "stale_worker_pid_reused",
        "stale_worker_terminated",
    ]


class RunStateEvent(_StrictEvent):
    event: Literal["cancel_requested", "cancelled"]
    error: str | None = None


class RunFailedEvent(_StrictEvent):
    event: Literal["failed"]
    error: str


class ResumeRequestedEvent(_StrictEvent):
    event: Literal["resume_requested"]
    checkpoint: str | None = None
    restart_from_beginning: bool | None = None

    @model_validator(mode="after")
    def validate_resume_source(self) -> ResumeRequestedEvent:
        checkpoint_resume = self.checkpoint is not None
        restart = self.restart_from_beginning is True
        if checkpoint_resume == restart:
            raise ValueError(
                "resume_requested requires exactly one resume source"
            )
        return self


class WorkerStartedEvent(_StrictEvent):
    event: Literal["worker_started"]
    pid: int = Field(gt=0)
    process_log: str


class WorkerLaunchFailedEvent(_StrictEvent):
    event: Literal["worker_launch_failed"]
    error: str


class WorkerExitedEvent(_StrictEvent):
    event: Literal["worker_exited"]
    return_code: int


class TrainingStartedEvent(_StrictEvent):
    event: Literal["training_started"]
    mode: Literal["inline", "subprocess"]
    attempt: int | None = Field(default=None, ge=1)
    pid: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_execution_identity(self) -> TrainingStartedEvent:
        if self.mode == "subprocess" and (
            self.attempt is None or self.pid is None
        ):
            raise ValueError("subprocess training events require attempt and pid")
        if self.mode == "inline" and (
            self.attempt is not None or self.pid is not None
        ):
            raise ValueError("inline training events cannot contain attempt or pid")
        return self


class TrainingResumedEvent(_StrictEvent):
    event: Literal["training_resumed"]
    transitions: int = Field(ge=0)
    checkpoint_path: str
    updates: int = Field(ge=0)
    episode_cursor: int = Field(ge=0)


class TrainingCheckpointEvent(_StrictEvent):
    event: Literal["checkpoint_saved", "behavior_cloning_checkpoint"]
    transitions: int = Field(ge=0)
    checkpoint_path: str
    snapshot_path: str
    policy_checkpoint_path: str | None
    fraction: float | None = Field(default=None, ge=0, le=1)
    target_transitions: int | None = Field(default=None, ge=0)


class BehaviorCloningCompletedEvent(_StrictEvent):
    event: Literal["behavior_cloning_complete"]
    transitions: int = Field(ge=0)
    loss: float
    epoch: int = Field(ge=0)


class PpoUpdateEvent(_StrictEvent):
    event: Literal["ppo_update"]
    transitions: int = Field(ge=0)
    update: int = Field(ge=1)
    loss_objective: float
    loss_critic: float
    loss_entropy: float
    mean_episode_return: float
    rollout_terminal_fraction: float = Field(ge=0, le=1)


class TrainingCompletedEvent(_StrictEvent):
    event: Literal["training_completed"]
    artifacts: dict[str, object]


class EvaluationStartedEvent(_StrictEvent):
    event: Literal["evaluation_started"]


class EvaluationCompletedEvent(_StrictEvent):
    event: Literal["evaluation_completed"]
    artifacts: dict[str, object]


class MatrixEvaluationStartedEvent(_StrictEvent):
    event: Literal["matrix_evaluation_started"]
    mode: Literal["subprocess"]
    attempt: int = Field(ge=1)
    pid: int = Field(gt=0)
    matrix_id: str


class MatrixEvaluationCompletedEvent(_StrictEvent):
    event: Literal["matrix_evaluation_completed"]
    matrix_id: str
    evaluation_id: str
    acceptance: dict[str, object]
    ablations: list[dict[str, object]]


class SelectionEvidenceUnavailableEvent(_StrictEvent):
    event: Literal["selection_evidence_unavailable"]
    path: str
    effect: Literal["transitions_to_95_ablation_evidence_omitted"]


RunEventPayload = Annotated[
    RunCreatedEvent
    | RunClaimedEvent
    | RunInterruptedEvent
    | RunStateEvent
    | RunFailedEvent
    | ResumeRequestedEvent
    | WorkerStartedEvent
    | WorkerLaunchFailedEvent
    | WorkerExitedEvent
    | TrainingStartedEvent
    | TrainingResumedEvent
    | TrainingCheckpointEvent
    | BehaviorCloningCompletedEvent
    | PpoUpdateEvent
    | TrainingCompletedEvent
    | EvaluationStartedEvent
    | EvaluationCompletedEvent
    | MatrixEvaluationStartedEvent
    | MatrixEvaluationCompletedEvent
    | SelectionEvidenceUnavailableEvent,
    Field(discriminator="event"),
]

_PAYLOAD_ADAPTER: TypeAdapter[RunEventPayload] = TypeAdapter(RunEventPayload)


class RunEventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sequence: int = Field(ge=1)
    created_at: str
    payload: RunEventPayload

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value


def validate_run_event_payload(
    payload: Mapping[str, object],
) -> RunEventPayload:
    return _PAYLOAD_ADAPTER.validate_python(dict(payload), strict=True)


def dump_run_event_payload(payload: RunEventPayload) -> dict[str, object]:
    return payload.model_dump(mode="json", exclude_unset=True)
