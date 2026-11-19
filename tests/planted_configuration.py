"""Secrets planted in the run configuration of real Inspect logs, for tests that look for them.

An Inspect log records how its evaluation was run, and several of those fields can hold a
credential, an internal endpoint or a local path: the generate config's ``extra_headers`` and
``extra_body``, the base URL, the model arguments, each model role's configuration, the
sandbox configuration, the solver and task arguments, the task attributes, the metadata and the
origin of the git checkout. A report may name these fields and fingerprint them; it may never
copy a value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from inspect_ai.log import EvalLog, EvalRevision
from inspect_ai.model import GenerateConfig, ModelConfig
from inspect_ai.util import SandboxEnvironmentSpec

# The fields of evals_df that plant_configuration writes, as a plan names them under schema 2.0.
PLANTED_FIELDS = (
    "model_generate_config",
    "model_base_url",
    "model_args",
    "model_roles",
    "sandbox_config",
    "solver_args",
    "task_arg_secret",
    "task_attribs",
    "metadata",
    "git_origin",
)


def plant_configuration(log: EvalLog, tag: str) -> tuple[str, ...]:
    """Write a secret ending in ``tag`` into every configuration field of ``log``.

    Returns the secrets. Each is distinct, so a test that finds one knows which field it came
    from. The generate config also gets a temperature, an ordinary setting that may be
    fingerprinted like the rest.
    """

    secrets = {
        "header": f"sk-planted-header-{tag}",
        "body": f"planted-body-{tag}",
        "host": f"planted-host-{tag}",
        "model_arg": f"planted-model-arg-{tag}",
        "role_header": f"planted-role-header-{tag}",
        "role_host": f"planted-role-host-{tag}",
        "role_arg": f"planted-role-arg-{tag}",
        "sandbox": f"planted-sandbox-{tag}",
        "solver_arg": f"planted-solver-arg-{tag}",
        "task_arg": f"planted-task-arg-{tag}",
        "attribute": f"planted-attribute-{tag}",
        "metadata": f"planted-metadata-{tag}",
        "git": f"planted-git-password-{tag}",
    }
    spec = log.eval
    spec.model_generate_config = GenerateConfig(
        temperature=0.5,
        extra_headers={"Authorization": f"Bearer {secrets['header']}"},
        extra_body={"api_token": secrets["body"]},
    )
    spec.model_base_url = f"http://{secrets['host']}.invalid:8000/v1"
    spec.model_args = {"api_key": secrets["model_arg"]}
    spec.model_roles = {
        "grader": ModelConfig(
            model="mockllm/grader",
            config=GenerateConfig(extra_headers={"X-Api-Key": secrets["role_header"]}),
            base_url=f"http://{secrets['role_host']}.invalid/v1",
            args={"api_key": secrets["role_arg"]},
        )
    }
    spec.sandbox = SandboxEnvironmentSpec(
        type="docker", config=f"/home/{secrets['sandbox']}/compose.yaml"
    )
    spec.solver_args = {"api_token": secrets["solver_arg"]}
    spec.solver_args_passed = {"api_token": secrets["solver_arg"]}
    spec.task_args = {"secret": secrets["task_arg"]}
    spec.task_args_passed = {"secret": secrets["task_arg"]}
    spec.task_attribs = {"endpoint": secrets["attribute"]}
    spec.metadata = {"api_token": secrets["metadata"]}
    spec.revision = EvalRevision(
        type="git",
        origin=f"https://user:{secrets['git']}@git.invalid/evals.git",
        commit="0123abc",
    )
    return tuple(secrets.values())


def leaks(texts: Mapping[str, str], secrets: Iterable[str]) -> list[tuple[str, str]]:
    """Each (name, secret) pair where a planted secret occurs in a text."""

    planted = tuple(secrets)
    return [(name, secret) for name, text in texts.items() for secret in planted if secret in text]


def written_texts(*directories: Path) -> dict[str, str]:
    """Every file under the directories, decoded leniently, by its path."""

    return {
        str(path): path.read_bytes().decode("utf-8", errors="replace")
        for directory in directories
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
