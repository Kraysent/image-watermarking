from pathlib import Path

import huggingface_hub
import mlflow
import mlflow.artifacts
import mlflow.environment_variables as mlflow_env
import mlflow.tracking


def latest_checkpoint_artifact(run_id: str) -> str:
    client = mlflow.tracking.MlflowClient()
    artifacts = [
        artifact.path
        for artifact in client.list_artifacts(run_id)
        if Path(artifact.path).name.startswith("epoch_") and artifact.path.endswith(".pt")
    ]
    if not artifacts:
        raise FileNotFoundError(f"checkpoint artifact not found for run: {run_id}")

    def epoch_number(path: str) -> int:
        return int(Path(path).stem.removeprefix("epoch_"))

    return max(artifacts, key=epoch_number)


def push_checkpoint(
    *,
    run_id: str,
    repo_id: str,
    tracking_uri: str,
    private: bool = False,
    client_cert_path: str | None = None,
    server_cert_path: str | None = None,
) -> str:
    if client_cert_path is not None:
        mlflow_env.MLFLOW_TRACKING_CLIENT_CERT_PATH.set(client_cert_path)
    if server_cert_path is not None:
        mlflow_env.MLFLOW_TRACKING_SERVER_CERT_PATH.set(server_cert_path)

    mlflow.set_tracking_uri(tracking_uri)
    artifact_path = latest_checkpoint_artifact(run_id)
    checkpoint_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path))

    api = huggingface_hub.HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
