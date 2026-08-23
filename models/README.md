# Production model artifacts

This directory is a read-only mount in production. Do not commit model binaries or biometric data.
Place the approved ONNX files and `manifest.json` here through the deployment pipeline, then run
`prom-verify-artifacts` before rollout.
