# Deployment

## Local
```bash
python run.py                    # FastAPI, :8000
streamlit run frontend/app.py    # Streamlit, :8501
```

## Docker
```bash
docker-compose up
```
Brings up backend (`:8000`) and frontend (`:8501`) together. Make sure
`data/models/model_final.pt` exists before building the image — it isn't
generated at build time.

## Edge / lower-latency inference
Run `python scripts/export_model.py` after training to produce
`data/models/model_final.torchscript.pt`. TorchScript drops the Python
dependency for inference and is what the "edge device optimization"
portfolio bullet refers to — swap `core/model.py`'s `torch.load` +
`load_state_dict` for `torch.jit.load(settings.TORCHSCRIPT_MODEL_PATH)`
when deploying to a constrained target.

## Environment variables
See `.env.example`. `DEVICE` should be set to `cuda` only on a host that
actually has a GPU + matching CUDA-enabled torch build; default `cpu`
works everywhere.
