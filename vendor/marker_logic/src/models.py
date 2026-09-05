from surya.fast_layout import FastLayoutPredictor
from surya.inference import SuryaInferenceManager
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor


def create_model_dict(
    device=None,
    dtype=None,
    attention_implementation: str | None = None,
    inference_manager: SuryaInferenceManager | None = None,
    inference_backend: str | None = None,
) -> dict:
    """Build the predictor set marker uses.

    Two converter modes share this set:
      - balanced (default): the VLM ``layout_model`` + full-page
        ``recognition_model``.
      - fast: the lightweight rf-detr ``fast_layout_model`` + block-mode OCR,
        OCRing only garbled/empty content.

    All the heavy models run in shared surya servers, so every predictor here is
    a thin client and marker worker processes stay light (this is what lets many
    workers share one GPU without each loading a model):
      - ``layout_model`` / ``recognition_model``: clients of the VLM
        ``inference_manager`` (lazy - only spawns a server when OCR is actually
        needed, so clean digital docs in fast mode never start it).
      - ``fast_layout_model``: a client of the shared fast-layout server (owns
        the single rf-detr instance + continuous-batches across all clients).
        Holds no model; spawns/attaches the server on first call, not here.
      - ``ocr_error_model``: a client of the shared ocr-error server (the
        DistilBert model runs in one server process; workers POST text to it).

    Every predictor here is now a thin server client, so marker worker
    processes hold NO models and many can share one GPU without each loading a
    copy. Tables are reconstructed from the PDF text layer (pdftext heuristics)
    for digital pages and from full-page OCR for scanned pages - there is no
    dedicated table model. ``device``/``dtype``/``attention_implementation`` are
    accepted for call-site compatibility (e.g. worker_init) but are now no-ops -
    model devices are set server-side.
    """
    manager = inference_manager or SuryaInferenceManager(method=inference_backend)
    return {
        "inference_manager": manager,
        "layout_model": LayoutPredictor(manager),
        # Thin client of the shared fast-layout server (holds no model; the
        # server owns the one rf-detr instance and continuous-batches across
        # workers). Cheap to construct even in balanced mode, where it is never
        # called. The reading-order head defaults off: pdftext pages are
        # reordered from the PDF's character order (LineBuilder), so learned
        # order is requested per call only for the pages that need it.
        "fast_layout_model": FastLayoutPredictor(use_order=False),
        "recognition_model": RecognitionPredictor(manager),
        # Thin client of the shared ocr-error server (the DistilBert model runs
        # in one server process; N marker workers just POST text to it). Holds
        # no model, so it doesn't add per-worker GPU load. device/dtype/
        # attention_implementation are server-side now and ignored here (kept in
        # the signature for call-site compatibility, e.g. worker_init).
        "ocr_error_model": OCRErrorPredictor(),
    }


def shutdown_models(model_dict: dict) -> None:
    """Stop the shared inference server if this process spawned it."""
    manager = model_dict.get("inference_manager")
    if manager is not None:
        manager.stop()
