# Light on purpose: rtldoc's actual runtime dependencies are just PyMuPDF
# and numpy (both have prebuilt wheels, no compiler needed) -- opencv and
# the Arabic-shaping libs in pyproject.toml's extras aren't imported by the
# pipeline at all today, see pyproject.toml for why. Anything importing
# cv2 (the not-yet-wired scanned-page fallback) needs `--extra-index cv`
# added to the pip install line below.
#
# tesseract-ocr IS installed here: ocr.py's scanned-page fallback shells out
# to the `tesseract` binary directly (no pip package), so it needs to be on
# PATH for scanned pages to OCR inside the container.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY rtldoc/ ./rtldoc/

RUN pip install --no-cache-dir .

ENTRYPOINT ["rtldoc"]
CMD ["--help"]
