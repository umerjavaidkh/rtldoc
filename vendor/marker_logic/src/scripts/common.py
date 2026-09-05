import base64
import io
import sys

from PIL import Image
import click
import pypdfium2
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.models import create_model_dict
from marker.settings import settings


@st.cache_data()
def parse_args():
    # Use to grab common cli options
    @ConfigParser.common_options
    def options_func():
        pass

    def extract_click_params(decorated_function):
        if hasattr(decorated_function, "__click_params__"):
            return decorated_function.__click_params__
        return []

    cmd = CustomClickPrinter("Marker app.")
    extracted_params = extract_click_params(options_func)
    cmd.params.extend(extracted_params)
    ctx = click.Context(cmd)
    try:
        cmd_args = sys.argv[1:]
        cmd.parse_args(ctx, cmd_args)
        return ctx.params
    except click.exceptions.ClickException as e:
        return {"error": str(e)}


@st.cache_resource()
def load_models():
    return create_model_dict()


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


def img_to_html(img, img_alt):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=settings.OUTPUT_IMAGE_FORMAT)
    img_bytes = img_bytes.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    img_html = f'<img src="data:image/{settings.OUTPUT_IMAGE_FORMAT.lower()};base64,{encoded}" alt="{img_alt}" style="max-width: 100%;">'
    return img_html


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=96):
    if "pdf" in pdf_file.type:
        doc = open_pdf(pdf_file)
        page = doc[page_num]
        png_image = (
            page.render(
                scale=dpi / 72,
            )
            .to_pil()
            .convert("RGB")
        )
    else:
        png_image = Image.open(pdf_file).convert("RGB")
    return png_image


@st.cache_data()
def page_count(pdf_file: UploadedFile):
    if "pdf" in pdf_file.type:
        doc = open_pdf(pdf_file)
        return len(doc) - 1
    else:
        return 1
