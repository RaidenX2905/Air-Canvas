import json
import os
import platform
import re
import subprocess
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Air Canvas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom minimal CSS to remove default Streamlit margins and padding
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden; display: none !important;}
        footer {visibility: hidden; display: none !important;}
        header {visibility: hidden; display: none !important;}
        [data-testid="stHeader"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }
        iframe {
            display: block;
            border: none;
            width: 100vw;
            height: 100vh;
        }
        body {
            overflow: hidden;
            background-color: #0b0f17;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


def clean_hw_name(name: str, kind: str) -> str:
    """Format verbose hardware strings into concise, minimal titles."""
    if not name or name == "None Detected":
        return ""
    s = name
    s = re.sub(r"\[[0-9a-fA-F:]+\]", "", s)
    s = re.sub(r"\(rev\s+[0-9a-fA-F]+\)", "", s)
    s = re.sub(r"Corporation", "", s, flags=re.I)
    s = re.sub(r"Company", "", s, flags=re.I)

    if kind == "cpu":
        s = re.sub(r"\(R\)", "", s, flags=re.I)
        s = re.sub(r"\(TM\)", "", s, flags=re.I)
        s = re.sub(r"\b\d+th\s+Gen\b", "", s, flags=re.I)
        s = re.sub(r"Core\s+", "", s, flags=re.I)
        s = re.sub(r"with Radeon Graphics.*", "", s, flags=re.I)
        s = re.sub(r"Processor.*", "", s, flags=re.I)
        s = re.sub(r"CPU\s*@.*", "", s, flags=re.I)
    elif kind == "dgpu":
        s = re.sub(r"\[(.*?)\]", r"\1", s)
        s = re.sub(r"\b[A-Z]{1,2}\d{3,4}\w*\b", "", s)
        s = re.sub(r"GeForce\s+", "", s, flags=re.I)
        s = re.sub(r"Max-Q\s*/\s*Mobile", "", s, flags=re.I)
        s = re.sub(r"Laptop GPU", "", s, flags=re.I)
        s = re.sub(r"Mobile", "", s, flags=re.I)
        s = re.sub(r"/PCIe.*", "", s, flags=re.I)
    elif kind == "igpu":
        s = re.sub(r"Alder Lake.*?\[", "[", s, flags=re.I)
        s = re.sub(r"Raptor Lake.*?\[", "[", s, flags=re.I)
        s = re.sub(r"Tiger Lake.*?\[", "[", s, flags=re.I)
        s = re.sub(r"\[(.*?)\]", r"\1", s)
        s = re.sub(r"\bGT\d+\b", "", s)
        if "iris" in s.lower() and "intel" not in s.lower():
            s = "Intel " + s

    s = re.sub(r"\s+", " ", s).strip()
    return s


def probe_host_hardware() -> dict:
    """Accurately probe CPU, dGPU, and iGPU details across OS platforms."""
    raw_cpu = f"{platform.processor() or 'CPU'}"
    raw_dgpu = "None Detected"
    raw_igpu = "None Detected"

    # Linux probing
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        raw_cpu = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        try:
            out = subprocess.check_output(["lspci", "-nnk"], stderr=subprocess.DEVNULL, universal_newlines=True)
            for line in out.splitlines():
                l_low = line.lower()
                if any(k in l_low for k in ["vga", "3d", "display"]):
                    clean = line.split(":", 2)[-1].strip()
                    if any(d in l_low for d in ["nvidia", "geforce", "rtx", "gtx", "radeon rx"]):
                        raw_dgpu = clean
                    elif any(ig in l_low for ig in ["intel", "iris", "uhd", "integrated", "radeon graphics", "vega"]):
                        raw_igpu = clean
        except Exception:
            pass

    # Windows probing
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            )
            controllers = [c.strip() for c in out.splitlines() if c.strip() and "name" not in c.lower()]
            for c in controllers:
                c_low = c.lower()
                if any(d in c_low for d in ["nvidia", "geforce", "rtx", "gtx", "radeon rx"]):
                    raw_dgpu = c
                elif any(ig in c_low for ig in ["intel", "iris", "uhd", "radeon graphics"]):
                    raw_igpu = c
        except Exception:
            pass

    # macOS probing
    elif platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], stderr=subprocess.DEVNULL, universal_newlines=True)
            for line in out.splitlines():
                if "Chipset Model:" in line:
                    chip = line.split(":", 1)[1].strip()
                    if "apple" in chip.lower():
                        raw_dgpu = chip
                        raw_igpu = chip
                    else:
                        raw_dgpu = chip
        except Exception:
            pass

    return {
        "cpu": clean_hw_name(raw_cpu, "cpu"),
        "dgpu": clean_hw_name(raw_dgpu, "dgpu"),
        "igpu": clean_hw_name(raw_igpu, "igpu"),
        "platform": platform.system(),
    }


@st.cache_resource
def load_frontend_html() -> str:
    """Read frontend files once and cache in memory for instantaneous zero-overhead serving."""
    html_path = FRONTEND_DIR / "index.html"
    css_path = FRONTEND_DIR / "style.css"
    engine_path = FRONTEND_DIR / "canvas_engine.js"
    app_path = FRONTEND_DIR / "app.js"

    html_content = html_path.read_text(encoding="utf-8")
    css_content = css_path.read_text(encoding="utf-8")
    engine_content = engine_path.read_text(encoding="utf-8")
    app_content = app_path.read_text(encoding="utf-8")

    host_hw = probe_host_hardware()
    hw_injection_script = f"""
    <script>
      window.__AIR_CANVAS_HOST_HW__ = {json.dumps(host_hw)};
    </script>
    """

    # In-line external relative assets into a single self-contained bundle
    html_bundle = html_content.replace(
        '<link rel="stylesheet" href="style.css">',
        f"<style>\n{css_content}\n</style>",
    )
    html_bundle = html_bundle.replace(
        '<script src="canvas_engine.js"></script>',
        f"{hw_injection_script}\n<script>\n{engine_content}\n</script>",
    )
    html_bundle = html_bundle.replace(
        '<script type="module" src="app.js"></script>',
        f'<script type="module">\n{app_content}\n</script>',
    )
    return html_bundle

frontend_markup = load_frontend_html()

# Render component filling the viewport
components.html(frontend_markup, height=920, scrolling=False)

