"""
An in-notebook control panel built on ipywidgets.

Contains no generation logic. Every decision about what to generate
lives in `session`, and every decision about how to generate it lives
in the pipeline; this module only wires input elements to those and
puts the result on screen.

The image is drawn into a single output area that is cleared before
each redraw, so a new result replaces the previous one rather than
accumulating down the notebook.
"""

from __future__ import annotations

import logging
import traceback

from ..pipeline import Pipeline
from .session import (
    GenerationOutcome,
    RANDOM_SEED_SENTINEL,
    build_request,
    describe_outcome,
    resolution_labels,
    to_display_image,
)


PROMPT_PLACEHOLDER = "Describe the image to generate"
GENERATE_BUTTON_LABEL = "Generate"
RANDOM_SEED_LABEL = "Random seed"

# Wide enough that a typical prompt is visible without scrolling.
PROMPT_ROWS = 3


def build_control_panel(pipeline: Pipeline, logger: logging.Logger):
    """
    Build and return the control panel.

    Importing ipywidgets inside the function rather than at module scope
    keeps it an optional dependency: someone using the pipeline from a
    script should not need a notebook toolkit installed.

    Returns the assembled widget. The caller displays it, rather than
    this function displaying it as a side effect, so that it can be
    embedded in a larger layout.
    """
    try:
        import ipywidgets
        from IPython.display import clear_output, display
    except ImportError as error:
        raise ImportError(
            "ipywidgets and IPython are required for the notebook interface. "
            "Install them with: pip install ipywidgets"
        ) from error

    buckets = pipeline.resolution_buckets

    prompt_input = ipywidgets.Textarea(
        placeholder=PROMPT_PLACEHOLDER,
        layout=ipywidgets.Layout(width="100%", height=f"{PROMPT_ROWS * 2}em"),
    )
    resolution_input = ipywidgets.Dropdown(
        options=resolution_labels(buckets),
        value=resolution_labels(buckets)[0],
        description="Resolution",
    )
    seed_input = ipywidgets.IntText(value=RANDOM_SEED_SENTINEL, description="Seed")
    random_seed_note = ipywidgets.HTML(
        f"<i>{RANDOM_SEED_SENTINEL} means {RANDOM_SEED_LABEL.lower()}</i>"
    )
    generate_button = ipywidgets.Button(
        description=GENERATE_BUTTON_LABEL, button_style="primary"
    )
    status_output = ipywidgets.HTML()
    image_output = ipywidgets.Output()

    def on_generate(_button) -> None:
        generate_button.disabled = True
        status_output.value = "Generating..."
        try:
            request = build_request(
                prompt_input.value, resolution_input.value, seed_input.value, buckets
            )
            image = pipeline.generate(request)
            outcome = GenerationOutcome(
                image=image,
                seed=request.seed,
                resolution_label=request.resolution.label,
            )

            with image_output:
                # Clearing with wait keeps the previous image on screen
                # until the new one is ready, which avoids a flicker to
                # blank between generations.
                clear_output(wait=True)
                display(_as_pillow_image(to_display_image(outcome.image)))

            status_output.value = describe_outcome(outcome)
        except Exception:
            logger.error("Generation failed:\n%s", traceback.format_exc())
            status_output.value = "Generation failed; see the log for details."
        finally:
            generate_button.disabled = False

    generate_button.on_click(on_generate)

    return ipywidgets.VBox(
        [
            prompt_input,
            ipywidgets.HBox([resolution_input, seed_input, random_seed_note]),
            generate_button,
            status_output,
            image_output,
        ]
    )


def _as_pillow_image(display_image):
    """Wrap an eight-bit array for display, importing Pillow lazily."""
    from PIL import Image

    return Image.fromarray(display_image)
