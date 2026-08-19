"""
A browser interface built on Gradio.

Like the widget panel, this holds no generation logic: inputs are
turned into a request by `session` and handed to the pipeline. The two
front ends deliberately share that layer rather than each implementing
their own input handling, so a fix to one is a fix to both.

Gradio replaces the displayed image on each run by default, which is
the behaviour wanted here: one image at a time, replaced rather than
accumulated.
"""

from __future__ import annotations

import logging
import traceback

import numpy as np

from ..pipeline import Pipeline
from .session import (
    GenerationOutcome,
    RANDOM_SEED_SENTINEL,
    build_request,
    describe_outcome,
    resolution_labels,
    to_display_image,
)


INTERFACE_TITLE = "FLUX.2 Klein-4B on TPU"
PROMPT_PLACEHOLDER = "Describe the image to generate"
GENERATE_BUTTON_LABEL = "Generate"

# Bound to the local machine by default. A hosted notebook needs a
# shared link to be reachable, which the caller opts into rather than
# getting by default, since it exposes the interface publicly.
DEFAULT_SERVER_NAME = "0.0.0.0"


def build_interface(pipeline: Pipeline, logger: logging.Logger):
    """
    Build and return the Gradio interface without launching it.

    Returning rather than launching lets the caller decide how it is
    served, which matters because the sharing option has privacy
    consequences that belong to the person running it rather than to
    this function.
    """
    try:
        import gradio
    except ImportError as error:
        raise ImportError(
            "Gradio is required for the browser interface. Install it with: "
            "pip install gradio"
        ) from error

    buckets = pipeline.resolution_buckets
    labels = resolution_labels(buckets)

    def run(prompt: str, resolution_label: str, seed: int):
        try:
            request = build_request(prompt, resolution_label, int(seed), buckets)
            image = pipeline.generate(request)
            outcome = GenerationOutcome(
                image=image,
                seed=request.seed,
                resolution_label=request.resolution.label,
            )
            return to_display_image(outcome.image), describe_outcome(outcome)
        except Exception:
            logger.error("Generation failed:\n%s", traceback.format_exc())
            # An empty image rather than None, so the output component
            # clears predictably instead of keeping a stale result
            # beside a failure message.
            return np.zeros((1, 1, 3), dtype=np.uint8), (
                "Generation failed; see the log for details."
            )

    with gradio.Blocks(title=INTERFACE_TITLE) as interface:
        gradio.Markdown(f"## {INTERFACE_TITLE}")

        with gradio.Row():
            with gradio.Column():
                prompt_input = gradio.Textbox(
                    label="Prompt", placeholder=PROMPT_PLACEHOLDER, lines=3
                )
                resolution_input = gradio.Dropdown(
                    choices=labels, value=labels[0], label="Resolution"
                )
                seed_input = gradio.Number(
                    value=RANDOM_SEED_SENTINEL,
                    label=f"Seed ({RANDOM_SEED_SENTINEL} for a random one)",
                    precision=0,
                )
                generate_button = gradio.Button(GENERATE_BUTTON_LABEL, variant="primary")

            with gradio.Column():
                image_output = gradio.Image(label="Result", type="numpy")
                status_output = gradio.Textbox(label="Status", interactive=False)

        generate_button.click(
            fn=run,
            inputs=[prompt_input, resolution_input, seed_input],
            outputs=[image_output, status_output],
        )

    return interface
