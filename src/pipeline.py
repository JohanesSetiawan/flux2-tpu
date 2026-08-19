"""
End-to-end image generation.

This is the only module that sees all three model components at once,
and the only one that holds state across calls. Everything below it is
either a pure function or a loader.

The design centres on one observation: of the three components, only
the transformer runs more than once per image. The text encoder runs
once per prompt and the decoder once per generation, while the
transformer runs once per sampling step. That asymmetry drives both the
caching and the memory strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import download_bundle, resolve_huggingface_token, restore_component
from .config import (
    ExecutionConfig,
    InferenceConfig,
    MemoryResidencyStrategy,
    ResolutionBucket,
    SamplingConfig,
    TextEncoderConfig,
    TransformerConfig,
    VaeDecoderConfig,
    resolve_residency_strategy,
)
from .execution import (
    build_device_mesh,
    configure_compilation_cache,
    evict_to_host,
    move_to_accelerator,
    plan_component_residency,
    replicate_parameters,
    shard_stacked_blocks,
)
from .models import decode_latent, encode_prompt, predict_velocity
from .sampling import (
    compute_sigma_schedule,
    denoise_latent,
    pack_latent_to_tokens,
    unpack_tokens_to_latent,
)
from .tokenization import load_tokenizer, tokenize_prompts


# The latent grid is coarser than the image by this factor: the
# autoencoder downsamples spatially, and its output is then packed in
# blocks whose contents move into the channel axis.
LATENT_TOKEN_STRIDE = 16

# Channel count of the packed latent the transformer operates on.
PACKED_LATENT_CHANNELS = 128

# The decoder's output covers roughly minus one to one. Mapping it to
# unit range is the last step before an image can be written out.
DECODER_OUTPUT_MINIMUM = -1.0
DECODER_OUTPUT_MAXIMUM = 1.0

# Used only to trigger compilation. Content is irrelevant, since
# compilation depends on shapes rather than values, but it must be
# non-empty to take the same path a real request does.
WARM_UP_PROMPT = "warm up"
WARM_UP_SEED = 0


@dataclass(frozen=True)
class GenerationRequest:
    """
    One image to generate.

    Seed is required rather than defaulted, because an implicit seed
    makes a result impossible to reproduce and reproducibility is what
    every parity check in this project depends on.
    """

    prompt: str
    resolution: ResolutionBucket
    seed: int


class Pipeline:
    """
    A loaded model, ready to generate.

    Holds the three components and a cache of prompt embeddings.
    Construct once per session; calling `generate` repeatedly is cheap
    relative to construction.
    """

    def __init__(
        self,
        config: InferenceConfig,
        logger: logging.Logger,
        text_encoder_config: TextEncoderConfig | None = None,
        transformer_config: TransformerConfig | None = None,
        vae_config: VaeDecoderConfig | None = None,
        sampling_config: SamplingConfig | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._text_encoder_config = text_encoder_config or TextEncoderConfig()
        self._transformer_config = transformer_config or TransformerConfig()
        self._vae_config = vae_config or VaeDecoderConfig()
        self._sampling_config = sampling_config or SamplingConfig()
        self._execution_config = execution_config or ExecutionConfig()

        self._residency = resolve_residency_strategy(
            config.residency_strategy, jax.device_count()
        )
        logger.info(
            "Memory residency strategy resolved to %s across %d device(s)",
            self._residency.value,
            jax.device_count(),
        )

        self._residency_plan = plan_component_residency(
            self._residency, ("text_encoder", "transformer", "vae")
        )
        for entry in self._residency_plan:
            logger.info(
                "  %s: %s",
                entry.component_name,
                "resident" if entry.resident else "held on host, moved in when used",
            )

        self._bundle_path: Path | None = None
        self._text_encoder_parameters: dict | None = None
        self._transformer_parameters: dict | None = None
        self._vae_parameters: dict | None = None
        self._tokenizer = None

        # Conditioning depends only on the prompt text, never on the
        # seed or the resolution, so a repeated prompt can skip the text
        # encoder entirely. Under the swapped residency strategy that
        # also avoids a host transfer, which is the dominant cost of a
        # repeat generation.
        self._conditioning_cache: dict[str, jnp.ndarray] = {}

    @property
    def resolution_buckets(self) -> tuple[ResolutionBucket, ...]:
        """The resolutions this pipeline will accept, in offer order."""
        return self._config.resolution_buckets

    def load(self) -> None:
        """
        Download the checkpoint and restore every component.

        Separate from construction so that the caller controls when the
        multi-gigabyte work happens, which matters in a notebook where
        construction and use sit in different cells.
        """
        source = self._config.checkpoint_source
        token = resolve_huggingface_token(
            self._logger, source.huggingface_token_environment_variable
        )
        self._bundle_path = download_bundle(source, self._logger, token)

        self._tokenizer = load_tokenizer(self._bundle_path, self._logger)
        self._text_encoder_parameters = restore_component(
            self._bundle_path, "text_encoder", self._logger
        )
        self._transformer_parameters = restore_component(
            self._bundle_path, "transformer", self._logger
        )
        self._vae_parameters = restore_component(self._bundle_path, "vae", self._logger)

        configure_compilation_cache(self._execution_config, self._logger)

        mesh = build_device_mesh(self._logger)
        # Block stacks are the only parameters large enough to be worth
        # splitting; everything else is replicated, which needs no
        # collectives and costs little at these sizes.
        self._transformer_parameters = {
            group_name: (
                shard_stacked_blocks(group, mesh, self._logger)
                if group_name in ("double_blocks", "single_blocks")
                else replicate_parameters(group, mesh)
            )
            for group_name, group in self._transformer_parameters.items()
        }

        if not self._text_encoder_is_resident():
            self._text_encoder_parameters = evict_to_host(
                self._text_encoder_parameters, self._logger, "text_encoder"
            )

        self._logger.info("All components loaded")

    def _text_encoder_is_resident(self) -> bool:
        return next(
            entry.resident
            for entry in self._residency_plan
            if entry.component_name == "text_encoder"
        )

    def _require_loaded(self) -> None:
        if self._transformer_parameters is None:
            raise RuntimeError("Pipeline.load must be called before generating")

    def encode(self, prompt: str) -> jnp.ndarray:
        """
        Encode a prompt, reusing a cached result when the same text has
        been seen before.
        """
        self._require_loaded()

        if prompt in self._conditioning_cache:
            self._logger.info("Reusing cached conditioning for this prompt")
            return self._conditioning_cache[prompt]

        self._logger.info("Encoding prompt")
        tokenized = tokenize_prompts(
            self._tokenizer,
            [prompt],
            self._text_encoder_config.sequence_length,
            self._logger,
        )

        # Under the swapped strategy the encoder lives on the host
        # between prompts. It is moved in, used, and released here
        # rather than being held, which is the whole point of evicting
        # it: the conditioning it produces is kept instead, and that is
        # a few megabytes against several gigabytes.
        parameters = self._text_encoder_parameters
        if not self._text_encoder_is_resident():
            parameters = move_to_accelerator(parameters, self._logger, "text_encoder")

        conditioning = encode_prompt(
            jnp.asarray(tokenized.token_ids),
            jnp.asarray(tokenized.token_is_real),
            parameters,
            self._text_encoder_config,
            self._execution_config,
        )
        conditioning = jax.block_until_ready(conditioning)

        if not self._text_encoder_is_resident():
            # Dropping the reference is what actually frees the
            # accelerator copy; without this the eviction saves nothing.
            del parameters

        self._conditioning_cache[prompt] = conditioning
        return conditioning

    def _initial_noise(
        self, resolution: ResolutionBucket, seed: int
    ) -> tuple[jnp.ndarray, int, int]:
        """
        Draw the starting noise for one generation.

        The latent grid is derived from the requested resolution rather
        than passed alongside it, so the two cannot disagree.
        """
        latent_height = resolution.height // LATENT_TOKEN_STRIDE
        latent_width = resolution.width // LATENT_TOKEN_STRIDE

        key = jax.random.key(seed)
        noise = jax.random.normal(
            key, (1, latent_height, latent_width, PACKED_LATENT_CHANNELS)
        )
        return noise, latent_height, latent_width

    def warm_up(
        self, buckets: tuple[ResolutionBucket, ...] | None = None
    ) -> None:
        """
        Compile the generation program for each supported resolution.

        Compilation is per shape, not per prompt, so one generation at a
        given resolution pays for every later generation at that
        resolution. Doing it up front rather than on the first real
        request is the difference between a person waiting once at
        startup and waiting the first time they ask for something.

        A short throwaway prompt is used and its result discarded. The
        prompt content is irrelevant to what gets compiled, since only
        shapes matter, but it must be non-empty to exercise the same
        path a real request takes.

        Note that the conditioning cache is left populated with the
        warm-up prompt afterwards. That is harmless: it is a few
        megabytes, and a real prompt will simply miss the cache and
        encode normally.
        """
        self._require_loaded()
        buckets = buckets or self._config.resolution_buckets

        for bucket in buckets:
            self._logger.info("Warming up %s", bucket.label)
            request = GenerationRequest(
                prompt=WARM_UP_PROMPT, resolution=bucket, seed=WARM_UP_SEED
            )
            # generate returns a materialised numpy array, so the
            # computation is already forced by the time it returns. The
            # result is discarded: only the compiled program is wanted.
            self.generate(request)

        self._logger.info("Warm-up complete for %d resolution(s)", len(buckets))

    def generate(self, request: GenerationRequest) -> np.ndarray:
        """
        Generate one image.

        Returns
        -------
        Array of shape (height, width, 3) with values in unit range,
        ready to be written as an image.
        """
        self._require_loaded()
        resolution = request.resolution

        self._logger.info(
            "Generating %s, seed %d", resolution.label, request.seed
        )

        conditioning = self.encode(request.prompt)

        noise, latent_height, latent_width = self._initial_noise(resolution, request.seed)
        latent_tokens = pack_latent_to_tokens(noise)

        sigma_schedule = compute_sigma_schedule(
            resolution.image_tokens, self._sampling_config
        )
        self._logger.info(
            "Schedule over %d image tokens: %s",
            resolution.image_tokens,
            np.array2string(sigma_schedule, precision=4),
        )

        def velocity_at(tokens: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
            return predict_velocity(
                tokens,
                conditioning,
                timesteps,
                latent_height,
                latent_width,
                self._transformer_parameters,
                self._transformer_config,
                self._execution_config,
            )

        denoised_tokens = denoise_latent(
            latent_tokens, sigma_schedule, velocity_at, self._logger, self._execution_config
        )

        latent = unpack_tokens_to_latent(denoised_tokens, latent_height, latent_width)

        self._logger.info("Decoding latent to image")
        decoded = decode_latent(latent, self._vae_parameters, self._vae_config)

        return to_display_range(np.asarray(decoded[0]))


def to_display_range(decoded: np.ndarray) -> np.ndarray:
    """
    Map decoder output into unit range and clip.

    The decoder's output is not bounded, so values outside the expected
    range do occur and would wrap around if cast without clipping.
    Clipping is applied rather than rescaling by the observed extremes,
    because rescaling would make an image's brightness depend on its own
    most extreme pixel.
    """
    span = DECODER_OUTPUT_MAXIMUM - DECODER_OUTPUT_MINIMUM
    normalized = (decoded - DECODER_OUTPUT_MINIMUM) / span
    return np.clip(normalized, 0.0, 1.0)
