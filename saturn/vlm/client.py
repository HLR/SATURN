"""ProbabilisticVLMClient — thin async shim over an OpenAI-compatible vLLM server.

Used by the ``QwenVLvLLM`` agent to score yes/no prompts with vLLM logprobs.

Design notes
------------
* One ``AsyncOpenAI`` client per process, reused across requests.
* A single ``asyncio.Semaphore`` bounds the number of in-flight requests to
  avoid DoS'ing the vLLM server when the sample semaphore is wide open.
* ``score(...)`` returns a plain ``float`` in [0, 1] — the caller wraps it in
  a CPU tensor with ``requires_grad=True`` to preserve the contract of
  ``Agent._score``.
  the semaphore keeps overall load bounded.
"""

from __future__ import annotations

from saturn.settings import env
import asyncio
import base64
import io
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from PIL import Image
from saturn.log import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token bucket constants for yes/no extraction.
# ---------------------------------------------------------------------------

# Yes / No buckets. Tokens are matched after strip() and case-folding, so
# "Yes", " yes" and "YES" all land in the Yes bucket.
_YES_TOKENS = {"Yes"}
_NO_TOKENS = {"No"}

# Log-probability assigned when a bucket is entirely missing from top-k.
_MISSING_LOGPROB = math.log(1e-6)


# ---------------------------------------------------------------------------
# Lazy imports — keep this module import-safe when openai / httpx are absent.
# ---------------------------------------------------------------------------


def _lazy_openai():
    from openai import AsyncOpenAI  # type: ignore

    return AsyncOpenAI


@dataclass
class VLMScoringConfig:
    """Tunables for the scoring path.

    Attributes
    ----------
    max_top_logprobs : int
        Value passed as ``top_logprobs`` to vLLM. A wide window keeps the
        Yes/No buckets in view when the first token isn't a canonical
        "Yes"/"No"; a bucket missing from the window falls back to
        ``_MISSING_LOGPROB``, which pulls P(Yes) toward 0.5.
    yes_tokens, no_tokens : set[str]
        Token-string buckets used to classify the first generated token's
        logprob entries into P(Yes) / P(No).
    """

    max_top_logprobs: int = 20
    yes_tokens: frozenset = frozenset(_YES_TOKENS)
    no_tokens: frozenset = frozenset(_NO_TOKENS)


# ---------------------------------------------------------------------------
# Image encoding helpers.
# ---------------------------------------------------------------------------


def _pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data-URL for the chat-completions API."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def build_chat_messages(images: Sequence[Image.Image], text: str) -> List[dict]:
    """Build an OpenAI-style chat payload with vision content."""
    content: List[dict] = []
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Client.
# ---------------------------------------------------------------------------


class ProbabilisticVLMClient:
    """Async client for a single vLLM OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        max_concurrency: int = 32,
        scoring_config: Optional[VLMScoringConfig] = None,
        request_timeout: float = 300.0,   # long planner replies queue behind scoring calls under load
        seed: Optional[int] = None,
    ) -> None:
        self.base_url = base_url or env("SAPY_VLM_BASE_URL")
        self.model = model or env("SAPY_VLM_MODEL")
        if not self.model:
            raise ValueError(
                "ProbabilisticVLMClient requires a model name via arg or "
                "the SAPY_VLM_MODEL env var."
            )
        self.api_key = api_key
        self.scoring = scoring_config or VLMScoringConfig()
        self.request_timeout = request_timeout
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client: Any | None = None
        # Seed: explicit arg wins; otherwise SAPY_VLM_SEED env var; otherwise None
        # (no seed sent to vLLM).
        if seed is None:
            env_seed = env("SAPY_VLM_SEED")
            if env_seed:
                try:
                    seed = int(env_seed)
                except ValueError:
                    seed = None
        self.seed: Optional[int] = seed

    # -- lifecycle ---------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            AsyncOpenAI = _lazy_openai()
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.request_timeout,
            )
        return self._client

    def _request_extra_body(self) -> Optional[dict[str, Any]]:
        """Disable chat-template "thinking" for every scoring/grounding call.

        The scoring path reads ``logprobs.content[0]`` -- the FIRST generated
        token -- and softmaxes its {Yes,No} buckets. A model that opens with a
        reasoning preamble ("The user is asking...") puts neither Yes nor No in
        that position, so both buckets miss, and P(Yes) collapses to the 0.5
        fallback or to ``_MISSING_LOGPROB``.

        Sent for every model: templates that do not declare the variable
        ignore it, and the few that reject it are handled by
        ``_thinking_kwargs_unsupported``.
        """
        if getattr(self, "_thinking_kwargs_unsupported", False):
            return None
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def _is_template_kwarg_error(self, exc: Exception) -> bool:
        """True if *exc* looks like the server rejecting chat_template_kwargs."""
        msg = str(exc).lower()
        return (
            "chat_template" in msg
            or "enable_thinking" in msg
            or "unexpected keyword" in msg
            or "unknown field" in msg
        )

    async def _create(self, client: Any, request: dict) -> Any:
        """chat.completions.create, retrying once without chat_template_kwargs.

        A backbone whose template rejects the kwarg would otherwise fail every
        call; degrade to thinking-enabled instead, and warn once.
        """
        try:
            return await client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is ours
            eb = request.get("extra_body") or {}
            if not (self._is_template_kwarg_error(exc) and "chat_template_kwargs" in eb):
                raise
            self._thinking_kwargs_unsupported = True
            log.warning(
                f"[vlm_client] {self.model}: server rejected chat_template_kwargs "
                f"({exc}); retrying without it and disabling for this session. "
                f"Scoring may degrade if this model emits a reasoning preamble.",
            )
            eb = {k: v for k, v in eb.items() if k != "chat_template_kwargs"}
            if eb:
                request["extra_body"] = eb
            else:
                request.pop("extra_body", None)
            return await client.chat.completions.create(**request)

    async def aclose(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result

    # -- scoring -----------------------------------------------------------

    async def score(
        self,
        images: Sequence[Image.Image],
        text: str,
    ) -> float:
        """Return P(Yes) for a single yes/no prompt."""
        messages = await asyncio.to_thread(build_chat_messages, images, text)
        return await self._score_messages(messages)

    async def _score_messages(self, messages: List[dict]) -> float:
        async with self._sem:
            client = self._get_client()
            request: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                max_tokens=1,
                temperature=1,
                logprobs=True,
                top_logprobs=self.scoring.max_top_logprobs,
            )
            if self.seed is not None:
                request["seed"] = self.seed
            extra_body = self._request_extra_body()
            if extra_body is not None:
                request["extra_body"] = extra_body
            resp = await self._create(client, request)
        return self._extract_yes_prob(resp)

    def _extract_yes_prob(self, response: Any) -> float:
        """Softmax over {Yes,No} bucket max-logprobs; fallback epsilon for a
        missing bucket."""
        try:
            choice = response.choices[0]
            lp = choice.logprobs.content[0].top_logprobs
        except (AttributeError, IndexError, TypeError):
            return 0.5  # degenerate fallback: be non-committal

        yes_tokens = {t.strip().lower() for t in self.scoring.yes_tokens}
        no_tokens = {t.strip().lower() for t in self.scoring.no_tokens}
        yes_lp: Optional[float] = None
        no_lp: Optional[float] = None
        for entry in lp:
            tok = getattr(entry, "token", None)
            val = getattr(entry, "logprob", None)
            if tok is None or val is None:
                continue
            norm = tok.strip().lower()
            if norm in yes_tokens:
                yes_lp = val if yes_lp is None else max(yes_lp, val)
            elif norm in no_tokens:
                no_lp = val if no_lp is None else max(no_lp, val)

        if yes_lp is None and no_lp is None:
            return 0.5
        if yes_lp is None:
            yes_lp = _MISSING_LOGPROB
        if no_lp is None:
            no_lp = _MISSING_LOGPROB

        m = max(yes_lp, no_lp)
        ey = math.exp(yes_lp - m)
        en = math.exp(no_lp - m)
        return ey / (ey + en)

    # -- generation --------------------------------------------------------

    async def generate(
        self,
        images: Sequence[Image.Image],
        text: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
    ) -> str:
        """Plain chat completion — used for ``_query`` and ``ground``."""
        messages = await asyncio.to_thread(build_chat_messages, images, text)
        async with self._sem:
            client = self._get_client()
            request: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if self.seed is not None:
                request["seed"] = self.seed
            if top_p is not None:
                request["top_p"] = top_p
            if presence_penalty is not None:
                request["presence_penalty"] = presence_penalty

            # Merge thinking-override extra_body with caller-supplied sampling
            # params (e.g. top_k) that vLLM accepts inside extra_body.
            extra_body = self._request_extra_body() or {}
            if top_k is not None:
                extra_body["top_k"] = top_k
            if extra_body:
                request["extra_body"] = extra_body

            resp = await self._create(client, request)
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

