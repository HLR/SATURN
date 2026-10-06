"""CodeGenerator: prompts the code LLM for a predicate program and caches it.

Model-agnostic HTTP client side (DeepSeek/OpenRouter); imported by saturn.pipeline only.

Program cache: a program is stored under a content address of everything that
determines it (see ``CodeGenerator._program_cache_key``): the prompt template,
provider, model, decoding parameters including the seed, and the fully rendered
prompt. The value is the raw extracted snippet (before ``fix_framework_code`` and
before the execution template wraps it).
"""
from saturn.settings import env
import hashlib
import os
import json
import re
from openai import OpenAI
import fcntl
import time


import ast
from typing import Optional
from saturn.log import get_logger

log = get_logger(__name__)

PROGRAM_KEY_PREFIX = "prog-v2:"


# One code-model request may take this long; the client does not retry on its own (max_retries=0):
# CodeGenerator._create retries, and gives up once a program has waited MAX_REQUEST_WALL_S.
REQUEST_TIMEOUT_S = 180.0
MAX_REQUEST_WALL_S = 900.0


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def codegen_seed() -> Optional[int]:
    """Seed sent with every code-LLM request.

    Unset/empty SAPY_CODEGEN_SEED means seed 0: decoding is deterministic by
    default (temperature 0 + fixed seed), and a different seed is an explicit
    choice. A non-integer value is a typo; it is logged and no seed is sent
    rather than aborting a multi-hour run.
    """
    raw = env("SAPY_CODEGEN_SEED")
    if raw in (None, ""):
        return 0
    try:
        return int(raw)
    except ValueError:
        log.warning(f"[CodeGenerator] ignoring non-integer SAPY_CODEGEN_SEED={raw!r}")
        return None


class ScoreBBoxFixer(ast.NodeTransformer):
    """
    Node transformer that:
      1. For score(...) calls whose question mentions both "red bounding box"
         and "green bounding box":
         - Ensures num_objects == 2.
         - If "green bounding box" appears before "red bounding box" in the text,
           swaps the color phrases so the logical mapping is:
             red -> first object, green -> second object.
      2. For score(...) calls whose question mentions ONLY "green bounding box"
         and is effectively a one object predicate (num_objects == 1 or missing):
         - Replaces "green bounding box" with "red bounding box".
      3. For cases where both red and green are mentioned but num_objects is not 2:
         - Forces num_objects = 2.
    """

    RED_PHRASE = "red bounding box"
    GREEN_PHRASE = "green bounding box"

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # First, visit children so other transforms propagate
        self.generic_visit(node)

        # We only care about bare score(...) calls
        func = node.func
        if isinstance(func, ast.Name) and func.id in ["score", "query"]:
            node = self._fix_score_call(node)

        return node

    def _get_question_str(self, node: ast.Call) -> Optional[ast.Constant]:
        """Return the question string Constant node if the first arg is a string."""
        if not node.args:
            return None
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first
        return None

    def _get_num_objects_info(self, node: ast.Call):
        """
        Return (index_in_args, keyword_index, value) for num_objects.
        Only one of index_in_args or keyword_index will be not None.
        """
        arg_index = None
        kw_index = None
        value = None

        # Keyword form: score("...", num_objects=1, ...)
        for i, kw in enumerate(node.keywords):
            if kw.arg == "num_objects":
                kw_index = i
                if isinstance(kw.value, ast.Constant) and isinstance(
                    kw.value.value, int
                ):
                    value = kw.value.value
                return arg_index, kw_index, value

        # Positional form: score("...", 1, "class")
        if len(node.args) >= 2:
            second = node.args[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, int):
                arg_index = 1
                value = second.value

        return arg_index, kw_index, value

    def _set_num_objects(self, node: ast.Call, new_val: int):
        """
        Set num_objects to new_val in either positional or keyword form,
        or add it as a keyword if missing.
        """
        arg_index, kw_index, value = self._get_num_objects_info(node)

        if kw_index is not None:
            node.keywords[kw_index].value = ast.Constant(value=new_val)
        elif arg_index is not None:
            node.args[arg_index] = ast.Constant(value=new_val)
        else:
            # Append as keyword if not present
            node.keywords.append(
                ast.keyword(arg="num_objects", value=ast.Constant(value=new_val))
            )

    def _swap_red_green_in_text_order_sensitive(self, text: str) -> str:
        """
        If green appears before red in the string, swap the phrases
        "red bounding box" and "green bounding box".
        This effectively corrects which color is tied to which position.
        """
        if (
            self.RED_PHRASE in text
            and self.GREEN_PHRASE in text
            and text.index(self.GREEN_PHRASE) < text.index(self.RED_PHRASE)
        ):
            # Swap the phrases using a temporary marker
            tmp = "__TMP_RED_BOUNDING_BOX__"
            text = text.replace(self.RED_PHRASE, tmp)
            text = text.replace(self.GREEN_PHRASE, self.RED_PHRASE)
            text = text.replace(tmp, self.GREEN_PHRASE)
        return text

    def _fix_score_call(self, node: ast.Call) -> ast.Call:
        q_node = self._get_question_str(node)
        if q_node is None:
            return node

        q_text = q_node.value
        has_red = self.RED_PHRASE in q_text
        has_green = self.GREEN_PHRASE in q_text

        arg_index, kw_index, num_val = self._get_num_objects_info(node)

        # Case 1: both red and green mentioned
        if has_red and has_green:
            # Fix color assignment if green appears first
            q_text_fixed = self._swap_red_green_in_text_order_sensitive(q_text)

            # Ensure num_objects = 2
            self._set_num_objects(node, 2)

            # Update the question literal
            q_node.value = q_text_fixed
            return node

        # Case 2: only green mentioned, treat as a misplaced single object prompt
        if has_green and not has_red:
            # If num_objects is explicitly 1 or missing, we assume a mistake
            if num_val == 1 or num_val is None:
                q_text_fixed = q_text.replace(self.GREEN_PHRASE, self.RED_PHRASE)
                q_node.value = q_text_fixed
                # Make sure num_objects = 1 if not set
                self._set_num_objects(node, 1)
                return node

        return node


def fix_framework_code(source: str) -> tuple[str, bool]:
    """
    Return (new_source, changed).

    `changed` is True if the AST after running ScoreBBoxFixer
    is different from the original AST.
    """
    try:
        tree = ast.parse(source)
        old_dump = ast.dump(tree, include_attributes=False)

        fixer = ScoreBBoxFixer()
        new_tree = fixer.visit(tree)
        ast.fix_missing_locations(new_tree)

        new_dump = ast.dump(new_tree, include_attributes=False)
        changed = new_dump != old_dump

        new_source = ast.unparse(new_tree)
        return new_source, changed
    except Exception as e:
        log.error(f"Error processing source code: {e}")
        return source, False


def extract_code(text_response):
    code_matches_html = re.findall(r"<code>(.*?)</code>", text_response, re.DOTALL)
    code_matches_md = re.findall(r"```python(.*?)```", text_response, re.DOTALL)
    if code_matches_html:
        return code_matches_html[-1].strip()
    if code_matches_md:
        return code_matches_md[-1].strip()
    # No fence: accept the text only if it is a syntactically valid program.
    # A truncated or prose-only completion must not become "the program".
    import ast as _ast
    try:
        _ast.parse(text_response)
    except SyntaxError:
        return None
    return text_response.strip()


def extract_objects(text_response):
    objects_matches_html = re.findall(
        r"<objects>(.*?)</objects>", text_response, re.DOTALL
    )
    if objects_matches_html:
        return objects_matches_html[-1].strip()
    return None



class CodeGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        program_cache_path: str,
        code_prompt_path: str,
        write_program_cache: bool = False,
        use_cache: bool = True,
        provider: str = "deepseek",
    ):
        self.api_key = api_key
        self.write_program_cache = write_program_cache
        self.model_name = model_name
        self.cache_file = program_cache_path
        self.objects_file = program_cache_path.replace(".json", "_objects.json")
        self.code_prompt_path = code_prompt_path
        self.provider = provider.lower()
        if self.provider == "openrouter":
            # OpenRouter, pinned to one upstream and quantization
            # (SAPY_OPENROUTER_PROVIDER / _QUANT) so the served weights cannot
            # drift; the pin is sent by _request_kwargs.
            # OpenRouter needs its own key: always read OPENROUTER_API_KEY here, ignoring api_key.
            or_key = env("OPENROUTER_API_KEY")
            if not or_key:
                raise RuntimeError("provider=openrouter but OPENROUTER_API_KEY is empty")
            self.client = OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1", timeout=REQUEST_TIMEOUT_S, max_retries=0)
            self._or_provider = env("SAPY_OPENROUTER_PROVIDER")
            self._or_quant = env("SAPY_OPENROUTER_QUANT")
            pin = (f"pinned {self._or_provider}/{self._or_quant}, fallbacks off"
                   if self._or_provider else "no provider pin")
            log.info(f"[CodeGenerator] provider=openrouter model={model_name} {pin}")
        elif self.provider == "deepseek":
            # the default: the OpenAI client pointed at DeepSeek's endpoint (deepseek-chat)
            if not api_key:   # the runner passes "": the key comes from the environment
                api_key = env("DEEPSEEK_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=REQUEST_TIMEOUT_S, max_retries=0)
        else:
            raise ValueError(f"Unknown code-generation provider '{provider}' (use 'deepseek' or 'openrouter')")
        self.use_cache = use_cache
        if use_cache:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    self.program_cache = json.load(f)
            else:
                self.program_cache = {}
            if os.path.exists(self.objects_file):
                with open(self.objects_file, "r") as f:
                    self.objects_cache = json.load(f)
            else:
                self.objects_cache = {}
        else:
            self.program_cache = {}
            self.objects_cache = {}

    def _log_provider_failure(self, err) -> None:
        """Provenance for FAILED calls too: an outage must not look like 'no program'."""
        try:
            import json as _json
            import time as _time
            with open(env("SAPY_PROVIDER_LOG"), "a") as _fh:
                _fh.write(_json.dumps({"ts": _time.time(), "error": f"{type(err).__name__}: {err}"[:300]}) + "\n")
        except Exception:
            log.debug("suppressed: provider failure log write failed", exc_info=True)

    def _is_reasoning_model(self) -> bool:
        return str(self.model_name).lower() == "deepseek-reasoner"

    def _is_deepseek_v4(self) -> bool:
        # also matches OpenRouter's namespaced id "deepseek/deepseek-v4-flash"
        return "deepseek-v4-" in str(self.model_name).lower()

    def _request_kwargs(self, messages, response_format=None) -> dict:
        """Everything sent to the chat API for *messages* (also hashed into the cache key)."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        if self._is_reasoning_model():
            kwargs["max_tokens"] = 16384
        else:
            kwargs.update(
                {
                    "max_tokens": 8192,
                    "temperature": 0.0,
                    "n": 1,
                    "top_p": 1.0,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                }
            )
            if self._is_deepseek_v4():
                # DeepSeek-V4 runs in non-thinking mode; set it explicitly
                # rather than trusting the API default.
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        # Greedy decoding alone does not make a hosted MoE reproducible (batch
        # composition shifts logits), so a fixed seed is always sent.
        seed = codegen_seed()
        if seed is not None:
            kwargs["seed"] = seed
        if self.provider == "openrouter":
            # OpenRouter ignores DeepSeek's `thinking` field; its switch is
            # `reasoning.enabled`.
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            # Pin the upstream so the served weights/quantization cannot drift.
            # SAPY_OPENROUTER_PROVIDER="" = unpinned (non-DeepSeek code models).
            if self._or_provider:
                kwargs["extra_body"]["provider"] = {
                    "order": [self._or_provider],
                    "quantizations": [self._or_quant],
                    "allow_fallbacks": False,
                }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    # Seconds to wait before each new attempt after a rate limit, a 5xx, a timeout or a dropped connection.
    # An upstream limit (OpenRouter: "temporarily rate-limited upstream") lasts longer than the
    # client's own two quick retries; without this, a busy run silently loses programs and repairs.
    RETRY_WAITS = (5, 10, 20, 40, 60, 60, 60)

    def _create(self, kwargs):
        """chat.completions.create, waiting out rate limits and transient server errors."""
        import random
        import time
        import openai
        start = time.monotonic()
        for wait in self.RETRY_WAITS + (None,):
            try:
                return self.client.chat.completions.create(**kwargs)
            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
                if wait is None or time.monotonic() - start + wait > MAX_REQUEST_WALL_S:
                    raise
                log.warning(f"[CodeGenerator] {type(e).__name__}; retrying in ~{wait}s")
                time.sleep(wait * (0.75 + 0.5 * random.random()))

    def _chat_completion(self, messages, response_format=None):
        kwargs = self._request_kwargs(messages, response_format)
        response = self._create(kwargs)
        # one JSON line per call: which upstream/model actually served it
        try:
            import json as _json
            import time as _time
            _u = getattr(response, "usage", None)
            with open(env("SAPY_PROVIDER_LOG"), "a") as _fh:
                _fh.write(_json.dumps({"ts": _time.time(), "id": getattr(response, "id", None),
                    "provider": getattr(response, "provider", None) or self.provider,
                    "model": getattr(response, "model", None),
                    "prompt_tokens": getattr(_u, "prompt_tokens", None),
                    "completion_tokens": getattr(_u, "completion_tokens", None)}) + "\n")
        except Exception as _e:
            log.warning(f"[CodeGenerator] provider log failed: {_e}")
        return response

    @staticmethod
    def _response_content(response):
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            preview = reasoning_content[:1200]
            suffix = "..." if len(reasoning_content) > 1200 else ""
            log.debug(f"[CODEGEN REASONING]\n{preview}{suffix}")
        return message.content

    def _program_cache_key(self, prompt_format: str, request: dict) -> str:
        """Content address of one codegen request.

        sha256 over the prompt template text (by its sha), the provider, and the
        full request: model, decoding parameters incl. seed, and the fully
        rendered messages. Anything that can change the program changes the key,
        so a cached program is only ever replayed for the request that made it.
        """
        payload = {
            "template_sha256": _sha256(prompt_format),
            "provider": self.provider,
            "request": request,
        }
        return PROGRAM_KEY_PREFIX + _sha256(json.dumps(payload, sort_keys=True, default=str))

    def _reload_caches(self) -> None:
        """Pick up programs written by other processes since the last call."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.program_cache = json.load(f)
        if os.path.exists(self.objects_file):
            with open(self.objects_file, "r") as f:
                self.objects_cache = json.load(f)

    @staticmethod
    def _finalize(code: Optional[str]) -> Optional[str]:
        """The one post-processing step applied to every program, fresh or cached."""
        if not code:
            return code
        return fix_framework_code(code)[0]

    def _render_prompt(
        self,
        prompt_format: str,
        query: str,
        object_groundings_block: str = "",
        scene_facts_block: str = "",
        clarified_query_block: str = "",
    ) -> str:
        prompt = prompt_format.replace("{query}", query)
        prompt = prompt.replace("{object_groundings_block}", object_groundings_block or "")
        prompt = prompt.replace("{scene_facts_block}", scene_facts_block or "")
        prompt = prompt.replace("{clarified_query_block}", clarified_query_block or "")
        return prompt

    def generate_code(
        self,
        query,
        reasoning=None,
        prompt_format=None,
        force_generate=False,
        object_groundings_block: str = "",
        clarified_query_block: str = "",
        scene_facts_block: str = "",
    ):
        """Generate a program for *query* with the code LLM."""
        if self.use_cache:
            self._reload_caches()

        if prompt_format is None:
            with open(self.code_prompt_path, "r", encoding="utf-8") as f:
                prompt_format = f.read()
        prompt = self._render_prompt(
            prompt_format, query, object_groundings_block,
            scene_facts_block, clarified_query_block,
        )
        if reasoning:
            prompt += f"\nUse this reasoning chain: {json.dumps(reasoning)}"

        messages = [
            {"role": "system", "content": "You are an expert Python programmer."},
            {"role": "user", "content": prompt},
        ]
        request = self._request_kwargs(messages)
        cache_key = self._program_cache_key(prompt_format, request)

        if not force_generate:
            if self.program_cache.get(cache_key) and cache_key in self.objects_cache:
                return self._finalize(self.program_cache[cache_key]), self.objects_cache[cache_key]

        try:
            response = self._chat_completion(messages)
            text_response = self._response_content(response)
        except Exception as e:
            log.error(f"[CODEGEN] Error for query: {query}\nError: {e}")
            self._log_provider_failure(e)
            return None, None

        log.debug(f"[CODEGEN]\n{text_response}")
        if env("SAPY_THINKING_LOG"):
            # The program cache keeps only the code; keep the coder's stated reasoning next to it.
            m = re.search(r"<thinking>(.*?)</thinking>", text_response, re.DOTALL)
            try:
                with open(env("SAPY_THINKING_LOG"), "a") as fh:
                    fh.write(json.dumps({"query": query, "cache_key": cache_key,
                                         "thinking": m.group(1).strip() if m else None}) + "\n")
            except OSError as e:
                log.warning(f"[CodeGenerator] thinking log failed: {e}")
        extracted_code = extract_code(text_response)
        objects = extract_objects(text_response)

        # Store the raw snippet; an unparseable response is not a program and
        # is not cached (a replay would otherwise return the failure forever).
        objects = objects or ""
        if extracted_code:
            self.program_cache[cache_key] = extracted_code
            self.objects_cache[cache_key] = objects
            if self.write_program_cache:
                # Another process may have stored this key first; return what is on disk.
                extracted_code, objects = self._persist_caches_atomic(
                    cache_key, extracted_code, objects
                )

        return self._finalize(extracted_code), objects

    def retry_generate_code(
        self,
        query: str,
        failed_code: str,
        error_message: str,
        object_groundings_block: str = "",
        clarified_query_block: str = "",
        scene_facts_block: str = "",
        prompt_format: Optional[str] = None,
    ):
        """Re-generate code after a runtime error.

        Replays the original prompt, the failed program as the assistant turn,
        then the error, and asks for a corrected program in the same output
        format. No fix-up hints are added: advice such as "handle missing
        objects" makes the model add defensive guards and drop the required
        output blocks. ``prompt_format`` must match what the first attempt used, so the
        replayed prompt is the one that produced ``failed_code``.

        Returns:
            (extracted_code, objects) or (None, None) on failure.
        """
        if prompt_format is None:
            with open(self.code_prompt_path, "r", encoding="utf-8") as f:
                prompt_format = f.read()
        prompt = self._render_prompt(
            prompt_format, query, object_groundings_block,
            scene_facts_block, clarified_query_block,
        )
        error_context = (
            "Running this program raised:\n"
            f"```\n{error_message}\n```\n"
            "Return the corrected program, in the same output format the task above asks for."
        )
        messages = [
            {"role": "system", "content": "You are an expert Python programmer."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"```python\n{failed_code}\n```"},
            {"role": "user", "content": error_context},
        ]

        try:
            response = self._chat_completion(messages)
            text_response = self._response_content(response)
            log.debug(f"[RETRY] LLM response:\n{text_response}")
            # Retries are never cached: only first-attempt programs are.
            return self._finalize(extract_code(text_response)), extract_objects(text_response)
        except Exception as e:
            log.error(f"[RETRY] Error during retry code generation: {e}")
            self._log_provider_failure(e)
            return None, None


    # --------------------
    # Persistence helpers
    # --------------------
    def _persist_caches_atomic(self, query_key: str, code=None, objects=None):
        """Atomically merge-update and persist both caches under an exclusive file lock.
        Ensures concurrent writers don't clobber each other's updates and reads see complete files.

        ``code`` / ``objects`` are the entry to store under ``query_key``
        (default: the in-memory entries). Pass them explicitly: another thread
        sharing this generator may rebind ``self.program_cache`` (reload or its
        own persist) between the store and this call, dropping the key.
        Returns the (code, objects) now stored for ``query_key``.
        """
        if code is None:
            code = self.program_cache.get(query_key)
        if objects is None:
            objects = self.objects_cache.get(query_key)
        lock_path = f"{self.cache_file}.lock"
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.objects_file) or ".", exist_ok=True)

        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # Re-read latest on-disk caches while holding the lock to merge safely
                disk_prog = {}
                disk_objs = {}
                if os.path.exists(self.cache_file):
                    try:
                        with open(self.cache_file, "r") as f:
                            disk_prog = json.load(f)
                    except Exception:
                        # If file is corrupt, keep going with empty dict
                        disk_prog = {}
                if os.path.exists(self.objects_file):
                    try:
                        with open(self.objects_file, "r") as f:
                            disk_objs = json.load(f)
                    except Exception:
                        disk_objs = {}

                # If another process already populated this query, prefer on-disk values
                if query_key in disk_prog and query_key in disk_objs:
                    code = disk_prog[query_key]
                    objects = disk_objs.get(query_key, "")
                else:
                    # Merge our in-memory caches into on-disk dicts
                    disk_prog.update(self.program_cache)
                    disk_objs.update(self.objects_cache)
                    if code is not None:
                        disk_prog[query_key] = code
                    if objects is not None:
                        disk_objs[query_key] = objects

                # Write both files atomically
                self._atomic_write_json(self.cache_file, disk_prog)
                self._atomic_write_json(self.objects_file, disk_objs)

                # Update in-memory copies to reflect disk
                self.program_cache = disk_prog
                self.objects_cache = disk_objs
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        return code, objects

    @staticmethod
    def _atomic_write_json(path: str, data: dict):
        """Write JSON to a temp file and atomically replace the target file."""
        dir_name = os.path.dirname(path) or "."
        base = os.path.basename(path)
        tmp_name = f".{base}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
        tmp_path = os.path.join(dir_name, tmp_name)
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
