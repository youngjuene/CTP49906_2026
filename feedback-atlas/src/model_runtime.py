"""One executor owns the already-loaded model and blocking projection work."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import threading

import numpy as np

from src.textnorm import normalize_text


class ModelRuntime:
    def __init__(self, embedder, *, content_token_limit=1792, total_token_limit=2048,
                 batch_size=32):
        if not 1 <= content_token_limit <= 1792 or not 1 <= total_token_limit <= 2048:
            raise ValueError("invalid model token limits")
        if not 1 <= batch_size <= 32:
            raise ValueError("invalid inference batch size")
        self.embedder = embedder
        self.spec = embedder.spec
        self.dim = embedder.dim
        self.model_id = embedder.model_id
        self.content_token_limit = content_token_limit
        self.total_token_limit = min(total_token_limit, getattr(embedder, "max_seq_length", total_token_limit))
        self.batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atlas-model")
        self._worker_id = None
        self._closed = False
        self.unit_cache_key = self._identity("clustering")
        self.similarity_key = self._identity("similarity")

    def _identity(self, purpose):
        identity = getattr(self.embedder, "task_identity", lambda task: f"{self.model_id}:{task}")(purpose)
        policy = f"{identity}:{self.dim}:pool-token-weighted-v1:nfc-ws-v1:{self.content_token_limit}:{self.total_token_limit}"
        return f"{self.model_id}:{purpose}:pool-v1:{hashlib.sha256(policy.encode()).hexdigest()[:24]}"

    async def run(self, fn, *args):
        if self._closed:
            raise RuntimeError("model runtime closed")
        return await asyncio.get_running_loop().run_in_executor(self._executor, partial(self._invoke, fn, args))

    def run_sync(self, fn, *args):
        """Blocking startup entry; nested worker calls reuse the owning thread."""
        if self._closed:
            raise RuntimeError("model runtime closed")
        if threading.get_ident() == self._worker_id:
            return fn(*args)
        return self._executor.submit(self._invoke, fn, args).result()

    def _invoke(self, fn, args):
        self._worker_id = threading.get_ident()
        return fn(*args)

    def _check_worker(self):
        if threading.get_ident() != self._worker_id:
            raise RuntimeError("model encoding requires the shared runtime worker")

    def close(self):
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _counts(self, text, purpose):
        count = getattr(self.embedder, "token_counts", None)
        if count is None:
            from src.embedder import HashEmbedder
            if not isinstance(self.embedder, HashEmbedder):
                raise TypeError("real embedder must expose token_counts and encode_task")
            return len(text), len(text)  # explicit fake, not a real token claim
        return count(text, purpose)

    def text_windows(self, text, purpose="clustering"):
        """Partition text exactly and check actual prompt/special-token overhead.

        Tokenization need not be monotonic: each accepted prefix is recounted.
        Codepoint slicing avoids text changes from token-id decoding.
        """
        self._check_worker()
        if not text:
            return [(text, 1)]
        windows = []
        start = 0
        while start < len(text):
            suffix = text[start:]
            content, prepared = self._counts(suffix, purpose)
            if content <= self.content_token_limit and prepared <= self.total_token_limit:
                windows.append((suffix, max(content, 1)))
                break
            low, high, accepted = 1, len(suffix), 0
            while low <= high:
                middle = (low + high) // 2
                content, prepared = self._counts(suffix[:middle], purpose)
                if content <= self.content_token_limit and prepared <= self.total_token_limit:
                    accepted = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if not accepted:
                raise ValueError("model prompt leaves no room for one code point")
            window = suffix[:accepted]
            content, prepared = self._counts(window, purpose)
            if content > self.content_token_limit or prepared > self.total_token_limit:
                raise ValueError("prepared input exceeds model token limit")
            windows.append((window, max(content, 1)))
            start += accepted
        return windows

    def _encode(self, texts, purpose):
        self._check_worker()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        groups = [self.text_windows(normalize_text(text), purpose) for text in texts]
        flat = [window for group in groups for window, _ in group]
        encoder = getattr(self.embedder, "encode_task", None)
        batches = []
        for start in range(0, len(flat), self.batch_size):
            batch = flat[start:start + self.batch_size]
            for text in batch:
                content, prepared = self._counts(text, purpose)
                if content > self.content_token_limit or prepared > self.total_token_limit:
                    raise ValueError("prepared input exceeds model token limit")
            vectors = encoder(batch, purpose) if encoder else self.embedder.encode(batch)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(batch), self.dim) or not np.isfinite(vectors).all():
                raise ValueError("invalid model vectors")
            batches.append(vectors)
        vectors = np.vstack(batches)
        result = []
        cursor = 0
        for group in groups:
            part = vectors[cursor:cursor + len(group)]
            cursor += len(group)
            if len(group) == 1:
                result.append(part[0])
            else:
                pooled = np.average(part, axis=0, weights=[weight for _, weight in group])
                norm = np.linalg.norm(pooled)
                if norm <= 1e-12:
                    raise ValueError("long unit pooled to zero vector")
                result.append((pooled / norm).astype(np.float32))
        return np.asarray(result, dtype=np.float32)

    def encode_similarity(self, texts):
        return self._encode(texts, "similarity")

    def encode_units(self, texts):
        return self._encode(texts, "clustering")

    def encode(self, texts):
        """AtlasState compatibility, still restricted to the shared worker."""
        return self.encode_units(texts)
