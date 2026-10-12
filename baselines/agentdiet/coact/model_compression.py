from litellm import completion

class _ModelCompressionStrategy:
    completion = staticmethod(completion)

    def __init__(
        self,
        model: str,
        api_endpoint: str,
        temperature: float = 0.6,
        top_p: float | None = 0.95,
        top_k: int | None = 20,
        min_p: float | None = 0.0,
        presence_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        max_tokens: int = 81920,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize model-backed compression settings.

        Args:
            model: LiteLLM model identifier.
            api_endpoint: API base URL for the model.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            min_p: Minimum probability parameter.
            presence_penalty: Presence penalty.
            repetition_penalty: Repetition penalty.
            max_tokens: Maximum tokens in compressed output.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum LiteLLM retry attempts.
        """
        self.model = model
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

