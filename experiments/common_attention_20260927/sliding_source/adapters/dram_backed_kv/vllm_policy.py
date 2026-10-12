"""Original Online Selector V2 with an engine-independent inference boundary.

Selection, catalog construction, goal maintenance and projection are imported
from the protected reference. Only model execution and tensor layout change.
"""
from hashlib import sha256
from types import SimpleNamespace as NS

from .backend import load_upstream

upstream = load_upstream()


class Policy(upstream.OnlineSelectorSession):
    def __init__(self, tokenizer, model_config, model, budget, options, session_id):
        import torch
        self.args = NS(model=model, switch_threshold=budget, selection_mode="selector", **options)
        self.tokenizer, self.torch = tokenizer, torch
        self.full_context = False
        self.session_id = session_id
        self.instance_id = "online-swe"
        self.turn_id = 0
        self.logical_tokens, self.pages = [], []
        self.page_turns = {}
        self.mandatory_end = self.cataloged_end = self.sealed_end = 0
        self.previous_active = ()
        self.ever_selected = set()
        self.previous_action = self.root_goal = None
        self.index = upstream.PageVectorIndex()
        self.goal_rewriter = upstream._create_goal_rewriter(self.args)
        self.goal_constructor = upstream.MaintainedGoalConstructor(self.goal_rewriter,
            entity_drift_steps=3, initialize_with_rewriter=self.goal_rewriter is not None)
        c = model_config
        self.bytes_per_token = c.layer_types.count("full_attention") * 2 * c.num_key_value_heads * c.head_dim * 2
        _, projection_hash = upstream._projection(torch, c.hidden_size,
            options["projection_dimension"], options["projection_seed"])
        layer = options["hidden_layer"] % len(c.layer_types)
        if not -len(c.layer_types) <= options["hidden_layer"] < len(c.layer_types):
            raise ValueError("hidden layer outside decoder stack")
        self.provenance = upstream.PageVectorProvenance(model=model, revision="local",
            hidden_layer=layer, pooling="last_token", projection_hash=projection_hash,
            dimension=options["projection_dimension"], dtype="int8",
            vector_source="online_main_model_hidden_states")
        self.publisher = upstream.PageVectorPublisher(self.index, self.provenance)
        self.goal_control_hash = upstream._sha_json({"constructor": self.goal_constructor.name,
            "config": self.goal_constructor._config()})
        self._page_vectors, self._goal_query_cache = {}, {}
        self._feature_extractor = upstream.DeterministicFeatureExtractor()
        # The real runtime lives in the GPU worker. Its resident IDs are copied
        # across the RPC boundary before constructing the original TurnContext.
        self.resident = ()
        self.runtime = NS(resident_page_ids=lambda sid: self.resident)

    def append(self, ids):
        start = len(self.logical_tokens)
        self.logical_tokens.extend(ids)
        for position in range(start, len(self.logical_tokens)):
            self.page_turns[position // upstream.PAGE_SIZE] = self.turn_id

    def ingest(self, report):
        for index, vector in report["vectors"]:
            self._page_vectors[int(index)] = upstream.CompactVector(
                tuple(vector["values"]), vector["dtype"], vector["scale"])
        self.resident = tuple(report["resident"])

    def _plan(self, context, goal):
        if self.args.budget_mode == "fixed":
            history = len(context.historical_page_ids) * upstream.PAGE_SIZE
            self.args.active_ratio = min(1.0, self.args.switch_threshold / max(1, history))
        return super()._plan(context, goal)

    def close(self):
        self.index.release_session(self.session_id)
        self.goal_constructor.release_session(self.session_id)
        if self.goal_rewriter is not None:
            self.goal_rewriter.close()
