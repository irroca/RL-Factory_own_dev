"""Domain-weighted sampler for multi-domain RL training.

Assigns per-sample weights based on domain, so that minority domains
(e.g. financial, science) are sampled more frequently relative to
majority domains (e.g. biomedical).

Supports three weighting strategies:

1. **equal** — all domains are sampled with equal probability.
   Within each domain, samples are drawn uniformly.

2. **custom** — user-specified per-domain weights via
   ``data.sampler.domain_weights``, e.g. ``{biomedical: 2, financial: 1, science: 1}``.

3. **temperature** — smooth between natural proportions (T=1) and
   equal proportions (T→∞) using  w_i ∝ n_i^{1/T}.
   Controlled by ``data.sampler.temperature`` (default=3).

Configuration example (in training shell script):
    data.sampler.class_path=verl.experimental.dataset.domain_weighted_sampler \
    data.sampler.class_name=DomainWeightedSampler \
    data.sampler.strategy=equal \
    data.sampler.domain_key=data_source
"""

import logging
from collections import defaultdict
from collections.abc import Iterator, Sized
from typing import Optional

import torch
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler

logger = logging.getLogger(__name__)


class DomainWeightedSampler(AbstractSampler):
    """Weighted random sampler that balances across domains."""

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        # data_source is an RLHFDataset with a .dataframe attribute
        self.data_source = data_source
        self.data_config = data_config

        sampler_cfg = data_config.get("sampler", {})
        self.domain_key = sampler_cfg.get("domain_key", "data_source")
        self.strategy = sampler_cfg.get("strategy", "equal")
        self.seed = data_config.get("seed", 1)

        # Build domain -> indices mapping
        dataframe = data_source.dataframe
        domain_column = dataframe[self.domain_key]

        domain_indices: dict[str, list[int]] = defaultdict(list)
        for idx, domain in enumerate(domain_column):
            domain_indices[domain].append(idx)

        self.domain_indices = dict(domain_indices)
        self.domains = sorted(self.domain_indices.keys())
        self.n_samples = len(data_source)

        # Compute per-domain sampling weights
        domain_weights = self._compute_domain_weights(sampler_cfg)

        # Convert to per-sample weights
        sample_weights = torch.zeros(self.n_samples, dtype=torch.double)
        for domain in self.domains:
            indices = self.domain_indices[domain]
            # weight per sample = domain_weight / domain_size
            w = domain_weights[domain] / len(indices)
            for i in indices:
                sample_weights[i] = w

        self.sample_weights = sample_weights
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        # Log sampling plan
        logger.info("DomainWeightedSampler initialized:")
        for domain in self.domains:
            n = len(self.domain_indices[domain])
            w = domain_weights[domain]
            expected_per_epoch = w * self.n_samples
            logger.info(
                f"  {domain}: {n} samples, weight={w:.4f}, "
                f"expected samples/epoch={expected_per_epoch:.0f}"
            )

    def _compute_domain_weights(self, sampler_cfg) -> dict[str, float]:
        """Compute normalized domain weights based on strategy."""
        domain_sizes = {d: len(self.domain_indices[d]) for d in self.domains}

        if self.strategy == "equal":
            # Each domain gets equal total weight
            raw = {d: 1.0 for d in self.domains}

        elif self.strategy == "custom":
            # User-provided domain weights
            user_weights = sampler_cfg.get("domain_weights", {})
            raw = {}
            for d in self.domains:
                # Strip prefix "multi_domain_" for user-friendly keys
                short_name = d.replace("multi_domain_", "")
                w = user_weights.get(short_name, user_weights.get(d, 1.0))
                raw[d] = float(w)

        elif self.strategy == "temperature":
            temperature = float(sampler_cfg.get("temperature", 3.0))
            raw = {d: n ** (1.0 / temperature) for d, n in domain_sizes.items()}

        else:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                "Choose from: equal, custom, temperature"
            )

        # Normalize
        total = sum(raw.values())
        return {d: w / total for d, w in raw.items()}

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self.sample_weights,
            num_samples=self.n_samples,
            replacement=True,
            generator=self.generator,
        )
        yield from indices.tolist()

    def __len__(self) -> int:
        return self.n_samples
