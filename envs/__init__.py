from .base import Env as BaseEnv
from .mmbase import MMEnv
from .search import SearchEnv
from .multi_domain_search import MultiDomainSearchEnv
from .vision import VisionEnv
from .reward_rollout_example import RewardRolloutEnv

# Define public interface for the module
# Specifies which classes will be imported when using "from module import *"
__all__ = ['BaseEnv', 'SearchEnv', 'MultiDomainSearchEnv', 'RewardRolloutEnv', 'VisionEnv', 'MMEnv']


# Environment registry mapping - connects environment names to their corresponding classes
# Facilitates dynamic environment creation by referencing names as strings
TOOL_ENV_REGISTRY = {
    'base': BaseEnv,
    'mmbase': MMEnv,
    'search': SearchEnv,
    'multi_domain_search': MultiDomainSearchEnv,
    'reward_rollout': RewardRolloutEnv,
    'vision': VisionEnv
}