from dataclasses import dataclass
from collections import Counter

# define data class for holding dataset-wide stats for the ParsedDataset:
@dataclass
class ParsedDatasetStats:
    steps_not_uniquely_covered: list
    n_times_step_taken: Counter
    galleries_not_uniquely_covered: list
    n_times_gallery_saved: Counter


# define data class for holding dataset-wide stats for the PostParsedDatset:
@dataclass
class PostParsedDatasetStats(ParsedDatasetStats):
    giant_component: set
    median_explore_mean: float
    median_explore_std: float
    median_exploit_mean: float
    median_exploit_std: float
