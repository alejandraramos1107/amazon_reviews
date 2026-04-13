import io
import re
import logging
from collections import Counter
from scipy.io import arff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_arff_with_unique_names(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    attr_counter = Counter()
    out_lines = []
    attr_pattern = re.compile(
        r"""^\s*@attribute\s+(?:'([^']*)'|"([^"]*)"|(\S+))""",
        re.IGNORECASE,
    )

    for line in lines:
        match = attr_pattern.match(line)
        if match:
            name = match.group(1) or match.group(2) or match.group(3)
            attr_counter[name] += 1
            if attr_counter[name] > 1:
                unique_name = f"{name}_{attr_counter[name] - 1}"
                line = attr_pattern.sub(f"@attribute '{unique_name}'", line, count=1)
        out_lines.append(line)

    logger.info(f"Archivo .arff procesado: {path}")
    return arff.loadarff(io.StringIO("".join(out_lines)))