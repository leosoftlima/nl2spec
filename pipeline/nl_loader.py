from pathlib import Path
from typing import Dict, List
from nl2spec.logging_utils import get_logger

log = get_logger(__name__)

SUPPORTED_DOMAINS = {"io", "lang", "util", "net"}

def load_nl_scenarios_by_domain(root: str) -> Dict[str, List[dict]]:
    root = Path(root)
    log.info("Loading NL baseline from: %s", root)

    if not root.exists():
        log.error("NL baseline directory not found: %s", root)
        raise FileNotFoundError(f"NL baseline directory not found: {root}")

    scenarios: Dict[str, List[dict]] = {}

    for domain_dir in root.iterdir():
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name.lower()
        txt_files = list(domain_dir.glob("*.txt"))

        log.info("Domain '%s': %d NL files", domain, len(txt_files))

        scenarios[domain] = []

        for txt in txt_files:
            scenarios[domain].append({
                "id": txt.stem,
                "domain": domain,
                "natural_language": txt.read_text(encoding="utf-8").strip(),
                "_path": str(txt),
            })

            log.debug("Loaded NL scenario: %s/%s", domain, txt.name)

    log.info("Finished loading NL baseline (%d domains)", len(scenarios))
    return scenarios