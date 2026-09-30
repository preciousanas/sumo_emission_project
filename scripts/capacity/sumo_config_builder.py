from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET


def build_sumocfg_for_run(
    base_sumocfg_path: Path,
    output_sumocfg_path: Path,
    net_file_path: Path,
    route_file_path: Path,
    begin: int = 0,
    end: int = 7200,
    step_length: float = 1.0,
) -> None:
    """
    Create a per-run sumocfg that points to the provided net/rou files.

    This avoids modifying your existing config/traffic.sumocfg which currently
    contains absolute Windows paths.

    Note on `end` default:
        Demand flows in config/traffic.rou.xml inject during [0, 3600].
        We let the simulation run until 7200 so any vehicles whose insertion
        was delayed near saturation (V2800-V3000) have extra time to enter,
        clear the intersection, and arrive. The analysis window is still
        restricted to the demand window [0, 3600] by Fix 2 in
        summary_parser.py, so extending `end` does NOT contaminate capacity
        measurements - it just improves demand-served completeness.
    """
    cfg = ET.Element("configuration")

    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value=str(net_file_path))
    ET.SubElement(inp, "route-files", value=str(route_file_path))

    time = ET.SubElement(cfg, "time")
    ET.SubElement(time, "begin", value=str(begin))
    ET.SubElement(time, "end", value=str(end))

    proc = ET.SubElement(cfg, "processing")
    ET.SubElement(proc, "step-length", value=str(step_length))

    output_sumocfg_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(cfg).write(output_sumocfg_path, encoding="utf-8", xml_declaration=True)
