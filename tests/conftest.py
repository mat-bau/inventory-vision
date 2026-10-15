import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Class names embedded in model/best.pt (checked from the checkpoint metadata).
MODEL_CLASSES = {0: "Resistor", 1: "Capacitor", 2: "Inductor", 3: "Diode", 4: "LED",
                 5: "Transistor", 6: "IC", 7: "Connector", 8: "Switch"}
