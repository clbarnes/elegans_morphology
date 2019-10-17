from pathlib import Path

PACKAGE_ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

NML_ROOT = PROJECT_ROOT / "c302" / "c302" / "NeuroML2"
POW_CONF_PATH = PROJECT_ROOT / ".pow" / "pow.conf"

WORM_IDENT = 'http://openworm.org/entities/bio#worm0-data'
