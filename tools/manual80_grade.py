"""Colab 비격리 채점 진입점."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.manual80 import grade
if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: manual80_grade.py GENERATION_DIR VENV')
    grade(sys.argv[1],sys.argv[2])
