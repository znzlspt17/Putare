"""Colab 재시도 비격리 채점 진입점."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.retry80_feedback import grade
if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: retry80_feedback_grade.py GENERATION_DIR VENV')
    grade(sys.argv[1],sys.argv[2])
