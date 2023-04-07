import pytest
import sys
sys.path.insert(0, "/jmain02/home/J2AD015/axf03/yxz79-axf03/nlp-prompt-attack/src")

from run import prep_template

def test_prep_template():
    template = "<cls> <sentence> . It was <mask> ."
    print(prep_template(template))