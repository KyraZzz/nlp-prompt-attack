import pytest
from src.run import prep_template


@pytest.mark.parametrize(
    "template, expected", [
        ("<cls> <sentence> . It was <mask> .",
         "<cls> <cap> <sentence> . <cap> It was <mask> ."),
        ("<cls> <sentence> ? <mask> , <question> .",
         "<cls> <cap> <sentence> ? <cap> <mask> , <question> ."),
        ("<cls> <premise> ? <mask> , <hypothesis> .",
         "<cls> <cap> <premise> ? <cap> <mask> , <hypothesis> ."),
        ("<cls> <mask> email : <text> .",
         "<cls> <cap> <mask> email : <text> ."),
        ("<cls> <tweet> . This post is <mask> .",
         "<cls> <cap> <tweet> . <cap> This post is <mask> .")]
)
def test_prep_template(template, expected):
    res = prep_template(template)
    assert res == expected
