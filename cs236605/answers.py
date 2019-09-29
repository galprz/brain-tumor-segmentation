import re
from IPython.display import display, Markdown


SOLUTION_BLOCK_PATTERN = \
    re.compile(r'(?P<indent>[ \t]+)'
               r'(?P<start>#+\s*={3,}\s+YOUR CODE:\s+={3,}\s*)'
               r'(?P<newline>\r?\n)'
               r'(?P<code>((?!\s*#+\s*={3,}).*\r?\n)+)'
               r'(?P<end>\s*#+\s*={3,}\s*\r?\n)',
               )

ANSWER_BLOCK_PATTERN = \
    re.compile(
        r'(?P<start>""")'
        r'(?P<newline>\r?\n)*'
        r'(?P<marker>\*{2}\s*your answer:\s*\*{2})'
        r'(\r?\n)*'
        r'(?P<answer>((?!""").*\r?\n)+)'
        r'(?P<end>""")',
        re.IGNORECASE)

SOLUTION_BLOCK_REPLACEMENT = r"raise NotImplementedError()"

ANSWER_BLOCK_REPLACEMENT = r"""
Write your answer using **markdown** and $\\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\\pi} -1 = 0$
"""


def clear_solutions(py_file_content):
    new_content, n_subs_code = SOLUTION_BLOCK_PATTERN.subn(
        '\g<indent>\g<start>\g<newline>'
        f'\g<indent>{SOLUTION_BLOCK_REPLACEMENT}\g<newline>'
        '\g<end>',
        py_file_content)

    new_content, n_subs_answers = ANSWER_BLOCK_PATTERN.subn(
        '\g<start>\g<newline>\g<marker>'
        '\g<newline>\g<newline>'
        f'{ANSWER_BLOCK_REPLACEMENT}'
        '\g<newline>\g<end>',
        new_content)

    if not (n_subs_code + n_subs_answers):
        new_content = None

    return new_content, n_subs_code, n_subs_answers


def display_answer(content):
    display(Markdown(content))
