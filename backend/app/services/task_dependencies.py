"""Select author pins for the selected task, leaving dependency closure to pip's solver."""
import re
from packaging.requirements import Requirement, InvalidRequirement

ALIASES = {'libzero':'zero', 'pillow':'pil', 'opencv-python':'cv2', 'pyyaml':'yaml',
           'scikit-learn':'sklearn', 'beautifulsoup4':'bs4', 'python-dateutil':'dateutil',
           'sentence-transformers':'sentence_transformers', 'nvidia-ml-py':'pynvml'}


def normalized(name): return re.sub(r'[-_.]+','-',name).casefold()


def task_requirements(lines, imports):
    """Preserve exact constraints and environment markers; record every excluded package.

    Includes and index directives need their full author file and are not pruned.
    """
    if not imports or any(line.startswith(('-r','-c','--')) for line in lines): return list(lines),[],[]
    used = {normalized(name) for name in imports}
    selected, excluded, distributions = [],[],set()
    for line in lines:
        try: req = Requirement(line)
        except InvalidRequirement:
            selected.append(line); continue
        package = normalized(req.name)
        module = normalized(ALIASES.get(package,package))
        if package in used or module in used:
            selected.append(line); distributions.update({package,module})
        else: excluded.append(line)
    missing = [name for name in imports if normalized(name) not in distributions]
    if not selected and not used.intersection({'torch','tensorflow','numpy','scipy','pandas','sklearn','scikit-learn','transformers','datasets','jax'}):
        return list(lines),[],[]  # Incomplete local source is not evidence to discard every author dependency.
    return selected,excluded,missing
