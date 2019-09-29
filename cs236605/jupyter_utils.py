import pathlib
import subprocess
import sys
import time


def nbconvert(nb_path, execute=False, inplace=False, clear_output=False,
              debug=False, stdout=False, allow_errors=False, timeout_sec=1800):

    args = ["jupyter", "nbconvert"]
    if execute:
        args.append("--execute")
    if allow_errors:
        args.append("--allow-errors")
    if clear_output:
        args.append("--ClearOutputPreprocessor.enabled=True")
    if inplace or clear_output:
        args.append("--inplace")
    if debug:
        args.append("--debug")
    if stdout:
        args.append("--stdout")
    if timeout_sec is not None:
        args.append(f"--ExecutePreprocessor.timeout={timeout_sec}")
    args.append(nb_path)

    true_flags = []
    for k, v in locals().items():
        if v is True:
            true_flags.append(k)
    true_flags = str.join('|', true_flags)

    print(f'>> Running nbconvert on notebook {nb_path} [{true_flags}]')
    ts = time.time()
    subprocess.check_output(args)
    print(f'>> Finished nbconvert on notebook {nb_path}, '
          f'elapsed={time.time()-ts:.3f}s')


def nbmerge(nb_paths, output_filename):
    if not output_filename.endswith('.ipynb'):
        output_filename += '.ipynb'

    args = ['nbmerge', '-o', output_filename, '-v']
    args.extend(nb_paths)

    nb_names = [pathlib.Path(nb_path).stem for nb_path in nb_paths]
    print(f'>> Running nbmerge on notebooks {str.join(", ", nb_names)}')

    subprocess.check_output(args)
