import traceback
import json
import hashlib
import os
from sympy import *
from imgcat import imgcat
from pathlib import Path

from dump import dump

def title(var):
    print("#"*60)
    print("#", var)
    print()


def show(orig_exp, n=None):
    # http://docs.python.org/library/traceback.html
    stack= traceback.extract_stack()
    var= stack[-2][3]

    # strip away the call to dump()
    var= '('.join(var.split('(')[1:])
    var= ')'.join(var.split(')')[:-1])

    if n is not None:
        var = var.split(",")[:-1]
        var = ",".join(var)

    prefix = r"\verb|" + var + "| = "
    #dump(prefix)

    if n is None:
        title(var)
        print(latex(orig_exp))
        latex_str = prefix + latex(orig_exp)
        dump_png(f"${latex_str}$")
        return orig_exp

    title(f"{var}, {n} denom pulled out")


    exp = orig_exp

    exp_n = simplify(n * exp)
    #dump(exp)
    with evaluate(False):
        try:
            f = Rational(1, n)
        except TypeError:
            f = 1/n
        exp = Mul(f, exp_n)

        #pprint(exp)
        #print()

        if isinstance(exp_n, MatrixBase):
            latex_str = prefix + latex(f) + " " + latex(exp_n)
            print(latex_str)
            dump_png(f"${latex_str}$")
        else:
            print(latex(exp))
            latex_str = prefix + latex(exp)
            dump_png(f"${latex_str}$")
        print()

    return orig_exp

def dump_png(expr, filename=None, D=150, padding=10):
    """
    Render `expr` to a PNG via sympy.preview, using a cache system.

    The function hashes the expression and options to create a unique filename
    in the .cache directory. If the file already exists, it won't re-render.

    Args:
        expr: The sympy expression to render
        filename: Optional custom filename (ignores caching if provided)
        D: DPI for the rendering
        padding: Padding in pixels around the rendered image
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    # If filename is explicitly provided, use it without caching
    if filename is None:
        # Create a hash of the expression and options
        hash_input = str(expr) + str(D) + str(padding)
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()
        filename = cache_dir / f"{file_hash}.png"
    else:
        filename = Path(filename)

    # Only render if the file doesn't already exist
    if not filename.exists():
        # Create a temporary file for the initial rendering
        temp_filename = "tmp_render.png"
        preview(expr, viewer='file', filename=temp_filename, dvioptions=['-D', str(D)])

        # Add padding if requested
        if padding and padding > 0:
            from PIL import Image
            img = Image.open(temp_filename)
            w, h = img.size
            new_img = Image.new("RGBA", (w + 2 * padding, h + 2 * padding), (255, 255, 255, 0))
            new_img.paste(img, (padding, padding))
            new_img.save(filename)
            os.remove(temp_filename)
        else:
            # Just move the file to the cache location
            Path(temp_filename).rename(filename)

    # Display the image
    imgcat(filename.read_bytes())
