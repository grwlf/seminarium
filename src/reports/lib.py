import matplotlib.pyplot as plt

from os import makedirs, environ, devnull
from os.path import join as osjoin
from altair import Chart
from contextlib import contextmanager
from subprocess import call, DEVNULL
from shutil import copyfile
from typing import Optional

# Altair
from altair_saver import save as altair_save

# Pylightnix
from pylightnix import (Registry, Config, Build, DRef, RRef, realize,
                        instantiate, mklens, mkdrv, mkconfig, selfref,
                        match_only, build_wrapper, writejson, readjson,
                        writestr, fsinit, realize1)

fsinit()

IMGDIR='img'

Filepath=str

@contextmanager
def markdown_image(name:str, alt:str='', attrs:str='',
                   with_inplace:bool=True,
                   with_enlarge:bool=False):
  if not name.endswith('.png'):
    name+='.png'
  fn:Filepath=osjoin(IMGDIR,name)
  makedirs(IMGDIR, exist_ok=True)
  yield fn
  if with_inplace:
    print("![%s](%s){%s}"%(alt,fn,attrs))
  if with_enlarge:
    print(f"([{'Enlarge' if with_inplace else 'Link'}](%s))"%(fn,))

@contextmanager
def markdown_image2(name:str, alt:str='', attrs:str='',
                   with_inplace:bool=True,
                   with_enlarge:bool=False):
  if not name.endswith('.png'):
    name+='.png'
  fn:Filepath=osjoin(IMGDIR,name)
  makedirs(IMGDIR, exist_ok=True)
  if with_inplace:
    tag="![%s](%s){%s}"%(alt,fn,attrs)
  if with_enlarge:
    tag=f"([{'Enlarge' if with_inplace else 'Link'}](%s))"%(fn,)
  yield (fn,tag)

@contextmanager
def latex_image(name:str, alt:str='', attrs:str=''):
  assert not name.endswith('.png')
  fn:Filepath=osjoin(IMGDIR,name+'.png')
  makedirs(IMGDIR, exist_ok=True)
  yield fn
  print("\\includegraphics[width=\\textwidth,keepaspectratio=true]{%s}"%(name,))

def feh(path:str):
  call(['eog',path])


def plotaltair(fpath:Filepath, c:Chart):
  altair_save(c, fpath)


def latex2pdf(expr:str,
              out_pdf:Filepath,
              font:str='normalsize',
              font_white:bool=False,
              texname:str='latex'
              )->None:
  makedirs('_tex',exist_ok=True)
  tex=f'_tex/{texname}.tex'
  pdf=f'_tex/{texname}.pdf'
  with open(tex, 'w') as f:
    f.write(('\\documentclass{standalone}\n'
             '\\usepackage{varwidth}\n\\usepackage{amsmath}\n'
             '\\usepackage{xcolor}\n' +
             ('\\color[rgb]{0.77,0.77,0.77}\n' if font_white else '') +
             '\\begin{document}\n'
             '\\begin{varwidth}{\\linewidth}\n'
             '\\strut{\\'+font+'{\n$$\n'+expr.strip()+'\n$$\n}}\n'
             '\\end{varwidth}\n\\end{document}'))
  with open('_tex/latex2pnf_stdout_stderr.txt','w') as f:
    ret=call(['pdflatex', f'{texname}.tex'],cwd='_tex', stdout=f, stderr=f)
    assert ret==0, f"pdflatex returned {ret}"
  ret=call(['mv', pdf, out_pdf])
  assert ret==0, f"cp returned {ret}"


def latex2png(expr:str, path:Filepath, geom:Optional[str]=None, **kwargs)->None:
  makedirs('_tex',exist_ok=True)
  out_png=path
  in_pdf=osjoin('_tex','out.pdf')
  latex2pdf(expr, in_pdf, **kwargs)
  ret=call(['convert'] +
           (['-density', geom] if geom is not None else []) +
           [in_pdf, '-quality', '90', out_png])
  assert ret==0, f"convert returned {ret}"


# environ['TERMINAL_IMAGES_COLS_PER_INCH']='10' # str(256 / (598 / 25.4))
# environ['TERMINAL_IMAGES_ROWS_PER_INCH']='5'  # str(67 / (315 / 25.4))
# environ['LC_NUMERIC']='C'

def kittydraw(path:str,
              row:Optional[float]=None,
              col:Optional[float]=None)->None:
  """ Send image to Kitty and then open it in "Feh" image viewer """
  ret=call(['upload-terminal-image.sh']+
           (['-c',str(int(col))] if col else []) +
           (['-r',str(int(row))] if row else []) +
           [path], stderr=DEVNULL)
  assert ret==0, f"upload-terminal-image.sh returned {ret}"
  # ret=call(['feh',path])
  # assert ret==0, f"feh returned {ret}"


def kittylatex(latex:str, nrows:Optional[int]=None)->None:
  makedirs('_tex',exist_ok=True)
  path=osjoin('_tex',f'kittylatex.pdf')
  latex2pdf(latex,path,font='Huge',font_white=True)
  kittydraw(path,nrows,2*nrows if nrows else None)

def kittyshow():
  makedirs('_plt',exist_ok=True)
  p=osjoin('_plt','lastplot.png')
  plt.savefig(p, transparent=True)
  kittydraw(p)

kshow=kittyshow





