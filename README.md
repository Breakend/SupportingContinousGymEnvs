<img src="assets/Mcgill.png" width=25% align="right" />

# gym-extensions
This python package is an extension to OpenAI Gym for auxiliary tasks (multitask learning, transfer learning, inverse reinforcement learning, etc.)


## Dependencies

- [Python 3.5.2](https://www.python.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [MuJoCo](http://mujoco.org/) (Optional)
- [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco) (Optional)
- [roboschool](https://github.com/openai/roboschool) (Optional)

## Installation

Check out the latest code:
```bash
git clone https://github.com/Breakend/gym-extensions-multitask.git
```

### Install without MuJoCo Support
```bash
pip3 install -e .
```

### Install with MuJoCo Support
Install MuJoCo according to [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco).
1. Obtain license for [MuJoCo](http://mujoco.org/)
2. Download MuJoCo 1.50 binaries
3. Unzip into `mjpro150` directory `~/.mujoco/mjproj150` and place license 
at `~/.mujoco/mjkey.txt`
4. Finally, install gym-extensions with mujoco-py enabled:

```bash
pip3 install -e .[mujoco]
```

### Test Installation
```
nosetests -v gym_extensions
```

### Possible Issues

Due to the dependency on OpenAI gym you may have some trouble when installing gym on macOS, to remedy:

```bash
# as per: https://github.com/openai/gym/issues/164
export MACOSX_DEPLOYMENT_TARGET=10.12; pip install -e .
```

Also, if you get the following error:
```
>>> import matplotlib.pyplot
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/florian/anaconda2/lib/python2.7/site-packages/matplotlib/pyplot.py", line 115, in <module>
    _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
  File "/home/florian/anaconda2/lib/python2.7/site-packages/matplotlib/backends/__init__.py", line 32, in pylab_setup
    globals(),locals(),[backend_name],0)
  File "/home/florian/anaconda2/lib/python2.7/site-packages/matplotlib/backends/backend_gtk.py", line 19, in <module>
    raise ImportError("Gtk* backend requires pygtk to be installed.")
ImportError: Gtk* backend requires pygtk to be installed.
```
the easiest fix is to switch backends for matplotlib. You can do this by setting `backend: TkAgg` in `~/.config/matplotlib/matplotlibrc` or `~/.matplotlib/matplotlibrc`

## More info

More information will be provided on our doc website: https://breakend.github.io/gym-extensions/

## Contributions

To contributing environments please use the general directory structure we have in place and provide **pull requests**. We're still working on making this extension to OpenAI gym the best possible so things may change. Any changes to existing environments should involve an incremental update to the name of the environment (i.e. Hopper-v0 vs. Hopper-v1). If you are not associated with McGill and contribute significantly, please add your association to:

<pre>docs/index.md</pre>

## Contributors

Here's a list of contributors!

+ <a href="https://github.com/Breakend">@Breakend</a>
+ <a href="https://github.com/weidi-chang">@weidi-chang</a>
+ <a href="https://github.com/florianshkurti">@florianshkurti</a>
+ <a href="https://github.com/johannah">@johannah</a>
+ <a href="https://github.com/vBarbaros">@vBarbaros</a>
