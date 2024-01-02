This is an application named PAT that traces the movements of Arabidopsis circumnutation using a [U-Net convolutional neural network model](https://en.wikipedia.org/wiki/U-Net)

The [Mao, et al. (2023) _A deep learning approach to track Arabidopsis seedlings’ circumnutation from time-lapse videos_](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-023-00984-5) paper explains the research as well as the meanings of the program option settings

There are two versions of the PAT code in this folder. One is a standalone python program, and the other is Jupyter notebook. It is not necessary to setup Jupyter to run the standalone program.

# Installation and Setup for PAT running in Python
[//]: <> (These instructions don't consider setting up Python virtual environments or whether using Conda would have been a better call. A topic for another day.)

[//]: <> (Here's what I've done so far to setup for klt/ml on MacOS Ventura 13.4.1, assuming a fresh-ish MacOS install:)

Clone this source repository to your local machine:

```
git clone https://github.com/Plant-Tracer/PlantTracerML.git
```

Open Terminal and individually run in the following commands:

If you don't already have brew, python3, and pip on your machine, you need them and here is one way to get them:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update && brew upgrade
brew install python3
brew install python-tk
python3 -m pip install --upgrade pip
```

Install the python packages that PAT needs:
```
cd .....PlantTracerML
python3 -m pip install -r requirements.txt
```

# Running PAT

Open a Terminal window.
Go your cloned repo folder named PlantTracerML, for example:
```
cd ~/git/PlantTracer/PlantTracerML
```

Execute:
```
python3 PAT.py
```


# Using PAT
Follow those steps to track the plant's movement:
1. Press the **Select video** button and select a video file, press the **Open** button
1. Optionally, press the **Select apex**, draw a square arond the apex area, and the press the **Confirm** button
1. Press the **Track** button to start the tracking process
1. Press the **Plot the graphs** button to view the result graphs
1. Press the **Save the results** button to save the results to file

## Program options
These are set with a column of buttons on the right side panel of the PAT frame. Their function and range of settings is explained more fully in the paper referenced above.

### Search range 1
### Search range 2
These options are used to change the hyperparameters of the deep learning model. The default setting works for most of the videos.

### Scale (draw a line)
We can draw a line in the video, compare with the ruler in the video and input the length of it to tell the program the distance scales between video and the real world. If you do it, the distance unit of results will change to millimeters in the real world.

### Frame interval
We can specify the time interval between frames. If you do it, the time unit of results will change to seconds in the real world.

### Enable color filter
This option is disabled by default and works for most cases, you should NOT enable it unless your video does not have a solid color background. 

# Installation and Setup for PAT running as a Jupyter notebook

These instructions Homebrew, python3, and python-tk has already been installed as above. Jupyter uses an embedded Iron Python interpreter by default, but does expect to find python-tk in its default package repository. These instructions use Jupyter Lab but it is likely that Jupyter Notebook would also work.
```
brew install jupyterlab
python3 -m pip install --upgrade pip
cd klt/ml
jupyter lab
Browse to localhost:8888
Open PAT.ipynb
Launcher — Console - Python3 (ipykernel)
!pip3 install -r requirements.txt #if this is a different package repository than in the default python3 installation
```
I got a 
```
ModuleNotFoundError: No module named '_tkinter'
```
at this point, and that was mysterious. I restarted JupyterLab and the error went away. So maybe installing the python packages before starting JupyterLab would have worked better, assuming that both the command line python environment and the Juptyer Python kernel are using the same package repository.
