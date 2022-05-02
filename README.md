# EDH_matchmaker #

## Intro ##

Long over due update on readme... here are some recent highlights.
The CLI option is no longer supported. The program, for the time being, is meant for use through UI.
Eventually, there will probably be some sort of Discord integration, but first, the rough edges have to be brushed out.

## Installation: ##

The software was developed on Linux and installation should be rather straight forward. It's cross-platform, so I believe it should work on mac too, but I can't check. Either way, Windows has it's own issues, so here's how you do it.

### Windows ###

First, you need python 3 and pip working.
The easiest way to do this on Windows is by downloading the latest python version from official site (an installer). Generally, following default steps is OK, but you have to change one thing:

**Important:** When installing, you have a couple of checkboxes. The one you want to tick is the "Add python to PATH" (or something similar).

Now this should generally work, but sometimes, it doesn't. Ask Microsoft why.

At any point during the following steps, you can check if python and pip are recognized in your command prompt/powershell. You can check it by running the command `python --version` and/or `pip --version`. If both are outputing something, you can skip the following steps, otherwise you have to troubleshoot your python installation and you can continue.

To make sure it's added, go to your start menu, and search for "Edit the system environment variables". Might not be exact steps, but something along those lines (*use google*).

Once you find it, depending on the version of Windows you have, you will have different sections. Search for "Path" variable and check if your python installation dirrectory is also added. Since this is not really the scope of this installation guide, you will have to figure this on your own as it can differ a bit from system to system, but generally something along those lines.

Right, now that you've checked that python is actually in your Path variable, it's still possible that python is not recognized. Windows 10 can

### Once python is sorted, the actual installation

First, you have to install the require packages. This is rather simple.
You have to open your command prompt/powershell/terminal in the directory where you've downloaded the software.
On windows, you can do this from the file menu. I assume linux/mac users shouldn't have a problem here.

The command is `pip install -r requirements.py`

This will install the required libraries for the program to run

## Running

Again, move to the directory and fire up:

`python run_ui.py`

On windows, you can also add a shortcut for this with different configurations.

## Runtime options

The program can also be run with some extra command parameters to set it up.

You can check it by running `python run_ui.py --help`. Again, you can add a shortcut WITH those parameters, if you don't want to add them every time manually.

Alternatively, you can also set everything up through GUI once the program is running.

# Closing words

This software is still in development. The best thing you can do to help me is by testing it and submiting bugs. Best way to do it is here, on github - open an issue and describe what's happening and how to reproduce it, so I can fix it. You can also add your tournament log.

Also you can star this project for more exposure :)

Best regards,
/E