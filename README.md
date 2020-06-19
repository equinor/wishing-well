# wishing-well
Team repo for Wishing Well team - Virtual summer internship 2020

To build do:

```docker build -t wishing-well-app .```

And to run:

```docker run -it --rm --name wwrunning wishing-well-app```


##To run with GUI in windows

Remember to use PowerShell, you might have issues using cmd.

Build like you normally would:

```docker build -t wishing-well-app .```

Then you have to install [VcXsrv](https://sourceforge.net/projects/vcxsrv/)

When it is installed, run XLaunch and go thrugh the wizard. Check off the Disable access control option.

When you are finished, VcXsrv should be running as an icon in the lowe right corner.

Then you need to find your IPv4 address by running:

```ipconfig```

Choose the address that you get your internet from, and put this into the DISPLAY variable:

```set-variable -name DISPLAY -value YOUR.IP.RIGHT.HERE:0.0```

Then you just have to run

```docker run -it --rm --name wwrunning DISPLAY=$DISPLAY wishing-well-app```

It should now display the GUI.
