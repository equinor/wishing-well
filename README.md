# wishing-well
Team repo for Wishing Well team - Virtual summer internship 2020


## Showing plot in browser

First, navigate to the wishing-well repository on your computer.
Then build the container using:

```docker build -t wishing-well-app .```

Then run it with:

```docker run -p 8080:8080 --name wwrunning wishing-well-app```


It should now display the plot at http://localhost:8080/

When you have changed your code, you need to run

```docker rm -f wwrunning```
Then build and run again.

Or, if you want to do it like the pros, just copy-paste this into your terminal (Powershell) It will remove the old container, build a new one, and run it:
```docker rm -f wwrunning; docker build -t wishing-well-app .; docker run -p 8080:8080 --name wwrunning wishing-well-app ```

