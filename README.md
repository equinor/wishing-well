# wishing-well

Team repo for Wishing Well team - Virtual summer internship 2020

## Showing plot in browser

First, navigate to the wishing-well repository on your computer.
Then build the container using:

`docker build -t wishing-well-app .`

Then run it with:

`docker run -it --mount type=bind,source="$(pwd)",target=/app -p 8080:8080 --name wwrunning wishing-well-app`

It should now display the plot at http://localhost:8080/

When you have changed your code, you need to run

`docker rm -f wwrunning`

Then just run again.

Or, if you want to do it like the pros, just copy-paste this into your terminal (Powershell) It will remove the old container, run it again and show the plot:

`docker rm -f wwrunning; docker run -dit --mount type=bind,source="$(pwd)",target=/app -p 8080:8080 --name wwrunning wishing-well-app; Start-Sleep -s 2; Start "http://localhost:8080/"; docker logs -f wwrunning`

(The Start-Sleep is there to give the server time to start. docker logs shows output)

For debugging, this is better. It prints output from the program as it goes (not when its finished):

`docker rm -f wwrunning; docker run -it --mount type=bind,source="$(pwd)",target=/app -p 8080:8080 --name wwrunning wishing-well-app`

Note that if you edit for example the env, you have to build the container again for the changes to take effect. If you are not seeing your changes being applied, try rebuilding.
