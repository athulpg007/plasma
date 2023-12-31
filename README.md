## Local development instructions
=================================
1. Requires a virtualenv with Python 3.11.x, or 3.12 (but not > 3.12).
2. Install the dependencies with pip install -r requirements.txt.
3. The data files used are located in data/.
4. For local development, set debug=True in app.run_server() at the end of the file.
5. The server must be run from the directory which contains the app.py file.
6. To run the server, from the PyCharm terminal: "cd app_v1", followed by "python app.py".
7. Go to http://127.0.0.1:8050/ in your web browser.
8. Test the application from the web browser.
9. Check for errors in the PyCharm terminal and the Dash debug panel at the bottom right of the web browser.
10. Before deploying to production server, verify debug=False in app.run_server().
==================================

## Deployment instructions (temporary for existing pilot phase)
==================================
1. Verify the new changes are pushed to the master branch in GitHub (git push origin master).
2. Log into the production server and do pull the changes (git pull).
3. Go to the plasma directory (cd plasma). Activate the virtual env (source venv/bin/activate)
4. Kill the running gunicorn process (pkill gunicorn)
5. Restart the server (gunicorn app:server)

The above instructions should be replaced with a Docker image based deployment using the .Dockerfile in the future.

To build the docker image (currently for local development)
```
docker build -t plasma:1.0 .
```
To run the image locally:
```
docker run -p 8050:8050 --name plasma-container plasma:1.0
```

Go to `http://127.0.0.1:8050/` to access the application.
