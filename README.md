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

## Deployment instructions (temporary)
==================================
1. Verify the new changes are pushed to the master branch in GitHub.
2. TO DO.

"""
