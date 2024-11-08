NOTE: MUCH OF THE CODE IS COMMENTED OUT TO PREVENT MULTIPLE API CALLS INCURRING LARGE FEES. THE CODE IS STILL PRESENTED IN COMMENTS HOWEVER.

To mininmize API calls, the code is segmented into three steps (I chose not to run steps 1 and 2 repeatedly by integrating them into one file, instead separating them so I could run the main file on their output.):
1. Calling Claude API on the initial prompt (see claude_api.py) to generate the JSON response for ambiguity/object information. This information should be stored in "response.json". 
2. Calling Background Removal API (see src/background/test_bg.py). The resulting image should be stored in the without_background image. This assumes an image that is the generated image is already stored in the static folder
3. The main file. This needs an image in the static folder, an image in the without_backround folder, a prompt in "prompt.txt" and a response.json file.