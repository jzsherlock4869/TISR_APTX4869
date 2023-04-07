ps -ef | grep train_track1 | awk -F" " '{print $2}' | xargs kill
