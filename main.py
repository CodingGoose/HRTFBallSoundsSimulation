from body_tracker import BodyTracker

bt = BodyTracker()

while True:
    bt.track_body()
    print(bt.timestamp)
    if bt.landmarks:
        for landmark in bt.landmarks:
            print(landmark)