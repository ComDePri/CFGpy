import json

with open(r"/home/royg/home/royg/CFGpy/CFGpy/behavioral/rm2.json") as f:
    rm2 = json.load(f)

game_version = rm2["publisherId"]
rm1 = []
for session in rm2["sessions"]:
    for rm2_event in session["events"]:
        rm1.append({
            "gameVersion": game_version,
            "player": session["id"],
            "serverTime": rm2_event['serverTimestamp'],
            "userTime": rm2_event['userTimestamp'],
            "type": rm2_event["type"].lower(),
            "id": rm2_event["id"],
            "customData": rm2_event["customData"]
        })

with open(r"/home/royg/home/royg/CFGpy/CFGpy/behavioral/converted.json", "w") as f:
    json.dump(rm1, f)
