"""Harder tool-calling cases: where the format instruction could matter.

Nested objects and arrays in arguments, parallel calls, a tool with no
arguments, strings that themselves contain braces or JSON. If the doubled
braces in the instruction ever confuse a model, it should be here.
"""
import json

TOOLS = [
  {"type":"function","function":{"name":"get_weather","description":"Current weather for a city.",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
  {"type":"function","function":{"name":"get_time","description":"Current UTC time. Takes no arguments.",
    "parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"create_event","description":"Create a calendar event.",
    "parameters":{"type":"object","properties":{
      "title":{"type":"string"},
      "start":{"type":"string","description":"ISO 8601, e.g. 2026-10-12T14:00"},
      "attendees":{"type":"array","items":{"type":"string"},"description":"email addresses"},
      "location":{"type":"object","properties":{"room":{"type":"string"},"floor":{"type":"integer"}}}},
      "required":["title","start"]}}},
  {"type":"function","function":{"name":"save_note","description":"Save a text note verbatim.",
    "parameters":{"type":"object","properties":{"text":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}},"required":["text"]}}},
  {"type":"function","function":{"name":"set_config","description":"Set a configuration value.",
    "parameters":{"type":"object","properties":{"key":{"type":"string"},"value":{"type":"object","description":"arbitrary JSON object"}},"required":["key","value"]}}},
]

CASES = []
# nested object + array
for title, start, who, room, floor in [
    ("Design review","2026-10-12T14:00",["anna@x.io","li@x.io"],"Aurora",3),
    ("1:1 with Omar","2026-10-13T09:30",["omar@x.io"],"Birch",2),
    ("Quarterly planning","2026-10-15T10:00",["ceo@x.io","cfo@x.io","cto@x.io"],"Summit",7),
    ("Retro","2026-10-16T16:00",["team@x.io"],"Cedar",1),
    ("Onboarding","2026-10-20T11:00",["new@x.io","hr@x.io"],"Dune",4),
    ("Security sync","2026-10-21T15:00",["sec@x.io"],"Echo",5),
]:
    CASES.append({"kind":"nested","user":f"Create an event \"{title}\" starting {start} with attendees {', '.join(who)}, in room {room} on floor {floor}.",
                  "calls":[{"name":"create_event","arguments":{"title":title,"start":start,"attendees":who,"location":{"room":room,"floor":floor}}}]})
# array only
for text, tags in [("Buy milk",["errand"]),("Call the bank about the card",["finance","todo"]),("Idea: witness for PRs",["idea","oss"]),
                   ("Renew passport before June",["todo","travel"]),("Read the ImpossibleBench paper",["reading"])]:
    CASES.append({"kind":"array","user":f"Save a note \"{text}\" with tags {', '.join(tags)}.",
                  "calls":[{"name":"save_note","arguments":{"text":text,"tags":tags}}]})
# strings containing braces / JSON
for text in ['Template is {name} and {date}', 'Use the pattern {{placeholder}} in the email', 'Config was {"debug": true}',
             'Regex: ^\\{[a-z]+\\}$', 'The answer is {42}']:
    CASES.append({"kind":"braces","user":f"Save this note exactly: {text}",
                  "calls":[{"name":"save_note","arguments":{"text":text}}]})
# object value
for key, val in [("retry",{"attempts":3,"backoff":"exponential"}),("theme",{"dark":True,"accent":"blue"}),
                 ("limits",{"rpm":600,"burst":50}),("features",{"beta":False,"flags":["a","b"]}),("cache",{"ttl":30,"size":1000})]:
    CASES.append({"kind":"object","user":f"Set config key {key} to the JSON object {json.dumps(val)}.",
                  "calls":[{"name":"set_config","arguments":{"key":key,"value":val}}]})
# no-arg tool
for q in ["What time is it right now (UTC)?","Tell me the current UTC time.","What's the time?","Current time please.","Give me the UTC time."]:
    CASES.append({"kind":"noargs","user":q,"calls":[{"name":"get_time","arguments":{}}]})
# parallel calls
for a, b in [("Oslo","Tokyo"),("Lima","Cairo"),("Berlin","Madrid"),("Toronto","Tbilisi"),("Paris","Rome"),("Seoul","Delhi")]:
    CASES.append({"kind":"parallel","user":f"What's the weather in {a} and in {b}?",
                  "calls":[{"name":"get_weather","arguments":{"city":a}},{"name":"get_weather","arguments":{"city":b}}]})
# mixed parallel: time + weather
for c in ["Oslo","Lima","Berlin"]:
    CASES.append({"kind":"parallel-mixed","user":f"What time is it, and what's the weather in {c}?",
                  "calls":[{"name":"get_time","arguments":{}},{"name":"get_weather","arguments":{"city":c}}]})

if __name__ == "__main__":
    print(len(CASES), "cases;", {k: sum(c["kind"]==k for c in CASES) for k in dict.fromkeys(c["kind"] for c in CASES)})
