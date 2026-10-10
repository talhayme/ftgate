"""Tool-calling cases: five tools, forty user requests, one right answer each.

Deterministic and small on purpose. The point is not to benchmark the model
but to compare the same model under two prompts, so what matters is that
every case has an unambiguous expected tool and expected required arguments.
"""
import json

TOOLS = [
  {"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.",
    "parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"},
      "unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}},
  {"type":"function","function":{"name":"convert_currency","description":"Convert an amount from one currency to another.",
    "parameters":{"type":"object","properties":{"amount":{"type":"number"},"from_currency":{"type":"string","description":"ISO code, e.g. USD"},
      "to_currency":{"type":"string","description":"ISO code, e.g. EUR"}},"required":["amount","from_currency","to_currency"]}}},
  {"type":"function","function":{"name":"set_timer","description":"Set a countdown timer.",
    "parameters":{"type":"object","properties":{"minutes":{"type":"integer"},"label":{"type":"string"}},"required":["minutes"]}}},
  {"type":"function","function":{"name":"search_contacts","description":"Find a contact by name.",
    "parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}},
  {"type":"function","function":{"name":"send_email","description":"Send an email.",
    "parameters":{"type":"object","properties":{"to":{"type":"string","description":"Recipient address"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject"]}}},
]

CITIES = ["Tbilisi","Berlin","Tokyo","Lima","Oslo","Cairo","Toronto","Madrid"]
CASES = []
for c in CITIES:
    CASES.append({"user": f"What's the weather like in {c} right now?", "tool": "get_weather", "args": {"city": c}})
for amt, f, t in [(100,"USD","EUR"),(250,"EUR","GBP"),(5000,"JPY","USD"),(80,"GBP","CHF"),(1200,"CAD","USD"),(300,"AUD","NZD"),(45,"USD","JPY"),(999,"CHF","EUR")]:
    CASES.append({"user": f"Convert {amt} {f} to {t}.", "tool": "convert_currency", "args": {"amount": amt, "from_currency": f, "to_currency": t}})
for m in [5, 10, 25, 45, 90, 3, 15, 60]:
    CASES.append({"user": f"Set a timer for {m} minutes.", "tool": "set_timer", "args": {"minutes": m}})
for n in ["Anna Petrova","John Smith","Li Wei","Maria Garcia","Omar Haddad","Sofia Rossi","Kenji Sato","Emma Brown"]:
    CASES.append({"user": f"Look up {n} in my contacts.", "tool": "search_contacts", "args": {"name": n}})
for to, subj in [("alice@example.com","Lunch tomorrow"),("bob@corp.io","Q3 report"),("team@startup.dev","Standup moved"),("dr.lee@clinic.org","Appointment"),
                 ("sam@example.com","Tickets"),("hr@company.com","Vacation request"),("nina@mail.net","Photos"),("ops@service.io","Outage")]:
    CASES.append({"user": f"Send an email to {to} with the subject \"{subj}\".", "tool": "send_email", "args": {"to": to, "subject": subj}})

if __name__ == "__main__":
    print(json.dumps({"tools": TOOLS, "cases": CASES}, indent=1))
