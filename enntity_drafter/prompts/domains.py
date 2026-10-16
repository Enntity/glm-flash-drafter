"""Enntity-authored synthetic domain library for the structured-coverage prompt set.

Every name, record, document and code file built from this module is generated
here from short vocabularies written for this project (no third-party text).
Field DSL per entity: "name:kind" or "name:kind(arg)", comma separated.
Kinds: id, name, email, phone, company, city, country, url, date, datetime,
int(lo,hi), num(lo,hi), bool, enum(a|b|c), str(vocab), list(vocab), text(vocab),
obj(entity), objlist(entity).
"""
from __future__ import annotations

import json
import random
import re
import string

FIRST = ("Amara Bruno Chen Dalia Emeka Farah Gideon Hana Ivo Jun Kalani Lucia Mateo Nadia Omar Priya Quinn Rafael "
         "Sana Tobias Uma Viktor Wen Ximena Yusuf Zofia Aiden Beatriz Callum Dmitri Elif Felix Greta Hiroshi Ines "
         "Jonas Keira Lars Mira Nikhil Olga Pavel Rosa Soren Talia Valentina Wei Yara Zane").split()
LAST = ("Abara Bergstrom Castillo Dubois Eze Fontaine Gallagher Haddad Ishikawa Janssen Kowalski Lindqvist Moreau "
        "Nakamura Okafor Petrov Quintero Rahman Silva Tanaka Udeh Varga Whitfield Xu Yilmaz Zielinski Arango Brandt "
        "Cohen Delgado Engstrom Farouk Grant Holm Iqbal Jaramillo Kim Larsen Mendes Novak Ortiz Park Reyes Sato "
        "Takahashi Vance Weber").split()
SYL = "vel brook ster quin mar tor lumen kade ash fen hollow crest north pine ridge sol vant orra bex cal dun".split()
CO_SUFFIX = ["Labs", "Logistics", "Systems", "Foods", "Health", "Analytics", "Works", "Partners", "Cooperative",
             "Outfitters", "Energy", "Studios", "Freight", "Clinic", "Bank", "Robotics"]
CITIES = ("Lisbon Nairobi Osaka Toronto Montevideo Krakow Hanoi Austin Lyon Accra Bergen Pune Valparaiso Tbilisi Cork "
          "Busan Denver Seville Leipzig Quito Perth Halifax Tallinn Marrakesh Cebu Rotterdam Calgary Bologna Durban "
          "Taipei").split()
COUNTRIES = ("PT KE JP CA UY PL VN US FR GH NO IN CL GE IE KR ES DE EC AU EE MA PH NL IT ZA TW").split()

VOCAB = {
    "product": ["trail running shoe", "ceramic pour-over set", "merino base layer", "bamboo cutting board",
                "noise-cancelling headphones", "standing desk frame", "cast iron skillet", "LED desk lamp",
                "waterproof backpack", "electric kettle", "yoga mat", "mechanical keyboard", "linen duvet cover",
                "insulated water bottle", "wireless charger", "hiking poles", "espresso grinder", "rain jacket"],
    "sku": ["SKU-%04d" % i for i in range(1000, 1060)],
    "carrier": ["Parcelway", "SwiftRoute", "BlueLine Freight", "Courierly", "Northstar Express"],
    "symptom": ["persistent cough", "mild fever", "lower back pain", "seasonal allergies", "migraine", "sprained ankle",
                "skin rash", "fatigue", "sore throat", "follow-up checkup"],
    "department": ["cardiology", "dermatology", "orthopedics", "pediatrics", "general practice", "neurology",
                   "radiology", "physiotherapy"],
    "course": ["Linear Algebra", "Organic Chemistry", "World History", "Intro to Programming", "Microeconomics",
               "Creative Writing", "Statistics", "Marine Biology", "Music Theory", "Data Structures"],
    "skill": ["Python", "SQL", "Kubernetes", "React", "Go", "negotiation", "Figma", "Rust", "data analysis",
              "public speaking", "Terraform", "TypeScript", "customer support", "Spanish"],
    "role": ["backend engineer", "product designer", "data analyst", "support lead", "site reliability engineer",
             "account executive", "QA engineer", "engineering manager", "technical writer"],
    "sensor_kind": ["temperature", "humidity", "pressure", "co2", "vibration", "flow_rate", "voltage", "lux"],
    "genre": ["synthwave", "jazz", "folk", "drum and bass", "ambient", "indie rock", "afrobeat", "classical", "lo-fi"],
    "cuisine": ["Ethiopian", "Peruvian", "Korean", "Lebanese", "Neapolitan", "Oaxacan", "Sichuan", "Georgian", "Thai"],
    "ingredient": ["chickpeas", "smoked paprika", "basmati rice", "coconut milk", "feta", "shallots", "ginger",
                   "black beans", "miso", "lemon zest", "tahini", "spinach", "farro", "cilantro"],
    "amenity": ["balcony", "parking", "dishwasher", "in-unit laundry", "gym", "elevator", "garden", "pet friendly"],
    "tag": ["urgent", "billing", "bug", "feature-request", "onboarding", "security", "performance", "docs", "ux"],
    "service": ["auth-service", "billing-api", "search-indexer", "notification-worker", "image-resizer",
                "gateway", "ledger-service", "report-builder", "session-store", "feature-flags"],
    "region": ["us-east", "us-west", "eu-central", "eu-west", "ap-south", "ap-northeast", "sa-east"],
    "crop": ["wheat", "barley", "soybean", "maize", "canola", "lentils", "potatoes", "sunflower"],
    "exercise": ["deadlift", "rowing intervals", "tempo run", "kettlebell swing", "pull-ups", "cycling",
                 "swim drills", "box jumps", "plank series"],
    "team": ["Harbor Hawks", "Ridge Runners", "Copper Foxes", "Valley Comets", "Iron Owls", "Delta Lynx",
             "Summit Bears", "Coastal Rays"],
    "venue": ["Riverside Hall", "The Foundry", "Granite Arena", "Lantern Theater", "Pier 9 Pavilion"],
    "body": ["Mars", "Jupiter", "Saturn", "Andromeda Galaxy", "Orion Nebula", "Pleiades", "Vega", "Ceres"],
    "instrument": ["refractor 102mm", "reflector 200mm", "binoculars 10x50", "Dobsonian 300mm"],
    "plan": ["free", "starter", "pro", "team", "enterprise"],
    "airport": ["LIS", "NBO", "KIX", "YYZ", "MVD", "KRK", "HAN", "AUS", "LYS", "ACC", "BGO", "PNQ"],
    "case_type": ["contract dispute", "trademark filing", "lease review", "employment claim", "estate planning"],
    "section": ["politics", "science", "culture", "sports", "business", "technology", "opinion"],
    "vehicle": ["box truck", "cargo van", "refrigerated trailer", "flatbed", "electric van"],
    "fuel": ["diesel", "electric", "hybrid", "CNG"],
    "game_item": ["ember sword", "frost shield", "healing tonic", "shadow cloak", "storm bow", "iron helm"],
    "book": ["The Quiet Orchard", "Maps of Salt", "A Ledger of Small Hours", "Glass Rivers", "The Ninth Lantern",
             "Notes from the Fen", "Copper Moon", "The Cartographer's Daughter"],
    "note": ["customer asked for a callback", "requires manager approval", "duplicate of an earlier request",
             "waiting on supplier confirmation", "flagged by the fraud check", "resolved after a config change",
             "needs translation before release", "escalated to tier two", "scheduled for the next maintenance window"],
}

# domain -> (label, {entity: field DSL})
DOMAINS = {
    "ecommerce": ("an online outdoor-gear store", {
        "order": "order_id:id, customer_name:name, customer_email:email, status:enum(pending|paid|shipped|delivered|refunded), "
                 "total:num(8,1800), currency:enum(USD|EUR|GBP), placed_at:datetime, gift_wrap:bool, lines:objlist(order_line)",
        "order_line": "sku:str(sku), product:str(product), quantity:int(1,6), unit_price:num(4,400)",
        "product": "sku:str(sku), title:str(product), price:num(4,400), stock:int(0,900), "
                   "category:enum(footwear|kitchen|apparel|electronics|outdoor|home), rating:num(1,5), tags:list(tag)",
    }),
    "clinic": ("a multi-specialty outpatient clinic (synthetic records only)", {
        "appointment": "appointment_id:id, patient_name:name, department:str(department), reason:str(symptom), "
                       "scheduled_for:datetime, duration_minutes:int(10,90), status:enum(booked|checked_in|completed|no_show|cancelled), telehealth:bool",
        "practitioner": "practitioner_id:id, full_name:name, department:str(department), years_experience:int(1,35), "
                        "accepting_new_patients:bool, languages:list(skill)",
    }),
    "logistics": ("a regional freight and parcel carrier", {
        "shipment": "tracking_number:id, origin_city:city, destination_city:city, carrier:str(carrier), weight_kg:num(0.2,1800), "
                    "status:enum(label_created|in_transit|out_for_delivery|delivered|exception), eta:date, signature_required:bool",
        "vehicle": "vehicle_id:id, kind:str(vehicle), fuel:str(fuel), capacity_kg:int(500,24000), odometer_km:int(1000,480000), "
                   "next_service:date, assigned_driver:name",
    }),
    "banking": ("a fictional retail bank's back office (synthetic data)", {
        "transaction": "transaction_id:id, account_holder:name, amount:num(-5000,9000), currency:enum(USD|EUR|JPY|CAD), "
                       "kind:enum(card|transfer|direct_debit|atm|fee|interest), booked_at:datetime, merchant:company, flagged:bool",
        "loan_application": "application_id:id, applicant:name, amount_requested:int(1000,450000), term_months:int(6,360), "
                            "purpose:enum(home|auto|education|business|personal), credit_score:int(420,850), decision:enum(approved|declined|manual_review)",
    }),
    "education": ("a university course-registration system", {
        "enrollment": "student_id:id, student_name:name, course:str(course), term:enum(fall|spring|summer), credits:int(1,6), "
                      "grade:enum(A|B|C|D|F|incomplete|in_progress), enrolled_on:date",
        "course_section": "section_id:id, course:str(course), instructor:name, room:str(venue), capacity:int(12,300), "
                          "enrolled:int(0,300), meets:list(tag)",
    }),
    "hr": ("a mid-size company's people-operations team", {
        "employee": "employee_id:id, full_name:name, email:email, role:str(role), team:enum(platform|growth|support|finance|design|data), "
                    "start_date:date, salary_band:enum(L1|L2|L3|L4|L5), remote:bool, skills:list(skill)",
        "leave_request": "request_id:id, employee:name, kind:enum(vacation|sick|parental|unpaid|bereavement), start:date, end:date, "
                         "days:int(1,40), approved:bool",
    }),
    "iot": ("an industrial IoT monitoring platform", {
        "reading": "sensor_id:id, kind:str(sensor_kind), value:num(-40,1200), unit:enum(C|%|kPa|ppm|mm_s|L_min|V|lx), "
                   "recorded_at:datetime, site:city, anomaly:bool",
        "device": "device_id:id, model:enum(TX-100|TX-220|HX-9|Q-Mesh|Nimbus-4), firmware:enum(1.4.2|1.5.0|2.0.1|2.1.0), "
                  "site:city, battery_pct:int(0,100), last_seen:datetime, online:bool",
    }),
    "music": ("a music-streaming service", {
        "track": "track_id:id, title:str(book), artist:name, genre:str(genre), duration_seconds:int(95,620), plays:int(0,9000000), "
                 "explicit:bool, released:date",
        "playlist": "playlist_id:id, owner:name, name:str(genre), followers:int(0,250000), public:bool, tracks:objlist(track)",
    }),
    "restaurant": ("a restaurant group's reservation and menu system", {
        "reservation": "reservation_id:id, guest_name:name, party_size:int(1,14), time:datetime, location:city, "
                       "status:enum(confirmed|seated|completed|no_show|cancelled), notes:text(note)",
        "menu_item": "item_id:id, name:str(ingredient), cuisine:str(cuisine), price:num(3,58), vegetarian:bool, "
                     "allergens:list(ingredient), calories:int(80,1600)",
    }),
    "realestate": ("a property-listing marketplace", {
        "listing": "listing_id:id, city:city, bedrooms:int(0,6), bathrooms:num(1,4), area_m2:int(18,420), rent_per_month:int(400,9000), "
                   "available_from:date, amenities:list(amenity), status:enum(active|pending|leased|withdrawn)",
        "viewing": "viewing_id:id, listing_id:id, visitor:name, phone:phone, scheduled_for:datetime, outcome:enum(scheduled|attended|missed|applied)",
    }),
    "support": ("a SaaS company's customer-support desk", {
        "ticket": "ticket_id:id, customer:company, reporter:name, subject:text(note), priority:enum(low|normal|high|urgent), "
                  "status:enum(open|pending|solved|closed), tags:list(tag), opened_at:datetime, plan:str(plan)",
        "agent": "agent_id:id, name:name, team:enum(tier1|tier2|billing|enterprise), open_tickets:int(0,60), csat:num(1,5), on_shift:bool",
    }),
    "infra": ("a cloud platform team running microservices", {
        "deployment": "deploy_id:id, service:str(service), version:enum(v1.4.0|v1.4.1|v2.0.0|v2.1.3|v3.0.0-rc1), region:str(region), "
                      "replicas:int(1,48), status:enum(pending|rolling|healthy|degraded|rolled_back), started_at:datetime, canary:bool",
        "incident": "incident_id:id, service:str(service), severity:enum(sev1|sev2|sev3|sev4), opened_at:datetime, "
                    "commander:name, customer_impact:bool, summary:text(note)",
    }),
    "ticketing": ("a live-event ticketing platform", {
        "event": "event_id:id, title:str(book), venue:str(venue), city:city, starts_at:datetime, capacity:int(150,25000), "
                 "sold:int(0,25000), category:enum(concert|theater|sports|conference|comedy)",
        "ticket": "ticket_id:id, event_id:id, holder:name, seat:enum(GA|A12|B4|C19|D7|VIP1|BAL3), price:num(12,640), scanned:bool",
    }),
    "agriculture": ("a farm-management cooperative", {
        "field_plot": "plot_id:id, crop:str(crop), hectares:num(0.5,320), planted_on:date, expected_yield_t:num(1,1900), "
                      "irrigated:bool, soil:enum(loam|clay|sandy|silt|peat)",
        "harvest": "harvest_id:id, plot_id:id, crop:str(crop), harvested_on:date, tonnes:num(0.5,1800), moisture_pct:num(8,30), grade:enum(A|B|C|feed)",
    }),
    "energy": ("a utility operating a regional electricity grid", {
        "meter_reading": "meter_id:id, customer:name, kwh:num(0,2400), read_at:datetime, tariff:enum(flat|time_of_use|ev|solar_export), estimated:bool",
        "outage": "outage_id:id, region:str(region), started_at:datetime, customers_affected:int(1,90000), cause:enum(storm|equipment|planned|vegetation|unknown), restored:bool",
    }),
    "fitness": ("a fitness-coaching app", {
        "workout": "workout_id:id, athlete:name, exercise:str(exercise), sets:int(1,8), reps:int(1,30), load_kg:num(0,220), "
                   "rpe:int(1,10), logged_on:date",
        "member": "member_id:id, name:name, plan:str(plan), joined:date, goal:enum(strength|endurance|mobility|weight_loss|rehab), sessions_per_week:int(1,7)",
    }),
    "legal": ("a small law firm's matter-tracking system (synthetic)", {
        "matter": "matter_id:id, client:company, case_type:str(case_type), lead_attorney:name, opened_on:date, "
                  "billable_hours:num(0,900), status:enum(intake|active|on_hold|closed), confidential:bool",
        "time_entry": "entry_id:id, matter_id:id, attorney:name, hours:num(0.1,12), activity:enum(research|drafting|meeting|court|review), worked_on:date",
    }),
    "news": ("a digital newsroom CMS", {
        "article": "article_id:id, headline:str(book), author:name, section:str(section), word_count:int(150,4200), "
                   "published_at:datetime, status:enum(draft|editing|scheduled|published|retracted), tags:list(section)",
        "comment": "comment_id:id, article_id:id, author:name, posted_at:datetime, likes:int(0,4000), flagged:bool",
    }),
    "gaming": ("an online multiplayer game backend", {
        "player": "player_id:id, handle:name, level:int(1,120), guild:str(team), region:str(region), last_login:datetime, "
                  "banned:bool, inventory:list(game_item)",
        "match": "match_id:id, mode:enum(duel|squad|raid|ranked|casual), map:enum(Ashfall|Tidewater|Glimmerdeep|Old Quarry), "
                 "duration_seconds:int(120,3600), winner:str(team), started_at:datetime",
    }),
    "travel": ("a corporate travel-booking tool", {
        "booking": "booking_id:id, traveler:name, origin:str(airport), destination:str(airport), depart:date, return_date:date, "
                   "fare_class:enum(economy|premium|business), cost:num(80,6400), refundable:bool",
        "hotel_stay": "stay_id:id, traveler:name, city:city, check_in:date, nights:int(1,21), nightly_rate:num(45,690), loyalty_number:id",
    }),
    "library": ("a public library network", {
        "loan": "loan_id:id, member:name, title:str(book), branch:city, borrowed_on:date, due_on:date, renewed:int(0,3), overdue:bool",
        "member": "card_number:id, name:name, email:email, joined:date, fines_owed:num(0,45), home_branch:city",
    }),
    "astronomy": ("an amateur astronomy club's observation log", {
        "observation": "observation_id:id, observer:name, target:str(body), instrument:str(instrument), observed_at:datetime, "
                       "seeing:int(1,5), magnitude:num(-4,14), notes:text(note)",
    }),
    "fleet": ("a city bike-share operator", {
        "trip": "trip_id:id, bike_id:id, rider:name, start_station:city, end_station:city, started_at:datetime, minutes:int(1,240), ebike:bool, fare:num(0,24)",
        "station": "station_id:id, name:city, docks:int(8,60), bikes_available:int(0,60), lat:num(-60,60), lon:num(-170,170), active:bool",
    }),
    "recipes": ("a recipe and meal-planning app", {
        "recipe": "recipe_id:id, title:str(cuisine), cuisine:str(cuisine), servings:int(1,12), prep_minutes:int(5,240), "
                  "ingredients:list(ingredient), vegan:bool, difficulty:enum(easy|medium|hard)",
    }),
}


class Field:
    __slots__ = ("name", "kind", "arg")

    def __init__(self, name, kind, arg):
        self.name, self.kind, self.arg = name, kind, arg


_FIELD = re.compile(r"\s*([a-z_0-9]+):([a-z]+)(?:\(([^)]*)\))?\s*")


def fields(domain: str, entity: str) -> list[Field]:
    out = []
    for part in DOMAINS[domain][1][entity].split(", "):
        m = _FIELD.fullmatch(part)
        assert m, (domain, entity, part)
        out.append(Field(m.group(1), m.group(2), m.group(3)))
    return out


def entities(domain: str) -> list[str]:
    return list(DOMAINS[domain][1])


def top_entities(domain: str) -> list[str]:
    """Entities not only used nested."""
    nested = {f.arg for e in entities(domain) for f in fields(domain, e) if f.kind in ("obj", "objlist")}
    return [e for e in entities(domain) if e not in nested]


def plural(e: str) -> str:
    if e.endswith("y") and not e.endswith("ay"):
        return e[:-1] + "ies"
    if e.endswith(("s", "ch")):
        return e + "es"
    return e + "s"


def title(e: str) -> str:
    return "".join(p.capitalize() for p in e.split("_"))


def human(name: str) -> str:
    return name.replace("_", " ")


# ---------------------------------------------------------------- values

def company(rng):
    return (rng.choice(SYL) + rng.choice(SYL)).capitalize() + " " + rng.choice(CO_SUFFIX)


def person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def value(rng: random.Random, domain: str, f: Field, entity: str = "x", depth: int = 0):
    k, a = f.kind, f.arg
    if k == "id":
        pre = re.sub(r"[^A-Z]", "", f.name.upper().replace("_ID", "").replace("_NUMBER", ""))[:3] or entity[:3].upper()
        return f"{pre}-{rng.randint(10000, 99999)}"
    if k == "name":
        return person(rng)
    if k == "email":
        return f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}@{rng.choice(SYL)}{rng.choice(SYL)}.example"
    if k == "phone":
        return f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    if k == "company":
        return company(rng)
    if k == "city":
        return rng.choice(CITIES)
    if k == "country":
        return rng.choice(COUNTRIES)
    if k == "url":
        return f"https://{rng.choice(SYL)}{rng.choice(SYL)}.example/{rng.choice(SYL)}/{rng.randint(1, 999)}"
    if k == "date":
        return f"{rng.choice((2025, 2026))}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if k == "datetime":
        return (f"{rng.choice((2025, 2026))}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
                f"T{rng.randint(0, 23):02d}:{rng.choice((0, 15, 30, 45)):02d}:00Z")
    if k == "int":
        lo, hi = map(int, a.split(","))
        return rng.randint(lo, hi)
    if k == "num":
        lo, hi = map(float, a.split(","))
        return round(rng.uniform(lo, hi), 2)
    if k == "bool":
        return rng.random() < 0.4
    if k == "enum":
        return rng.choice(a.split("|"))
    if k in ("str", "text"):
        return rng.choice(VOCAB[a])
    if k == "list":
        return rng.sample(VOCAB[a], rng.randint(1, min(4, len(VOCAB[a]))))
    if k == "obj":
        return record(rng, domain, a, depth + 1)
    if k == "objlist":
        return [record(rng, domain, a, depth + 1) for _ in range(rng.randint(1, 4 if depth == 0 else 2))]
    raise ValueError(k)


def record(rng, domain, entity, depth=0):
    return {f.name: value(rng, domain, f, entity, depth) for f in fields(domain, entity)}


def flat_fields(domain, entity):
    return [f for f in fields(domain, entity) if f.kind not in ("obj", "objlist")]


# ---------------------------------------------------------------- schema

def field_schema(domain, f: Field, describe: bool):
    k, a = f.kind, f.arg
    if k in ("id", "name", "email", "phone", "company", "city", "country", "url", "str", "text"):
        s = {"type": "string"}
    elif k == "date":
        s = {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD)"} if describe else {"type": "string"}
    elif k == "datetime":
        s = {"type": "string", "description": "ISO 8601 UTC timestamp"} if describe else {"type": "string"}
    elif k == "int":
        lo, hi = map(int, a.split(","))
        s = {"type": "integer", "minimum": lo, "maximum": hi}
    elif k == "num":
        s = {"type": "number"}
    elif k == "bool":
        s = {"type": "boolean"}
    elif k == "enum":
        s = {"type": "string", "enum": a.split("|")}
    elif k == "list":
        s = {"type": "array", "items": {"type": "string"}}
    elif k == "obj":
        s = entity_schema(domain, a, describe)
    elif k == "objlist":
        s = {"type": "array", "items": entity_schema(domain, a, describe)}
    else:
        raise ValueError(k)
    return s


def entity_schema(domain, entity, describe=False, subset=None):
    fs = [f for f in fields(domain, entity) if subset is None or f.name in subset]
    return {"type": "object", "properties": {f.name: field_schema(domain, f, describe) for f in fs},
            "required": [f.name for f in fs], "additionalProperties": False}


def describe_fields(domain, entity, subset=None) -> str:
    """Prose field list (for prompts that don't paste the schema)."""
    out = []
    for f in fields(domain, entity):
        if subset is not None and f.name not in subset:
            continue
        k, a = f.kind, f.arg
        d = {"int": "integer", "num": "number", "bool": "boolean", "list": "array of strings",
             "date": "date string YYYY-MM-DD", "datetime": "ISO 8601 UTC timestamp"}.get(k, "string")
        if k == "enum":
            d = "one of " + ", ".join(a.split("|"))
        elif k == "int":
            d = f"integer between {a.replace(',', ' and ')}"
        elif k in ("obj", "objlist"):
            d = ("object" if k == "obj" else "array of objects") + " with " + describe_fields(domain, a)
        out.append(f"`{f.name}` ({d})")
    return "; ".join(out)


def ident(rng, n=8):
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def dumps(x, rng=None, compact=None):
    if compact is None:
        compact = rng is not None and rng.random() < 0.3
    return json.dumps(x, separators=(",", ":")) if compact else json.dumps(x, indent=2)
