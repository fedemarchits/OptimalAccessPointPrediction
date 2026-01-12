import osmnx as ox
import json

def get_city_bbox_coords(city_name):
    gdf = ox.geocode_to_gdf(city_name)
    minx, miny, maxx, maxy = gdf.total_bounds
    return [
        [miny, minx],  # bottom-left (south, west)
        [miny, maxx],  # bottom-right (south, east)
        [maxy, maxx],  # top-right (north, east)
        [maxy, minx]   # top-left (north, west)
    ]

def generate_bboxes_for_cities(cities, output_file):
    results = {"cities": []}
    for city in cities:
        try:
            bbox = get_city_bbox_coords(city)
            results["cities"].append({
                "city": city,
                "bbox": bbox
            })
            print(f"✔ Processed {city}")
        except Exception as e:
            print(f"✘ Error processing {city}: {e}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved JSON to {output_file}")
    return results

if __name__ == "__main__":
    cities = [
        # Italy
        "Rome, Italy", "Milan, Italy", "Naples, Italy", "Turin, Italy",
        "Palermo, Italy", "Bologna, Italy", "Florence, Italy", "Bari, Italy", "Catania, Italy",
        # France
        "Paris, France", "Marseille, France", "Lyon, France", "Toulouse, France",
        "Nice, France", "Nantes, France", "Montpellier, France", "Strasbourg, France",
        "Bordeaux, France", "Lille, France",
        # Austria
        "Vienna, Austria", "Graz, Austria", "Linz, Austria", "Salzburg, Austria",
        "Innsbruck, Austria", "Klagenfurt, Austria",
        # Belgium
        "Brussels, Belgium", "Antwerp, Belgium", "Ghent, Belgium", "Charleroi, Belgium",
        "Liège, Belgium", "Bruges, Belgium", "Namur, Belgium",
        # Denmark
        "Copenhagen, Denmark",
        # Finland
        "Helsinki, Finland", "Espoo, Finland", "Tampere, Finland", "Vantaa, Finland",
        "Oulu, Finland", "Turku, Finland",
        # Greece
        "Thessaloniki, Greece", "Volos, Greece",
        # Netherlands
        "Amsterdam, Netherlands", "Rotterdam, Netherlands", "The Hague, Netherlands",
        "Utrecht, Netherlands", "Eindhoven, Netherlands", "Tilburg, Netherlands", "Groningen, Netherlands",
        # Norway
        "Oslo, Norway", "Bergen, Norway", "Trondheim, Norway", "Stavanger, Norway", "Tromsø, Norway",
        # Portugal
        "Lisbon, Portugal", "Porto, Portugal", "Vila Nova de Gaia, Portugal", "Amadora, Portugal",
        "Braga, Portugal", "Coimbra, Portugal",
        # Sweden
        "Stockholm, Sweden", "Malmö, Sweden", "Uppsala, Sweden",
        "Västerås, Sweden", "Örebro, Sweden", "Linköping, Sweden",
        # Switzerland
        "Zurich, Switzerland", "Geneva, Switzerland", "Basel, Switzerland", "Lausanne, Switzerland",
        "Bern, Switzerland", "Winterthur, Switzerland", "Lucerne, Switzerland",
        # United Kingdom
        "London, United Kingdom", "Birmingham, United Kingdom",
        "Leeds, United Kingdom", "Liverpool, United Kingdom", "Manchester, United Kingdom",
        "Bristol, United Kingdom", "Sheffield, United Kingdom", "Edinburgh, United Kingdom"
    ]

    generate_bboxes_for_cities(cities, "cities_bboxes_major_europe.json")
