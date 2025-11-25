import folium
from folium.plugins import HeatMap
import osmnx as ox
from geopy.geocoders import Nominatim
import numpy as np

# Initialize geolocator
geolocator = Nominatim(user_agent="maze_app")

def get_city_coordinates(city_name):
    """Geocode a city name to (lat, lon) with timeout."""
    try:
        location = geolocator.geocode(city_name, timeout=5)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"Error geocoding {city_name}: {e}")
    return None

def search_cities(query):
    """Search for cities matching the query string."""
    try:
        # Use Nominatim search with city/town filter
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="maze_app")
        results = geolocator.geocode(query, exactly_one=False, limit=10, timeout=5, addressdetails=True)
        if results:
            cities = []
            for result in results:
                # Get the display name and extract city name
                display_name = result.address
                
                # Try to get a short name
                if result.raw.get('address'):
                    addr = result.raw['address']
                    city_name = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('county') or addr.get('state') or display_name.split(',')[0]
                else:
                    city_name = display_name.split(',')[0]
                
                cities.append({"name": city_name, "display": display_name})
            return cities[:10]
    except Exception as e:
        print(f"Error searching cities: {e}")
    return []

def get_rent_data(city_name):
    """Placeholder for rent data fetching."""
    # Future implementation: Fetch or load rent data
    print(f"Fetching rent data for {city_name} (Placeholder)")
    return []

def fetch_amenities(lat, lon, dist=2000, tags=None):
    """Fetch POIs from OSM within a distance."""
    if not tags:
        return []
    
    try:
        # Fetch geometries
        gdf = ox.features_from_point((lat, lon), tags, dist=dist)
        if gdf.empty:
            return []
        
        # Extract centroids for heatmap
        points = []
        for geometry in gdf.geometry:
            if geometry.geom_type == 'Point':
                points.append([geometry.y, geometry.x])
            elif geometry.geom_type == 'Polygon' or geometry.geom_type == 'MultiPolygon':
                points.append([geometry.centroid.y, geometry.centroid.x])
        return points
    except Exception as e:
        print(f"Error fetching OSM data: {e}")
        return []

def generate_map_html(location, weights):
    """Generate Folium map HTML with weighted heatmaps."""
    lat, lon = location
    m = folium.Map(location=[lat, lon], zoom_start=13)

    # Define tags for each feature
    feature_tags = {
        "supermarket": {"shop": "supermarket"},
        "bakery": {"shop": "bakery"},
        "metro": {"station": "subway"}, # OSM tag for metro stations often station=subway or railway=subway
        "bus_stop": {"highway": "bus_stop"},
        "tram_stop": {"railway": "tram_stop"},
    }
    
    # Adjust metro tags for better coverage (e.g., railway=station + station=subway)
    # For simplicity, we'll use a broader query or specific ones.
    # Let's refine tags:
    feature_tags["metro"] = {"railway": "subway"} 

    layers = []
    
    # Fetch and create layers
    for feature, weight in weights.items():
        if weight == 0:
            continue
            
        if feature == "rent":
            # Placeholder for rent layer
            continue
            
        tags = feature_tags.get(feature)
        if tags:
            points = fetch_amenities(lat, lon, dist=3000, tags=tags)
            if points:
                # Add a HeatMap layer for this feature
                # We weight the intensity by the user's preference
                # Note: Folium HeatMap takes (lat, lon, weight). 
                # We can scale the 'weight' of each point by the user's preference.
                weighted_points = [[p[0], p[1], weight/10.0] for p in points]
                
                # We can add individual heatmaps or one combined one.
                # User asked for "IoU" / "Optimal location". 
                # A combined heatmap summing up scores is best.
                layers.extend(weighted_points)

    if layers:
        HeatMap(layers, radius=15, blur=20).add_to(m)

    # Add a marker for the center
    folium.Marker([lat, lon], popup="City Center").add_to(m)

    return m._repr_html_()
