import osmnx as ox
from .logger import timeit
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from typing import Dict, List, Tuple, Optional

# Initialize geolocator
geolocator = Nominatim(user_agent="maze_app")

# Global constants
MAX_DISTANCE_KM = 5.0  # Maximum distance for user points and amenity search
WALKING_SPEED_MPM = 83  # meters per minute (~5 km/h)

# OSM Tags for each feature type
FEATURE_TAGS = {
    "supermarket": {"shop": "supermarket"},
    "bakery": {"shop": "bakery"},
    "metro": {"railway": ["station", "subway_entrance"], "station": "subway"},
    "bus_stop": {"highway": "bus_stop"},
    "tram_stop": {"railway": "tram_stop"}
}


def time_to_radius(minutes: float) -> float:
    """Convert walking time (minutes) to radius (meters)."""
    return minutes * WALKING_SPEED_MPM


def radius_to_time(meters: float) -> float:
    """Convert radius (meters) to walking time (minutes)."""
    return meters / WALKING_SPEED_MPM

@timeit
def fetch_amenities(lat, lon, dist=2000, tags=None):
    """Fetch POIs from OSM within a distance (legacy single-type fetch)."""
    if not tags:
        return []
    
    try:
        gdf = ox.features_from_point((lat, lon), tags, dist=dist)
        if gdf.empty:
            return []
        
        points = []
        for geometry in gdf.geometry:
            if geometry.geom_type == 'Point':
                points.append([geometry.y, geometry.x])
            elif geometry.geom_type in ('Polygon', 'MultiPolygon'):
                points.append([geometry.centroid.y, geometry.centroid.x])
        return points
    except Exception as e:
        print(f"Error fetching OSM data: {e}")
        return []


@timeit
def fetch_all_amenities_batch(lat: float, lon: float, dist: int, 
                               feature_weights: Dict[str, int]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Fetch ALL amenities in a SINGLE Overpass query using combined tags.
    
    Args:
        lat, lon: Center coordinates
        dist: Search radius in meters
        feature_weights: Dict of feature_name -> weight (0 = skip)
    
    Returns:
        Dict mapping feature_name -> list of (lat, lon) points
    """
    # Build combined tags for active features
    combined_tags = {}
    active_features = []
    
    for feature, weight in feature_weights.items():
        if weight > 0 and feature in FEATURE_TAGS:
            active_features.append(feature)
            tags = FEATURE_TAGS[feature]
            for key, val in tags.items():
                if key in combined_tags:
                    # Merge values into list
                    existing = combined_tags[key]
                    if isinstance(existing, list):
                        if isinstance(val, list):
                            combined_tags[key] = existing + val
                        else:
                            combined_tags[key] = existing + [val]
                    else:
                        if isinstance(val, list):
                            combined_tags[key] = [existing] + val
                        else:
                            combined_tags[key] = [existing, val]
                else:
                    combined_tags[key] = val
    
    if not combined_tags:
        return {f: [] for f in feature_weights}
    
    # Single OSM query
    try:
        gdf = ox.features_from_point((lat, lon), combined_tags, dist=dist)
    except Exception as e:
        print(f"Error fetching OSM batch: {e}")
        return {f: [] for f in feature_weights}
    
    if gdf.empty:
        return {f: [] for f in feature_weights}
    
    # Categorize results by feature type
    results = {f: [] for f in feature_weights}
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Point':
            point = (geom.y, geom.x)
        elif geom.geom_type in ('Polygon', 'MultiPolygon'):
            point = (geom.centroid.y, geom.centroid.x)
        else:
            continue
        
        # Match to feature type based on tags
        if row.get('shop') == 'supermarket':
            results['supermarket'].append(point)
        elif row.get('shop') == 'bakery':
            results['bakery'].append(point)
        elif row.get('railway') in ('station', 'subway_entrance') or row.get('station') == 'subway':
            results['metro'].append(point)
        elif row.get('highway') == 'bus_stop':
            results['bus_stop'].append(point)
        elif row.get('railway') == 'tram_stop':
            results['tram_stop'].append(point)
    
    return results


def calculate_distance_km(coord1: tuple, coord2: tuple) -> float:
    """Calculate distance between two coordinates using geopy."""
    return geodesic(coord1, coord2).kilometers


def validate_points_distance(coords: list, max_km: float = MAX_DISTANCE_KM) -> tuple:
    """Validate that all points are within maximum distance of each other."""
    if len(coords) < 2:
        return True, ""
    
    for i, c1 in enumerate(coords):
        for j, c2 in enumerate(coords[i+1:], i+1):
            dist = calculate_distance_km(c1, c2)
            if dist > max_km:
                return False, f"Points {i+1} and {j+1} are {dist:.2f}km apart (max: {max_km}km)"
    
    return True, ""