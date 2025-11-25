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

def generate_map_html(location, weights, radius=3000):
    """Generate Folium map HTML with weighted heatmaps."""
    lat, lon = location
    m = folium.Map(location=[lat, lon], zoom_start=13)

    # Define tags for each feature
    feature_tags = {
        "supermarket": {"shop": "supermarket"},
        "bakery": {"shop": "bakery"},
        "metro": {"railway": "subway"},
        "bus_stop": {"highway": "bus_stop"},
        "tram_stop": {"railway": "tram_stop"},
    }
    
    # Calculate bounding box (OSM uses square/bbox, not circle)
    # Approximate: 1 degree lat/lon ≈ 111km
    lat_offset = radius / 111000  # degrees
    lon_offset = radius / (111000 * np.cos(np.radians(lat)))  # degrees, adjusted for latitude
    
    # Create grid for heatmap calculation
    grid_size = 50  # 50x50 grid
    lat_grid = np.linspace(lat - lat_offset, lat + lat_offset, grid_size)
    lon_grid = np.linspace(lon - lon_offset, lon + lon_offset, grid_size)
    
    # Store individual feature heatmaps
    feature_heatmaps = {}
    total_weight = sum(w for f, w in weights.items() if w > 0 and f != "rent")
    
    if total_weight == 0:
        # No features selected
        folium.Marker([lat, lon], popup="City Center", icon=folium.Icon(color='darkblue', icon='info-sign')).add_to(m)
        return m._repr_html_()
    
    # Fetch and create density maps for each feature
    for feature, weight in weights.items():
        if weight == 0 or feature == "rent":
            continue
            
        tags = feature_tags.get(feature)
        if tags:
            points = fetch_amenities(lat, lon, dist=radius, tags=tags)
            if points and len(points) > 0:
                # Create density grid for this feature
                points_array = np.array(points)
                lats = points_array[:, 0]
                lons = points_array[:, 1]
                
                # Create 2D histogram (density map)
                density, _, _ = np.histogram2d(
                    lats, lons,
                    bins=[lat_grid, lon_grid]
                )
                
                # Normalize density
                if density.max() > 0:
                    density = density / density.max()
                
                feature_heatmaps[feature] = density
    
    if not feature_heatmaps:
        # No data found
        folium.Marker([lat, lon], popup="City Center (No data found)", icon=folium.Icon(color='darkblue', icon='info-sign')).add_to(m)
    else:
        # Combine heatmaps using weighted average: (w1*h1 + w2*h2 + ...) / (w1+w2+...)
        combined_heatmap = np.zeros((grid_size-1, grid_size-1))
        for feature, density in feature_heatmaps.items():
            weight = weights[feature]
            combined_heatmap += density * weight
        
        combined_heatmap /= total_weight
        
        # Convert grid back to points for HeatMap visualization
        heatmap_points = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                if combined_heatmap[i, j] > 0.01:  # Only include points with significant density
                    lat_center = (lat_grid[i] + lat_grid[i+1]) / 2
                    lon_center = (lon_grid[j] + lon_grid[j+1]) / 2
                    intensity = combined_heatmap[i, j]
                    heatmap_points.append([lat_center, lon_center, intensity])
        
        if heatmap_points:
            HeatMap(
                heatmap_points, 
                radius=15, 
                blur=20,
                gradient={
                    0.0: 'blue',
                    0.3: 'cyan',
                    0.5: 'lime',
                    0.7: 'yellow',
                    1.0: 'red'
                }
            ).add_to(m)

    # Add a marker for the center
    folium.Marker([lat, lon], popup="City Center", icon=folium.Icon(color='darkblue', icon='info-sign')).add_to(m)
    
    # Add a RECTANGLE showing the search area (not circle, since OSM uses bounding box)
    bounds = [
        [lat - lat_offset, lon - lon_offset],  # Southwest corner
        [lat + lat_offset, lon + lon_offset]   # Northeast corner
    ]
    
    folium.Rectangle(
        bounds=bounds,
        color='blue',
        fill=False,
        weight=2,
        opacity=0.5,
        popup=f'Search area: {radius}m bounding box'
    ).add_to(m)

    return m._repr_html_()
