from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from .utils import MAX_DISTANCE_KM, time_to_radius, fetch_all_amenities_batch
from fastapi import APIRouter
from .logger import timeit, PerformanceTimer
from .cache import cache_geocode, set_cache_geocode, cache_amenities, set_cache_amenities
from concurrent.futures import ThreadPoolExecutor
import folium

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")


def geocode_single(address: str):
    """Geocode a single address with caching."""
    from geopy.geocoders import Nominatim
    
    if not address.strip():
        return None
    
    # Check cache first
    cached = cache_geocode(address)
    if cached:
        print(f"📦 Cache hit for: {address[:30]}...")
        return tuple(cached)
    
    # Geocode and cache
    try:
        geolocator = Nominatim(user_agent="maze_app")
        location = geolocator.geocode(address, timeout=10)
        if location:
            coords = (location.latitude, location.longitude)
            set_cache_geocode(address, coords)
            return coords
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
    return None


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/reverse_geocode")
@timeit
async def reverse_geocode(lat: float, lon: float):
    """Reverse geocode coordinates to address."""
    from geopy.geocoders import Nominatim
    try:
        geolocator = Nominatim(user_agent="maze_app")
        address = geolocator.reverse((lat, lon), timeout=10)
        if address:
            return {"address": address.address, "success": True}
    except Exception as e:
        print(f"Error reverse geocoding: {e}")
    return {"address": "Unknown location", "success": False}


@router.post("/generate_heatmap", response_class=HTMLResponse)
@timeit
async def generate_heatmap(
    request: Request,
    # User points (point1 is reference, required)
    point1_address: str = Form(...),
    point2_address: str = Form(""),
    point3_address: str = Form(""),
    # OSM feature weights (1-10)
    supermarket_weight: int = Form(5),
    bakery_weight: int = Form(5),
    metro_weight: int = Form(5),
    bus_weight: int = Form(5),
    tram_weight: int = Form(5),
    # Parameters
    radius: int = Form(2000),  # Search radius in meters
    feature_time: int = Form(6)  # Walking time in MINUTES
):
    from .utils import validate_points_distance
    from .heatmap_layers import HeatmapLayerGenerator
    
    # Convert walking time to radius
    feature_distance = time_to_radius(feature_time)
    
    # Parallel geocoding with caching
    addresses = [point1_address, point2_address, point3_address]
    
    with PerformanceTimer("Geocoding addresses (parallel + cached)"):
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(geocode_single, addresses))
    
    coords = [r for r in results if r is not None]
    
    if not coords:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Could not geocode any addresses. Please check your inputs."
        })
    
    # Validate distance
    is_valid, error_msg = validate_points_distance(coords, max_km=MAX_DISTANCE_KM)
    if not is_valid:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": error_msg
        })
    
    # Reference point is always the first
    ref_lat, ref_lon = coords[0]
    
    # Feature weights dict
    feature_weights = {
        "supermarket": supermarket_weight,
        "bakery": bakery_weight,
        "metro": metro_weight,
        "bus_stop": bus_weight,
        "tram_stop": tram_weight
    }
    
    # Build circles list
    circles = []
    
    # Check cache for amenities
    cached_amenities = cache_amenities(ref_lat, ref_lon, radius, feature_weights)
    
    if cached_amenities:
        print("📦 Cache hit for amenities")
        amenities_by_type = cached_amenities
    else:
        # BATCH fetch all amenities in ONE query
        with PerformanceTimer("Fetching OSM amenities (BATCH)"):
            amenities_by_type = fetch_all_amenities_batch(
                ref_lat, ref_lon, 
                dist=radius, 
                feature_weights=feature_weights
            )
        # Cache the result
        set_cache_amenities(ref_lat, ref_lon, radius, feature_weights, amenities_by_type)
    
    # Build circles from fetched amenities
    for feature, weight in feature_weights.items():
        if weight > 0:
            points = amenities_by_type.get(feature, [])
            for lat, lon in points:
                circles.append({
                    "lat": lat,
                    "lon": lon,
                    "radius": feature_distance,
                    "weight": weight
                })
    
    # Add user points (weight = 10, maximum)
    for lat, lon in coords:
        circles.append({
            "lat": lat,
            "lon": lon,
            "radius": feature_distance,
            "weight": 10
        })
    
    if not circles:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "No data found. Try adjusting your search radius or location."
        })
    
    # Generate heatmap
    try:
        with PerformanceTimer("Generating heatmap layers"):
            gen = HeatmapLayerGenerator(circles, output_dir="app/heatmap_output")
            result = gen.build(
                img_size='auto',
                max_zoom=15,
                colormaps={'weighted_avg': 'inferno'}
            )
        
        with PerformanceTimer("Creating Folium map"):
            m = gen.create_map(zoom=14, center=[ref_lat, ref_lon])
            
            # Add markers for user points
            for i, (lat, lon) in enumerate(coords):
                color = 'blue' if i == 0 else 'orange'
                label = 'Reference Point' if i == 0 else f'Point {i+1}'
                folium.Marker(
                    [lat, lon],
                    popup=label,
                    icon=folium.Icon(color=color, icon='star')
                ).add_to(m)
        
        map_html = m._repr_html_()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "map_html": map_html
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error generating heatmap: {str(e)}"
        })
