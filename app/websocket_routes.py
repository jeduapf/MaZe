"""
WebSocket endpoint for progressive heatmap loading.
Sends updates to the client as data is fetched and processed.
"""

from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .utils import time_to_radius, fetch_all_amenities_batch, validate_points_distance, MAX_DISTANCE_KM
from .cache import cache_geocode, set_cache_geocode, cache_amenities, set_cache_amenities
from .heatmap_layers import HeatmapLayerGenerator
from .logger import log_timing

router = APIRouter()


def geocode_single(address: str):
    """Geocode a single address with caching."""
    from geopy.geocoders import Nominatim
    
    if not address or not address.strip():
        return None
    
    cached = cache_geocode(address)
    if cached:
        return tuple(cached)
    
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


async def send_status(websocket: WebSocket, status: str, progress: int, data: dict = None):
    """Send a status update to the client."""
    message = {"status": status, "progress": progress}
    if data:
        message["data"] = data
    await websocket.send_json(message)


@router.websocket("/ws/heatmap")
async def heatmap_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for progressive heatmap generation.
    
    Client sends:
    {
        "addresses": ["address1", "address2", "address3"],
        "weights": {"supermarket": 5, "bakery": 5, "metro": 5, "bus_stop": 5, "tram_stop": 5},
        "radius": 2000,
        "feature_time": 6
    }
    
    Server responds with progress updates:
    {"status": "geocoding", "progress": 10}
    {"status": "fetching", "progress": 30}
    {"status": "generating", "progress": 60}
    {"status": "complete", "progress": 100, "data": {"map_html": "..."}}
    """
    await websocket.accept()
    
    try:
        # Wait for request from client
        data = await websocket.receive_json()
        
        addresses = data.get("addresses", [])
        weights = data.get("weights", {})
        radius = data.get("radius", 2000)
        feature_time = data.get("feature_time", 6)
        
        feature_distance = time_to_radius(feature_time)
        
        # Step 1: Geocoding (parallel)
        await send_status(websocket, "geocoding", 10)
        log_timing("▶️  [WS] Starting geocoding...")
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = await loop.run_in_executor(
                None, 
                lambda: list(executor.map(geocode_single, addresses))
            )
        
        coords = [r for r in results if r is not None]
        
        if not coords:
            await send_status(websocket, "error", 0, {"message": "Could not geocode any addresses"})
            return
        
        # Validate distance
        is_valid, error_msg = validate_points_distance(coords, max_km=MAX_DISTANCE_KM)
        if not is_valid:
            await send_status(websocket, "error", 0, {"message": error_msg})
            return
        
        ref_lat, ref_lon = coords[0]
        await send_status(websocket, "geocoded", 25, {"center": [ref_lat, ref_lon]})
        log_timing(f"✅ [WS] Geocoding complete: {len(coords)} points")
        
        # Step 2: Fetch amenities
        await send_status(websocket, "fetching", 30)
        log_timing("▶️  [WS] Fetching amenities...")
        
        feature_weights = {
            "supermarket": weights.get("supermarket", 5),
            "bakery": weights.get("bakery", 5),
            "metro": weights.get("metro", 5),
            "bus_stop": weights.get("bus_stop", 5),
            "tram_stop": weights.get("tram_stop", 5)
        }
        
        # Check cache
        cached_amenities = cache_amenities(ref_lat, ref_lon, radius, feature_weights)
        
        if cached_amenities:
            amenities_by_type = cached_amenities
            log_timing("📦 [WS] Cache hit for amenities")
        else:
            amenities_by_type = await loop.run_in_executor(
                None,
                lambda: fetch_all_amenities_batch(ref_lat, ref_lon, radius, feature_weights)
            )
            set_cache_amenities(ref_lat, ref_lon, radius, feature_weights, amenities_by_type)
        
        await send_status(websocket, "fetched", 50)
        log_timing(f"✅ [WS] Fetched amenities")
        
        # Step 3: Build circles
        circles = []
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
        
        # Add user points
        for lat, lon in coords:
            circles.append({
                "lat": lat,
                "lon": lon,
                "radius": feature_distance,
                "weight": 10
            })
        
        if not circles:
            await send_status(websocket, "error", 0, {"message": "No data found"})
            return
        
        # Step 4: Generate heatmap
        await send_status(websocket, "generating", 60)
        log_timing("▶️  [WS] Generating heatmap...")
        
        def generate_heatmap():
            gen = HeatmapLayerGenerator(circles, output_dir="app/heatmap_output")
            result = gen.build(
                img_size='auto',
                max_zoom=15,
                colormaps={'weighted_avg': 'inferno'},
                use_multicore=True
            )
            
            import folium
            m = gen.create_map(zoom=14, center=[ref_lat, ref_lon])
            
            for i, (lat, lon) in enumerate(coords):
                color = 'blue' if i == 0 else 'orange'
                label = 'Reference Point' if i == 0 else f'Point {i+1}'
                folium.Marker(
                    [lat, lon],
                    popup=label,
                    icon=folium.Icon(color=color, icon='star')
                ).add_to(m)
            
            return m._repr_html_()
        
        map_html = await loop.run_in_executor(None, generate_heatmap)
        
        await send_status(websocket, "complete", 100, {"map_html": map_html})
        log_timing("✅ [WS] Heatmap generation complete!")
        
    except WebSocketDisconnect:
        log_timing("⚠️  [WS] Client disconnected")
    except Exception as e:
        log_timing(f"❌ [WS] Error: {e}")
        try:
            await send_status(websocket, "error", 0, {"message": str(e)})
        except:
            pass
