"""
Stress test for heatmap generator with 5000 circles across Paris metro area.
"""
import time
import random
import numpy as np
from app.heatmap_layers import HeatmapLayerGenerator, _kernel_cache

# Paris metropolitan area bounds (approx 50km x 50km)
# Center: 48.8566, 2.3522
# Covers: Paris, Versailles, Saint-Denis, Boulogne, Montreuil, etc.
LAT_MIN, LAT_MAX = 48.70, 49.00  # ~33km north-south
LON_MIN, LON_MAX = 2.10, 2.60   # ~37km east-west

NUM_CIRCLES = 5000
RADII = [300, 400, 500, 600, 700, 800, 1000]  # meters
WEIGHTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def generate_random_circles(n):
    """Generate n random circles across Paris metro area."""
    circles = []
    for i in range(n):
        circles.append({
            "lat": random.uniform(LAT_MIN, LAT_MAX),
            "lon": random.uniform(LON_MIN, LON_MAX),
            "radius": random.choice(RADII),
            "weight": random.choice(WEIGHTS)
        })
    return circles

def run_stress_test():
    print(f"🚀 Stress Test: {NUM_CIRCLES} circles across Paris metro area")
    print(f"   Area: {LAT_MIN}-{LAT_MAX}°N, {LON_MIN}-{LON_MAX}°E")
    print()
    
    # Generate circles
    t0 = time.perf_counter()
    circles = generate_random_circles(NUM_CIRCLES)
    t_gen = (time.perf_counter() - t0) * 1000
    print(f"⏱️  Circle generation: {t_gen:.1f}ms")
    
    # Count unique weights
    weights = set(c["weight"] for c in circles)
    print(f"   Unique weights: {len(weights)}")
    print()
    
    # Initialize generator
    t0 = time.perf_counter()
    gen = HeatmapLayerGenerator(circles, output_dir="./benchmark_output")
    t_init = (time.perf_counter() - t0) * 1000
    print(f"⏱️  Initialization (projection): {t_init:.1f}ms")
    
    # Build heatmap with timing
    print()
    print("🔨 Building heatmap layers...")
    
    t0 = time.perf_counter()
    result = gen.build(img_size='auto', padding=0.02, gaussian_sigma=2, max_zoom=14)
    t_build = (time.perf_counter() - t0) * 1000
    
    print()
    print("=" * 60)
    print(f"✅ BUILD COMPLETE")
    print(f"   Resolution: {result['resolution'][0]}x{result['resolution'][1]} pixels")
    print(f"   Total build time: {t_build:.1f}ms ({t_build/1000:.2f}s)")
    print()
    
    # Convolution stats
    print(f"📊 Convolution stats: {gen.adaptive_conv.stats}")
    
    # Kernel cache stats
    print(f"📦 Kernel cache: {_kernel_cache.get_stats()}")
    print()
    
    # Create map
    t0 = time.perf_counter()
    m = gen.create_map(zoom=12)
    map_path = gen.save_map("stress_test_5000.html")
    t_map = (time.perf_counter() - t0) * 1000
    print(f"⏱️  Map creation: {t_map:.1f}ms")
    print(f"   Saved to: {map_path}")
    print()
    
    # Summary
    total = t_gen + t_init + t_build + t_map
    print("=" * 60)
    print("📈 PERFORMANCE BREAKDOWN")
    print("=" * 60)
    print(f"   Circle generation:  {t_gen:8.1f}ms ({t_gen/total*100:5.1f}%)")
    print(f"   Initialization:     {t_init:8.1f}ms ({t_init/total*100:5.1f}%)")
    print(f"   Build (convolution):{t_build:8.1f}ms ({t_build/total*100:5.1f}%)")
    print(f"   Map creation:       {t_map:8.1f}ms ({t_map/total*100:5.1f}%)")
    print(f"   ─────────────────────────────────")
    print(f"   TOTAL:              {total:8.1f}ms ({total/1000:.2f}s)")
    
    return result

if __name__ == "__main__":
    random.seed(42)  # Reproducible results
    run_stress_test()
