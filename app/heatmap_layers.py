"""
Optimized vectorized heatmap layer generator with multi-core processing.
Creates per-weight raster layers and weighted average layer using Web Mercator projection.
"""

from math import pi, log, tan
import numpy as np
from PIL import Image, ImageFilter
import folium
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .logger import timeit, PerformanceTimer, log_timing


# Module-level function for multiprocessing (must be picklable)
def _process_circle_batch(args):
    """Process a batch of circles and return accumulated mask."""
    circles_data, grid_shape, grid_bounds = args
    grid_x_min, grid_x_max, grid_y_min, grid_y_max = grid_bounds
    h, w = grid_shape
    
    # Recreate grid (can't pickle large arrays efficiently)
    xs = np.linspace(grid_x_min, grid_x_max, w, dtype=np.float32)
    ys = np.linspace(grid_y_max, grid_y_min, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
    
    count = np.zeros((h, w), dtype=np.uint16)
    for cx, cy, r in circles_data:
        dx = grid_x - cx
        dy = grid_y - cy
        mask = (dx*dx + dy*dy) <= (r**2)
        count[mask] += 1
    
    return count


class HeatmapLayerGenerator:
    """
    Generates vectorized heatmap layers from circles with efficient memory usage.
    
    Attributes:
        circles (list): List of dicts with "lat", "lon", "radius" (meters), "weight" (int)
        output_dir (str): Directory to save PNG outputs
        R (float): Earth radius for Web Mercator projection (meters)
    """
    
    R = 6378137.0  # Earth's radius for Web Mercator (spherical)
    
    def __init__(self, circles, output_dir="./heatmap_output"):
        """
        Initialize the generator with circles data.
        
        Args:
            circles: List of dicts with keys: "lat", "lon", "radius" (meters), "weight" (int)
            output_dir: Directory to save output PNGs
        """
        self.circles = self._validate_circles(circles)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.result = None
        self.map = None
    
    @staticmethod
    def _validate_circles(circles):
        """Validate and standardize circle data."""
        validated = []
        for c in circles:
            validated.append({
                "lat": float(c["lat"]),
                "lon": float(c["lon"]),
                "radius": float(c["radius"]),
                "weight": int(c["weight"])
            })
        return validated
    
    def _latlon_to_mercator(self, lats, lons):
        """Convert lat/lon (degrees) to Web Mercator (meters)."""
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        
        x = self.R * lon_r
        lat_r = np.clip(lat_r, -np.pi/2 + 1e-8, np.pi/2 - 1e-8)
        y = self.R * np.log(np.tan(np.pi/4 + lat_r / 2.0))
        
        return x, y
    
    def _mercator_to_latlon(self, x, y):
        """Convert Web Mercator (meters) to lat/lon (degrees)."""
        lon = (x / self.R) * 180 / np.pi
        lat = (2 * np.arctan(np.exp(y / self.R)) - np.pi / 2) * 180 / np.pi
        return lat, lon
    
    def _compute_bbox(self, padding=0.05):
        """Compute geographic bounding box covering all circles."""
        lats = np.array([c["lat"] for c in self.circles])
        lons = np.array([c["lon"] for c in self.circles])
        radii = np.array([c["radius"] for c in self.circles])
        
        xs, ys = self._latlon_to_mercator(lats, lons)
        
        min_x = (xs - radii).min()
        max_x = (xs + radii).max()
        min_y = (ys - radii).min()
        max_y = (ys + radii).max()
        
        pad_x = (max_x - min_x) * padding
        pad_y = (max_y - min_y) * padding
        
        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y
        
        min_lat, min_lon = self._mercator_to_latlon(min_x, min_y)
        max_lat, max_lon = self._mercator_to_latlon(max_x, max_y)
        
        min_lat, max_lat = min(min_lat, max_lat), max(min_lat, max_lat)
        min_lon, max_lon = min(min_lon, max_lon), max(min_lon, max_lon)
        
        return (min_lat, max_lat, min_lon, max_lon), (min_x, max_x, min_y, max_y)
    
    def _build_grid(self, min_x, max_x, min_y, max_y, img_w, img_h):
        """Create mercator grid using linspace (memory efficient)."""
        xs = np.linspace(min_x, max_x, img_w, dtype=np.float32)
        ys = np.linspace(max_y, min_y, img_h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
        return grid_x, grid_y
    
    def _colorize_layer(self, gray_array, colormap='viridis'):
        """Convert grayscale array to RGBA using a colormap."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        try:
            cmap = plt.get_cmap(colormap)
        except:
            cmap = plt.get_cmap('hot')
        
        normalized = gray_array.astype(np.float32) / 255.0
        rgba = cmap(normalized)
        rgba = (rgba * 255).astype(np.uint8)
        rgba[..., 3] = gray_array
        
        return rgba
    
    def _apply_gaussian_blur(self, gray_array, sigma):
        """Apply Gaussian blur to a grayscale array."""
        if sigma is None or sigma <= 0:
            return gray_array
        
        pil_img = Image.fromarray(gray_array)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(pil_img, dtype=np.uint8)
    
    def _calculate_optimal_resolution(self, bbox_mercator, max_zoom=15):
        """Calculate optimal image resolution based on zoom level."""
        min_x, max_x, min_y, max_y = bbox_mercator
        
        world_width_meters = 2 * 20037508.34
        pixels_per_meter = (256 * (2 ** max_zoom)) / world_width_meters
        
        width_pixels = int((max_x - min_x) * pixels_per_meter)
        height_pixels = int((max_y - min_y) * pixels_per_meter)
        
        width_pixels = max(256, min(width_pixels, 4096))
        height_pixels = max(256, min(height_pixels, 4096))
        
        return width_pixels, height_pixels
    
    def _process_weight_layer(self, weight, circle_indices, grid_x, grid_y, radii, centers_x, centers_y, gaussian_sigma):
        """Process a single weight layer efficiently."""
        sel_centers_x = centers_x[circle_indices]
        sel_centers_y = centers_y[circle_indices]
        sel_radii = radii[circle_indices]
        
        h, w = grid_x.shape
        count = np.zeros((h, w), dtype=np.uint16)
        
        for cx, cy, r in zip(sel_centers_x, sel_centers_y, sel_radii):
            dx = grid_x - float(cx)
            dy = grid_y - float(cy)
            mask = (dx*dx + dy*dy) <= (float(r)**2)
            count[mask] += 1
        
        if count.max() > 0:
            norm = np.clip((count.astype(np.float32) / count.max()) * 255.0, 0, 255).astype(np.uint8)
        else:
            norm = np.zeros((h, w), dtype=np.uint8)
        
        if gaussian_sigma is not None and gaussian_sigma > 0:
            norm = self._apply_gaussian_blur(norm, gaussian_sigma)
        
        return norm, int(count.max())
    
    def _process_weight_layer_parallel(self, weight, circle_indices, grid_bounds, grid_shape, radii, centers_x, centers_y, gaussian_sigma, num_workers=4):
        """Process a weight layer using multiple CPU cores."""
        sel_centers_x = centers_x[circle_indices]
        sel_centers_y = centers_y[circle_indices]
        sel_radii = radii[circle_indices]
        
        # Prepare circle data as list of tuples
        circles_data = list(zip(sel_centers_x.tolist(), sel_centers_y.tolist(), sel_radii.tolist()))
        
        if len(circles_data) < 10:
            # Too few circles, use single-threaded
            grid_x, grid_y = self._build_grid(*grid_bounds, grid_shape[1], grid_shape[0])
            return self._process_weight_layer(weight, circle_indices, grid_x, grid_y, radii, centers_x, centers_y, gaussian_sigma)
        
        # Split circles into batches for parallel processing
        batch_size = max(1, len(circles_data) // num_workers)
        batches = [circles_data[i:i+batch_size] for i in range(0, len(circles_data), batch_size)]
        
        # Process batches in parallel
        args_list = [(batch, grid_shape, grid_bounds) for batch in batches]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_circle_batch, args_list))
        
        # Combine results
        h, w = grid_shape
        count = np.zeros((h, w), dtype=np.uint16)
        for result in results:
            count += result
        
        if count.max() > 0:
            norm = np.clip((count.astype(np.float32) / count.max()) * 255.0, 0, 255).astype(np.uint8)
        else:
            norm = np.zeros((h, w), dtype=np.uint8)
        
        if gaussian_sigma is not None and gaussian_sigma > 0:
            norm = self._apply_gaussian_blur(norm, gaussian_sigma)
        
        return norm, int(count.max())
    
    @timeit
    def build(self, img_size='auto', padding=0.05, gaussian_sigma=2, colormaps=None, max_zoom=15, use_multicore=True):
        """Build all heatmap layers."""
        if colormaps is None:
            colormaps = {}
        
        (min_lat, max_lat, min_lon, max_lon), (min_x, max_x, min_y, max_y) = self._compute_bbox(padding)
        
        if img_size == 'auto':
            img_w, img_h = self._calculate_optimal_resolution((min_x, max_x, min_y, max_y), max_zoom)
            log_timing(f"🎯 Auto-calculated resolution: {img_w}x{img_h} pixels (zoom level {max_zoom})")
        elif isinstance(img_size, int):
            img_h = img_w = img_size
        else:
            img_h, img_w = img_size
        
        grid_x, grid_y = self._build_grid(min_x, max_x, min_y, max_y, img_w, img_h)
        grid_bounds = (min_x, max_x, min_y, max_y)
        grid_shape = (img_h, img_w)
        
        centers_x = np.array([self._latlon_to_mercator(c["lat"], c["lon"])[0] for c in self.circles], dtype=np.float32)
        centers_y = np.array([self._latlon_to_mercator(c["lat"], c["lon"])[1] for c in self.circles], dtype=np.float32)
        radii = np.array([c["radius"] for c in self.circles], dtype=np.float32)
        weights = np.array([c["weight"] for c in self.circles], dtype=np.int32)
        
        layers = {}
        png_paths = {}
        max_counts = {}
        colormaps_used = {}
        unique_weights = np.unique(weights)
        
        for w in unique_weights:
            indices = np.where(weights == w)[0]
            
            if use_multicore and len(indices) > 10:
                norm, max_count = self._process_weight_layer_parallel(
                    w, indices, grid_bounds, grid_shape, radii, centers_x, centers_y, gaussian_sigma
                )
            else:
                norm, max_count = self._process_weight_layer(
                    w, indices, grid_x, grid_y, radii, centers_x, centers_y, gaussian_sigma
                )
            
            layers[int(w)] = norm
            max_counts[int(w)] = max_count
            
            cmap = colormaps.get(int(w), 'viridis')
            colormaps_used[int(w)] = cmap
            
            rgba = self._colorize_layer(norm, cmap)
            png_path = self.output_dir / f"layer_weight_{w}.png"
            Image.fromarray(rgba).save(png_path)
            png_paths[int(w)] = str(png_path)
        
        if len(unique_weights) > 0:
            denom = float(unique_weights.sum())
            accum = np.zeros((img_h, img_w), dtype=np.float32)
            
            for w in unique_weights:
                weight_val = int(w)
                norm_layer = layers[weight_val].astype(np.float32) / 255.0
                accum = accum + (weight_val * norm_layer)
            
            weighted_avg = accum / denom
            max_val = weighted_avg.max()
            
            if max_val > 0:
                weighted_norm = np.clip((weighted_avg / max_val) * 255.0, 0, 255).astype(np.uint8)
            else:
                weighted_norm = np.zeros((img_h, img_w), dtype=np.uint8)
            
            if gaussian_sigma is not None and gaussian_sigma > 0:
                weighted_norm = self._apply_gaussian_blur(weighted_norm, gaussian_sigma)
            
            layers["weighted_avg"] = weighted_norm
            
            avg_cmap = colormaps.get('weighted_avg', 'hot')
            colormaps_used['weighted_avg'] = avg_cmap
            
            rgba = self._colorize_layer(weighted_norm, avg_cmap)
            png_path = self.output_dir / "layer_weighted_avg.png"
            Image.fromarray(rgba).save(png_path)
            png_paths["weighted_avg"] = str(png_path)
        
        self.result = {
            "layers": layers,
            "png_paths": png_paths,
            "bounds_latlon": (min_lat, max_lat, min_lon, max_lon),
            "max_counts": max_counts,
            "colormaps": colormaps_used,
            "resolution": (img_w, img_h)
        }
        
        return self.result
    
    def create_map(self, zoom=13, center=None):
        """Create a Folium map with all layers."""
        if self.result is None:
            raise ValueError("Must call build() before create_map()")
        
        min_lat, max_lat, min_lon, max_lon = self.result["bounds_latlon"]
        
        if center is None:
            center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
        
        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
        
        for key, path in self.result["png_paths"].items():
            if key == "weighted_avg":
                name = "Weighted Average (Heatmap)"
                show = True
                zindex = 2
            else:
                name = f"Weight {key}"
                show = False
                zindex = 1

            img_overlay = folium.raster_layers.ImageOverlay(
                image=path,
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                name=name,
                opacity=0.5,
                interactive=True,
                zindex=zindex,
                show=show
            )
            img_overlay.add_to(m)

        folium.LayerControl(collapsed=True).add_to(m)
        self.map = m
        
        return m
    
    def save_map(self, filename="heatmap_map.html"):
        """Save the Folium map to an HTML file."""
        if self.map is None:
            raise ValueError("Must call create_map() before save_map()")
        
        path = self.output_dir / filename
        self.map.save(path)
        return str(path)


# Demo
if __name__ == "__main__":
    sample_circles = [
        {"lat": 48.8566, "lon": 2.3522, "radius": 800, "weight": 1},
        {"lat": 48.8576, "lon": 2.3525, "radius": 800, "weight": 1},
        {"lat": 48.8558, "lon": 2.3510, "radius": 1000, "weight": 1},
        {"lat": 48.8590, "lon": 2.3550, "radius": 600, "weight": 5},
        {"lat": 48.8570, "lon": 2.3600, "radius": 700, "weight": 4},
        {"lat": 48.8545, "lon": 2.3580, "radius": 400, "weight": 5},
    ]
    
    gen = HeatmapLayerGenerator(sample_circles, output_dir="./heatmap_output")
    result = gen.build(img_size='auto', padding=0.06, gaussian_sigma=2, max_zoom=15)
    m = gen.create_map(zoom=14)
    map_path = gen.save_map()
    
    print("✓ Heatmap layers generated successfully!")
    print(f"  Output directory: {gen.output_dir}")
    print(f"  Resolution: {result['resolution'][0]}x{result['resolution'][1]} pixels")