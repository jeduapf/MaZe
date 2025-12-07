"""
Optimized vectorized heatmap layer generator with multi-core processing.
Creates per-weight raster layers and weighted average layer using Web Mercator projection.

Performance optimizations:
- Kernel caching with file persistence (FFT-ready kernels)
- float32 throughout for memory efficiency
- Explicit memory cleanup
- Vectorized operations
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import folium
from folium import FeatureGroup, LayerControl
from folium.vector_layers import Circle
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import math
import json

# Optimize torch for CPU if needed
torch.set_num_threads(os.cpu_count() or 1)

# Kernel cache directory
KERNEL_CACHE_DIR = Path(__file__).parent / "cache" / "kernels"
KERNEL_CACHE_FILE = KERNEL_CACHE_DIR / "kernel_cache.json"


class KernelCache:
    """
    Manages kernel caching with file persistence.
    Stores pre-computed circular kernels by radius (spatial kernels persisted to file).
    FFT kernels are computed on-demand and cached in memory only.
    """
    
    def __init__(self):
        self._spatial_cache = {}  # radius -> np.ndarray (persisted)
        self._fft_cache = {}      # (radius, H, W) -> torch.Tensor (memory only)
        self._load_from_file()
    
    def _load_from_file(self):
        """Load spatial kernel cache from JSON file."""
        try:
            KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if KERNEL_CACHE_FILE.exists():
                with open(KERNEL_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to integers
                    self._spatial_cache = {int(k): np.array(v, dtype=np.float32) 
                                          for k, v in data.items()}
                print(f"✅ Loaded {len(self._spatial_cache)} cached kernels")
        except Exception as e:
            print(f"⚠️ Could not load kernel cache: {e}")
            self._spatial_cache = {}
    
    def _save_to_file(self):
        """Save spatial kernel cache to JSON file."""
        try:
            KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Only save spatial kernels (numpy arrays with integer keys)
            data = {str(k): v.tolist() for k, v in self._spatial_cache.items()}
            with open(KERNEL_CACHE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"⚠️ Could not save kernel cache: {e}")
    
    def get_kernel(self, radius: int) -> np.ndarray:
        """Get or create a circular kernel for the given radius."""
        if radius in self._spatial_cache:
            return self._spatial_cache[radius]
        
        # Create new kernel
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = ((x*x + y*y) <= (radius*radius)).astype(np.float32)
        
        # Cache it (spatial only)
        self._spatial_cache[radius] = kernel
        self._save_to_file()
        
        return kernel
    
    def get_fft_kernel(self, radius: int, target_shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Get pre-computed FFT-ready kernel (ifftshifted in frequency domain).
        FFT kernels are cached in memory only (not persisted to file).
        
        Args:
            radius: Kernel radius in pixels
            target_shape: (H, W) shape of the image to convolve with
            device: PyTorch device
            
        Returns:
            FFT of the zero-padded and shifted kernel, ready for element-wise multiplication
        """
        H, W = target_shape
        cache_key = (radius, H, W)
        
        # Check memory-only FFT cache
        if cache_key not in self._fft_cache:
            # Get spatial kernel
            kernel_np = self.get_kernel(radius)
            h, w = kernel_np.shape
            
            # Pad kernel to target shape
            pad_left = (W - w) // 2
            pad_right = W - w - pad_left
            pad_top = (H - h) // 2
            pad_bottom = H - h - pad_top
            
            kernel_tensor = torch.from_numpy(kernel_np).to(device)
            kernel_padded = F.pad(kernel_tensor, (pad_left, pad_right, pad_top, pad_bottom))
            
            # Pre-compute FFT with ifftshift
            kernel_fft = torch.fft.fft2(torch.fft.ifftshift(kernel_padded))
            
            self._fft_cache[cache_key] = kernel_fft
        
        return self._fft_cache[cache_key]
    
    def clear(self):
        """Clear both caches."""
        self._spatial_cache = {}
        self._fft_cache = {}
        if KERNEL_CACHE_FILE.exists():
            KERNEL_CACHE_FILE.unlink()
    
    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            'spatial_kernels': len(self._spatial_cache),
            'fft_kernels': len(self._fft_cache),
            'spatial_radii': list(self._spatial_cache.keys())
        }


# Global kernel cache instance
_kernel_cache = KernelCache()


class AdaptiveConvolution:
    """
    Automatically selects between spatial convolution and FFT-based convolution
    based on computational complexity.
    
    Complexity Analysis:
    - Spatial conv2d: O(H * W * h * w) where (H,W) is image size, (h,w) is kernel size
    - FFT method: O(H * W * log(H * W)) for FFT operations
    
    FFT becomes more efficient when kernel size is large relative to image size.
    """
    
    def __init__(self, threshold_ratio=0.15):
        """
        Args:
            threshold_ratio: If (kernel_area / image_area) > threshold_ratio, use FFT.
                           Default 0.15 means use FFT when kernel covers >15% of image.
        """
        self.threshold_ratio = threshold_ratio
        self.stats = {
            'conv2d_calls': 0,
            'fft_calls': 0
        }
    
    def _conv2d_method(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Spatial domain convolution."""
        input_for_conv = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        kernel_for_conv = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        result = F.conv2d(input_for_conv, kernel_for_conv, padding='same').squeeze()
        return result
    
    def _fft_method(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Frequency domain convolution using FFT."""
        H, W = image.shape
        h, w = kernel.shape
        
        # Compute padding to center the kernel
        pad_left = (W - w) // 2
        pad_right = W - w - pad_left
        pad_top = (H - h) // 2
        pad_bottom = H - h - pad_top
        
        kernel_padded = F.pad(kernel, (pad_left, pad_right, pad_top, pad_bottom))
        
        # 2D FFTs
        X = torch.fft.fft2(image)
        # Shift kernel start to top-left for FFT
        K = torch.fft.fft2(torch.fft.ifftshift(kernel_padded)) 
        Y = torch.fft.ifft2(X * K)
        
        return torch.abs(Y)
    
    def _fft_method_cached(self, image: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
        """FFT convolution using pre-computed kernel FFT."""
        X = torch.fft.fft2(image)
        Y = torch.fft.ifft2(X * kernel_fft)
        return torch.abs(Y)
    
    def _should_use_fft(self, image_shape: tuple, kernel_shape: tuple) -> bool:
        """Decide whether to use FFT based on complexity analysis."""
        H, W = image_shape
        h, w = kernel_shape
        
        # Method 1: Simple ratio heuristic
        kernel_area = h * w
        image_area = H * W
        kernel_ratio = kernel_area / image_area
        
        if kernel_ratio > self.threshold_ratio:
            return True
        
        # Method 2: Actual complexity comparison
        spatial_ops = H * W * h * w
        fft_ops = 3 * H * W * math.log2(H * W)
        
        return fft_ops < spatial_ops
    
    def convolve(self, image: torch.Tensor, kernel: torch.Tensor, 
                 force_method: str = None, kernel_fft: torch.Tensor = None) -> torch.Tensor:
        """
        Perform 2D convolution using the optimal method.
        
        Args:
            image: Input image tensor
            kernel: Spatial kernel tensor
            force_method: 'conv2d', 'fft', or None for auto-select
            kernel_fft: Pre-computed kernel FFT for fast FFT convolution
        """
        if force_method == 'conv2d':
            self.stats['conv2d_calls'] += 1
            return self._conv2d_method(image, kernel)
        elif force_method == 'fft':
            self.stats['fft_calls'] += 1
            if kernel_fft is not None:
                return self._fft_method_cached(image, kernel_fft)
            return self._fft_method(image, kernel)
        
        # Auto-select based on complexity
        if self._should_use_fft(image.shape, kernel.shape):
            self.stats['fft_calls'] += 1
            if kernel_fft is not None:
                return self._fft_method_cached(image, kernel_fft)
            return self._fft_method(image, kernel)
        else:
            self.stats['conv2d_calls'] += 1
            return self._conv2d_method(image, kernel)
    
    def __call__(self, image: torch.Tensor, kernel: torch.Tensor, 
                 force_method: str = None, kernel_fft: torch.Tensor = None) -> torch.Tensor:
        return self.convolve(image, kernel, force_method, kernel_fft)


class HeatmapLayerGenerator:
    """
    Generates vectorized heatmap image layers from circles.

    Each layer is a grayscale image saved as PNG. Each one of them may have 
    several circles having the same weight that overlap.

    After creating each separate layer, a weighted sum combination of them 
    is created and saved as a PNG.
    
    Attributes:
        circles (pd.DataFrame): DataFrame with "lat", "lon", "radius" (meters), "weight" (int)
        output_dir (Path): Directory to save PNG outputs
        R (float): Earth radius for Web Mercator projection (meters)
    """
    
    R = 6378137.0  # Earth's radius for Web Mercator (spherical)
    
    def __init__(self, circles: list, output_dir: str = "./heatmap_output"):
        """
        Initialize the generator with circles data.
        
        Args:
            circles: List of dicts with keys: "lat", "lon", "radius" (meters), "weight" (int)
            output_dir: Directory to save output PNGs
        """
        # Vectorized validation using DataFrame constructor
        self.circles = pd.DataFrame(circles).astype({
            'lat': np.float32,
            'lon': np.float32,
            'radius': np.float32,
            'weight': np.int32
        })
        
        self._project_circles()
        self.result = None
        self.map = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adaptive_conv = AdaptiveConvolution()
        self._kernel_cache = _kernel_cache
 
    def _latlon_to_mercator(self, lats, lons):
        """Convert lat/lon (degrees) to Web Mercator (meters)."""
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        
        x = self.R * lon_r
        lat_r = np.clip(lat_r, -np.pi/2 + 1e-8, np.pi/2 - 1e-8)
        y = self.R * np.log(np.tan(np.pi/4 + lat_r / 2.0))
        
        return x.astype(np.float32), y.astype(np.float32)
    
    def _mercator_to_latlon(self, x, y):
        """Convert Web Mercator (meters) to lat/lon (degrees)."""
        lon = (x / self.R) * 180 / np.pi
        lat = (2 * np.arctan(np.exp(y / self.R)) - np.pi / 2) * 180 / np.pi
        return lat.astype(np.float32), lon.astype(np.float32)
    
    def visualize(self, map_center=None, zoom_start=13, 
                  color_map=None, default_color='blue'):
        """
        Create a Folium map with circles organized in layers by weight.
        
        Parameters:
        -----------
        map_center : tuple, optional
            (lat, lon) for map center. If None, uses mean of data points
        zoom_start : int, default=13
            Initial zoom level for the map
        color_map : dict, optional
            Dictionary mapping weight values to colors
        default_color : str, default='blue'
            Default color if color_map not provided
        
        Returns:
        --------
        folium.Map
            Interactive Folium map with layer control
        """
        
        df = self.circles.copy()

        # Set map center
        if map_center is None:
            map_center = [df['lat'].mean(), df['lon'].mean()]
        
        # Create base map
        m = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # Define color palette if not provided
        if color_map is None:
            unique_weights = sorted(df['weight'].unique())
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 
                    'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                    'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 
                    'black', 'lightgray']
            color_map = {weight: colors[i % len(colors)] 
                        for i, weight in enumerate(unique_weights)}
        
        # Group by weight and create layers
        for weight in sorted(df['weight'].unique()):
            feature_group = FeatureGroup(name=f'Weight {weight}')
            weight_df = df[df['weight'] == weight]
            
            for idx, row in weight_df.iterrows():
                Circle(
                    location=[row['lat'], row['lon']],
                    radius=row['radius'],
                    color=color_map.get(weight, default_color),
                    fill=True,
                    fillColor=color_map.get(weight, default_color),
                    fillOpacity=0.2,
                    popup=f"Weight: {row['weight']}<br>"
                        f"Radius: {row['radius']}m<br>"
                        f"Lat: {row['lat']:.4f}<br>"
                        f"Lon: {row['lon']:.4f}<br>"
                        f"x: {row['x']:.4f}<br>"
                        f"y: {row['y']:.4f}",
                    tooltip=f"Weight: {weight}"
                ).add_to(feature_group)
            
            feature_group.add_to(m)
        
        LayerControl().add_to(m)
        
        return m

    def _project_circles(self, padding=0.05, max_zoom=15):
        """Project circles to Web Mercator and compute pixel indices."""
        self.circles["x"], self.circles["y"] = self._latlon_to_mercator(
            self.circles["lat"].values, self.circles["lon"].values
        )

        (min_lat, max_lat, min_lon, max_lon), (min_x, max_x, min_y, max_y) = self._compute_bbox(padding)
        img_w, img_h = self._calculate_optimal_resolution((min_x, max_x, min_y, max_y), max_zoom)

        step_x = (max_x - min_x) / (img_w - 1)
        step_y = (max_y - min_y) / (img_h - 1)
        mean_step = (step_x + step_y) / 2.0

        # Compute unique radius per weight
        unique_weights_radius = {}
        for w in self.circles["weight"].unique():
            max_radius = self.circles[self.circles["weight"] == w]["radius"].max()
            unique_weights_radius[int(w)] = int(round(max_radius / mean_step))
        
        self.unique_weights_radius = pd.DataFrame(
            unique_weights_radius.items(), 
            columns=["weight", "iradius"]
        )

        # Compute pixel indices for each circle
        self.circles['ix'] = np.clip(
            np.round((self.circles['x'] - min_x) / step_x).astype(np.int32), 
            0, img_w - 1
        )
        self.circles['iy'] = np.clip(
            np.round((max_y - self.circles['y']) / step_y).astype(np.int32), 
            0, img_h - 1
        )
        
        # Store computed values for later use
        self._bbox_mercator = (min_x, max_x, min_y, max_y)
        self._bbox_latlon = (min_lat, max_lat, min_lon, max_lon)
        self._img_size = (img_w, img_h)

    def _compute_bbox(self, padding=0.05):
        """Compute geographic bounding box covering all circles."""
        xs, ys = self._latlon_to_mercator(
            self.circles["lat"].values, 
            self.circles["lon"].values
        )
        radii = self.circles["radius"].values

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
        
        min_lat, min_lon = self._mercator_to_latlon(np.array([min_x]), np.array([min_y]))
        max_lat, max_lon = self._mercator_to_latlon(np.array([max_x]), np.array([max_y]))
        
        min_lat, max_lat = float(min(min_lat[0], max_lat[0])), float(max(min_lat[0], max_lat[0]))
        min_lon, max_lon = float(min(min_lon[0], max_lon[0])), float(max(min_lon[0], max_lon[0]))
        
        return (min_lat, max_lat, min_lon, max_lon), (min_x, max_x, min_y, max_y)
    
    def _calculate_optimal_resolution(self, bbox_mercator, max_zoom=15):
        """Calculate optimal image resolution based on zoom level."""
        min_x, max_x, min_y, max_y = bbox_mercator
        
        world_width_meters = 2 * np.pi * self.R
        pixels_per_meter = (256 * (2 ** max_zoom)) / world_width_meters
        
        width_pixels = int((max_x - min_x) * pixels_per_meter)
        height_pixels = int((max_y - min_y) * pixels_per_meter)
        
        width_pixels = max(256, min(width_pixels, 10240))
        height_pixels = max(256, min(height_pixels, 10240))
        
        return width_pixels, height_pixels
    
    def _colorize_layer(self, gray_array: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
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
    
    def _apply_gaussian_blur(self, gray_array: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur to a grayscale array."""
        if sigma is None or sigma <= 0:
            return gray_array
        
        pil_img = Image.fromarray(gray_array)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(pil_img, dtype=np.uint8)
    
    def _process_layers_convolution(self, grid_shape: tuple, gaussian_sigma: float = None) -> dict:
        """
        Process layers using Adaptive Convolution (PyTorch).
        Uses cached kernels for efficiency.
        """
        H, W = grid_shape
        unique_wr = list(self.unique_weights_radius.itertuples(index=False))
        img_layers = {}
        
        for weight, radius in unique_wr:
            # Filter circles with this weight
            layer_circles = self.circles[self.circles["weight"] == weight]
            
            # Create sparse point map (float32 for memory efficiency)
            point_map_np = np.zeros((H, W), dtype=np.float32)
            
            # Vectorized assignment using numpy advanced indexing
            point_indices = (layer_circles['iy'].values, layer_circles['ix'].values)
            np.add.at(point_map_np, point_indices, 1.0)  # Accumulate for overlapping points
            
            # Convert to Tensor
            point_map = torch.from_numpy(point_map_np).to(self.device)
            
            # Get cached kernel
            radius_int = int(radius)
            kernel_np = self._kernel_cache.get_kernel(radius_int)
            kernel = torch.from_numpy(kernel_np).to(self.device)
            
            # Check if FFT will be used to potentially use cached FFT kernel
            use_fft = self.adaptive_conv._should_use_fft((H, W), kernel_np.shape)
            
            if use_fft:
                kernel_fft = self._kernel_cache.get_fft_kernel(radius_int, (H, W), self.device)
                img_tensor = self.adaptive_conv(point_map, kernel, force_method='fft', kernel_fft=kernel_fft)
            else:
                img_tensor = self.adaptive_conv(point_map, kernel)
            
            # Normalize and Convert back
            max_val = img_tensor.max()
            if max_val > 0:
                img_normalized = (img_tensor / max_val) * 255.0
                img_normalized = torch.clamp(img_normalized, 0, 255).byte()
            else:
                img_normalized = torch.zeros((H, W), dtype=torch.uint8, device=self.device)
            
            img_layer = img_normalized.cpu().numpy()
            
            # Explicit memory cleanup
            del point_map, kernel, img_tensor, img_normalized
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if gaussian_sigma is not None and gaussian_sigma > 0:
                img_layer = self._apply_gaussian_blur(img_layer, gaussian_sigma)
            
            img_layers[int(weight)] = img_layer
        
        return img_layers

    def return_metrics_testing(self):
        """Return computed metrics for testing."""
        return (
            self._bbox_latlon,
            self._bbox_mercator,
            self._img_size[0], 
            self._img_size[1],
            None, None  # grid_x, grid_y no longer computed
        )

    def build(self, img_size='auto', padding=0.05, gaussian_sigma=2, 
              colormaps=None, max_zoom=15):
        """
        Build all heatmap layers.
        
        Args:
            img_size: 'auto', int, or (height, width) tuple
            padding: Bounding box padding fraction
            gaussian_sigma: Gaussian blur sigma (or None to disable)
            colormaps: Dict mapping weight -> colormap name
            max_zoom: Maximum zoom level for resolution calculation
            
        Returns:
            dict with layers, png_paths, bounds_latlon, max_counts, colormaps, resolution
        """
        if colormaps is None:
            colormaps = {}
        
        # Re-project with current parameters
        self._project_circles(padding=padding, max_zoom=max_zoom)
        
        min_lat, max_lat, min_lon, max_lon = self._bbox_latlon
        min_x, max_x, min_y, max_y = self._bbox_mercator
        
        if img_size == 'auto':
            img_w, img_h = self._img_size
        elif isinstance(img_size, int):
            img_h = img_w = img_size
        else:
            img_h, img_w = img_size
        
        grid_shape = (img_h, img_w)
        
        weights = self.circles["weight"].values
        unique_weights = np.unique(weights)

        # Process layers using convolution
        layers = self._process_layers_convolution(grid_shape, gaussian_sigma)
        
        # Save each layer as PNG
        png_paths = {}
        colormaps_used = {}
        max_counts = {}
        
        for w in unique_weights:
            w_int = int(w)
            norm = layers[w_int]
            max_counts[w_int] = int(norm.max())
            
            cmap = colormaps.get(w_int, 'viridis')
            colormaps_used[w_int] = cmap
            
            rgba = self._colorize_layer(norm, cmap)
            png_path = self.output_dir / f"layer_weight_{w}.png"
            Image.fromarray(rgba).save(png_path)
            png_paths[w_int] = str(png_path)
        
        # Create weighted average layer
        if len(unique_weights) > 0:
            denom = float(unique_weights.sum())
            accum = np.zeros((img_h, img_w), dtype=np.float32)
            
            for w in unique_weights:
                weight_val = int(w)
                norm_layer = layers[weight_val].astype(np.float32) / 255.0
                accum += weight_val * norm_layer
            
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
    import time
    
    sample_circles = [
        {"lat": 48.8566, "lon": 2.3522, "radius": 800, "weight": 1},
        {"lat": 48.8576, "lon": 2.3525, "radius": 800, "weight": 1},
        {"lat": 48.8558, "lon": 2.3510, "radius": 1000, "weight": 1},
        {"lat": 48.8590, "lon": 2.3550, "radius": 600, "weight": 5},
        {"lat": 48.8570, "lon": 2.3600, "radius": 700, "weight": 4},
        {"lat": 48.8545, "lon": 2.3580, "radius": 400, "weight": 5},
    ]
    
    print("🚀 Starting optimized heatmap generation...")
    start = time.perf_counter()
    
    gen = HeatmapLayerGenerator(sample_circles, output_dir="./heatmap_output")
    result = gen.build(img_size='auto', padding=0.06, gaussian_sigma=2, max_zoom=15)
    m = gen.create_map(zoom=15)
    map_path = gen.save_map()
    
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"\n[OK] Heatmap layers generated successfully!")
    print(f"  Output directory: {gen.output_dir}")
    print(f"  Resolution: {result['resolution'][0]}x{result['resolution'][1]} pixels")
    print(f"  Total time: {elapsed:.2f}ms")
    print(f"  Convolution stats: {gen.adaptive_conv.stats}")