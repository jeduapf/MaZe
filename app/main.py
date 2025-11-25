from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.utils import get_city_coordinates, generate_map_html

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/cities")
async def search_cities(q: str = ""):
    """Search for cities matching the query."""
    from app.utils import search_cities
    if len(q) < 2:
        return []
    return search_cities(q)

@app.post("/generate_map", response_class=HTMLResponse)
async def generate_map(
    request: Request,
    city: str = Form(...),
    supermarket_weight: int = Form(5),
    bakery_weight: int = Form(5),
    metro_weight: int = Form(5),
    bus_weight: int = Form(5),
    tram_weight: int = Form(5),
    rent_weight: int = Form(5),
    radius: int = Form(3000)
):
    # Get coordinates
    location = get_city_coordinates(city)
    if not location:
        return templates.TemplateResponse("index.html", {"request": request, "error": "City not found"})
    
    # Weights dictionary
    weights = {
        "supermarket": supermarket_weight,
        "bakery": bakery_weight,
        "metro": metro_weight,
        "bus_stop": bus_weight,
        "tram_stop": tram_weight,
        "rent": rent_weight
    }

    # Generate map
    map_html = generate_map_html(location, weights, radius)
    
    return templates.TemplateResponse("index.html", {"request": request, "map_html": map_html, "city": city})
