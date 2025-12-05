import os
import asyncio
import aiohttp
from typing import List, Tuple, Optional
from aiohttp import ClientSession


class AsyncGeocoder:
    def __init__(self, api: str = "nominatim", google_api_env: str = "GOOGLE_API_KEY"):
        """
        api: which API to use by default ("nominatim" or "google")
        google_api_env: environment variable storing Google key
        """
        self.api = api.lower()
        self.google_api_key = os.getenv(google_api_env)

        if self.api == "google" and not self.google_api_key:
            raise ValueError(f"Google API key not found in environment variable {google_api_env}")

        # Nominatim = allow only 1 request per second
        self.nominatim_lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # FORWARD GEOCODING  (address → lat, lon)
    # -------------------------------------------------------------------------

    async def geocode_nominatim(self, session: ClientSession, address: str) -> Tuple[Optional[float], Optional[float]]:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json", "limit": 1}
        headers = {"User-Agent": "MaZe-App/1.0 (maze.app@example.com)"}

        async with self.nominatim_lock:
            await asyncio.sleep(1)  # Respect Nominatim rate limit

            try:
                async with session.get(url, params=params, headers=headers, timeout=10) as resp:
                    if resp.status != 200:
                        print(f"Nominatim returned status {resp.status}")
                        return None, None
                    data = await resp.json()
                    if not data:
                        print(f"No results for: {address}")
                        return None, None
                    return float(data[0]["lat"]), float(data[0]["lon"])
            except Exception as e:
                print(f"Nominatim error: {e}")
                return None, None

    async def geocode_google(self, session: ClientSession, address: str) -> Tuple[Optional[float], Optional[float]]:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address, "key": self.google_api_key}

        try:
            async with session.get(url, params=params, timeout=10) as resp:
                data = await resp.json()
                if data.get("status") != "OK":
                    return None, None
                loc = data["results"][0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
        except Exception:
            return None, None

    # -------------------------------------------------------------------------
    # REVERSE GEOCODING  (lat, lon → address)
    # -------------------------------------------------------------------------

    async def reverse_nominatim(self, session: ClientSession, lat: float, lon: float) -> Optional[str]:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "json"}
        headers = {"User-Agent": "MaZe-App/1.0 (maze.app@example.com)"}

        async with self.nominatim_lock:
            await asyncio.sleep(1)

            try:
                async with session.get(url, params=params, headers=headers, timeout=10) as resp:
                    if resp.status != 200:
                        print(f"Reverse geocoding returned status {resp.status}")
                        return None
                    data = await resp.json()
                    return data.get("display_name")
            except Exception as e:
                print(f"Reverse geocoding error: {e}")
                return None

    async def reverse_google(self, session: ClientSession, lat: float, lon: float) -> Optional[str]:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"latlng": f"{lat},{lon}", "key": self.google_api_key}

        try:
            async with session.get(url, params=params, timeout=10) as resp:
                data = await resp.json()
                if data.get("status") != "OK":
                    return None
                return data["results"][0]["formatted_address"]
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # INTERNAL ROUTERS
    # -------------------------------------------------------------------------

    async def _forward(self, session, address):
        if self.api == "nominatim":
            lat, lon = await self.geocode_nominatim(session, address)
            if (lat is None or lon is None) and self.google_api_key:
                lat, lon = await self.geocode_google(session, address)
            return lat, lon

        if self.api == "google":
            lat, lon = await self.geocode_google(session, address)
            if lat is None or lon is None:
                lat, lon = await self.geocode_nominatim(session, address)
            return lat, lon

    async def _reverse(self, session, lat, lon):
        if self.api == "nominatim":
            addr = await self.reverse_nominatim(session, lat, lon)
            if not addr and self.google_api_key:
                addr = await self.reverse_google(session, lat, lon)
            return addr

        if self.api == "google":
            addr = await self.reverse_google(session, lat, lon)
            if not addr:
                addr = await self.reverse_nominatim(session, lat, lon)
            return addr

    # -------------------------------------------------------------------------
    # PUBLIC METHODS (single + batch)
    # -------------------------------------------------------------------------

    async def geocode(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        async with aiohttp.ClientSession() as session:
            return await self._forward(session, address)

    async def reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
        async with aiohttp.ClientSession() as session:
            return await self._reverse(session, lat, lon)

    async def geocode_batch(self, addresses: List[str], max_concurrent: int = 5):
        semaphore = asyncio.Semaphore(max_concurrent)

        async with aiohttp.ClientSession() as session:
            async def task(addr):
                async with semaphore:
                    return await self._forward(session, addr)

            return await asyncio.gather(*(task(a) for a in addresses))

    async def reverse_batch(self, coords: List[Tuple[float, float]], max_concurrent: int = 5):
        semaphore = asyncio.Semaphore(max_concurrent)

        async with aiohttp.ClientSession() as session:
            async def task(lat, lon):
                async with semaphore:
                    return await self._reverse(session, lat, lon)

            return await asyncio.gather(*(task(lat, lon) for lat, lon in coords))


# ------------------------------------------------------------------------------
# Synchronous wrappers for FastAPI routes
# ------------------------------------------------------------------------------
def geocode_sync(address: str) -> Tuple[Optional[float], Optional[float]]:
    """Synchronous wrapper for geocoding (for use in non-async contexts)"""
    geocoder = AsyncGeocoder(api="nominatim")
    return asyncio.run(geocoder.geocode(address))

def reverse_geocode_sync(lat: float, lon: float) -> Optional[str]:
    """Synchronous wrapper for reverse geocoding"""
    geocoder = AsyncGeocoder(api="nominatim")
    return asyncio.run(geocoder.reverse_geocode(lat, lon))


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
async def main():
    geocoder = AsyncGeocoder(api="nominatim")  # default API

    # Forward geocoding (single)
    lat, lon = await geocoder.geocode("Eiffel Tower, Paris")
    print("Eiffel Tower:", lat, lon)

    # Reverse geocoding (single)
    address = await geocoder.reverse_geocode(lat, lon)
    print("Reverse:", address)

    # Batch geocoding
    addresses = [
        "Times Square, New York",
        "10 Downing Street, London"
    ]
    results = await geocoder.geocode_batch(addresses)
    print(results)

    # Batch reverse
    coords = [(lat, lon)]
    rev_results = await geocoder.reverse_batch(coords)
    print(rev_results)


if __name__ == "__main__":
    asyncio.run(main())
