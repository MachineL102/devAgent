# test_aiohttp.py
import aiohttp
import asyncio

async def test():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://www.google.com') as response:
            print("HTTP status:", response.status)

asyncio.run(test())
