import asyncio
from ui.app import app, orchestrator

async def main():
    # Start Orchestrator background tasks
    asyncio.create_task(orchestrator.run())
    # Run the Quart app
    await app.run_task(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    asyncio.run(main())