from sqlalchemy.ext.asyncio import AsyncSession

async def get_session(session_maker):
    async with session_maker() as session:
        yield session