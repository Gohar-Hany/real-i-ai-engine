import asyncio, sys; sys.path.insert(0, 'd:/REAL_i/src'); from motor.motor_asyncio import AsyncIOMotorClient; from helpers.config import get_settings; from models.UserModel import UserModel
async def seed():
 settings = get_settings()
 db = AsyncIOMotorClient(settings.MONGODB_URL)[settings.MONGODB_DATABASE]
 m = UserModel(db)
 existing = await m.get_user_by_email('goharhany@gmail.com')
 if not existing:
  await m.create_user('Gohar Hany', 'goharhany@gmail.com', 'admin123', 'admin', 'https://api.dicebear.com/7.x/avataaars/svg?seed=admin&backgroundColor=b6e3f4')
  print('Seeded admin')
 else:
  print('Admin exists')
asyncio.run(seed())
