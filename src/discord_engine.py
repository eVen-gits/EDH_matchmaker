import os
import discord
import asyncio
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

class DiscordPoster:
    def __init__(self):
        self.token = os.getenv("DISCORD_TOKEN")
        self.guild_id = int(os.getenv("GUILD_ID"))
        self.channel_id = int(os.getenv("CHANNEL_ID"))
        self.bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

    async def send_message(self, message: str):
        async def on_ready():
            guild = self.bot.get_guild(self.guild_id)
            if guild is None:
                print("Guild not found!")
                await self.bot.close()
                return

            channel = guild.get_channel(self.channel_id)
            if channel is None:
                print("Channel not found!")
                await self.bot.close()
                return

            await channel.send(message)
            await self.bot.close()

        self.bot.add_listener(on_ready)
        await self.bot.start(self.token)

    def post_message(self, message: str):
        asyncio.run(self.send_message(message))
