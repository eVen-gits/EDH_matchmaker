import os
import discord
import asyncio
from discord.ext import commands
from dotenv import load_dotenv
import threading

load_dotenv()


class DiscordPoster:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.token = os.getenv("DISCORD_TOKEN")
        self.guild_id = int(os.getenv("GUILD_ID", 0))
        self.channel_id = int(os.getenv("CHANNEL_ID", 0))

        # Set up bot
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)

        # Message queue
        self.message_queue = asyncio.Queue()

        # To detect when loop is ready
        self.loop_ready = threading.Event()

        # Background task reference
        self.queue_task = None

        @self.bot.event
        async def on_ready():
            print(f"Bot connected as {self.bot.user}")

            # Start background queue processor
            if not self.queue_task:
                self.queue_task = asyncio.create_task(self._process_queued_messages())

        # Start bot in background thread
        self.thread = threading.Thread(target=self._run_bot_loop, daemon=True)
        self.thread.start()

        self._initialized = True

    def _run_bot_loop(self):
        """Run the bot and make its loop accessible."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()  # Notify that loop is ready
        self.loop.run_until_complete(self.bot.start(self.token))

    async def _process_queued_messages(self):
        """Continuously process queued messages forever."""
        print("[DiscordPoster] Starting message queue processor...")
        while True:
            message = await self.message_queue.get()
            await self._send_message(message)
            self.message_queue.task_done()

    async def _send_message(self, message: str):
        """Send a message to the channel."""
        channel = self.bot.get_channel(self.channel_id)
        if channel:
            await channel.send(message)
            print(f"[Discord] Sent: {message}")
        else:
            print("[Discord] Error: Channel not found.")

    def post_message(self, message: str):
        """
        Thread-safe public method to queue a message.
        """
        #print(f"[DiscordPoster] Queuing message: {message}")

        # Wait until loop is ready
        self.loop_ready.wait()

        # Safely queue message
        asyncio.run_coroutine_threadsafe(self.message_queue.put(message), self.loop)

    async def close(self):
        """Gracefully shut down bot and background task."""
        if hasattr(self, 'loop') and self.loop.is_running():
            await self.bot.close()
            if self.queue_task:
                self.queue_task.cancel()

    @classmethod
    def instance(cls):
        """Get the singleton instance."""
        return cls()
