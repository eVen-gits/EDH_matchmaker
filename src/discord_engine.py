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

        self.token: str = os.getenv("DISCORD_TOKEN", "")
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
            # Start queue processor inside bot's event loop
            if not self.queue_task or self.queue_task.done():
                self.queue_task = self.bot.loop.create_task(self._process_queued_messages())

        # Start bot in background thread
        self.thread = threading.Thread(target=self._run_bot_loop, daemon=True)
        self.thread.start()

        self.loop_ready.wait()  # Ensure loop is ready

        self._initialized = True

    def _run_bot_loop(self):
        """Create and run the bot's event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()  # Notify that the loop is ready

        try:
            self.loop.run_until_complete(self.bot.start(self.token))
        except KeyboardInterrupt:
            self.loop.run_until_complete(self.bot.close())
        finally:
            self.loop.close()

    async def _process_queued_messages(self):
        """Continuously process queued messages and restart on failure."""

        while True:
            try:
                # Ensure we're in the correct event loop
                if asyncio.get_running_loop() != self.loop:
                    await asyncio.sleep(1)
                    continue  # Don't process messages in the wrong loop

                # Check if queue is empty before calling get()
                if self.message_queue.empty():
                    await asyncio.sleep(1)  # Avoid tight looping if there's nothing to process
                    continue

                message = await self.message_queue.get()

                await self._send_message(message)
                self.message_queue.task_done()

            except asyncio.CancelledError:
                break  # Exit the loop gracefully if the task is cancelled

            except Exception as e:
                await asyncio.sleep(3)  # Small delay before retrying


    async def _send_message(self, message: str):
        """Send a message to the channel and return the message ID."""
        channel = self.bot.get_channel(self.channel_id)
        if channel:
            sent_message = await channel.send(message)
            return sent_message.id
        else:
            return None

    def post_message(self, message: str):
        """Safely post a message from any thread using self.loop."""
        if not self.loop or not self.loop.is_running():
            return

        future = asyncio.run_coroutine_threadsafe(self.message_queue.put(message), self.loop)
        try:
            future.result()  # Wait for completion
        except Exception as e:
            print(f"[DiscordPoster] Error posting message: {e}")

    async def close(self):
        """Gracefully shut down bot and background task."""
        if self.loop and self.loop.is_running():
            await self.bot.close()
            if self.queue_task:
                self.queue_task.cancel()

    @classmethod
    def instance(cls):
        """Get the singleton instance."""
        return cls()
