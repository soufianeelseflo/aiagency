"""
agents/vpn_manager.py

Provides installation and runtime management for NordVPN client, including login, double VPN, connection cycling, and status checks.
"""
import subprocess
import logging
import time
from config.settings import settings

logger = logging.getLogger(__name__)

class VPNManager:
    def __init__(self):
        self.double_vpn = settings.get('NORDVPN_DOUBLE_VPN', True)
        self.login_email = None

    def install_client(self):
        """Installs NordVPN client via official script."""
        logger.info("Installing NordVPN client...")
        try:
            subprocess.run([
                "bash", "-c",
                "sh <(curl -sSf https://downloads.nordcdn.com/apps/linux/install.sh)"
            ], check=True)
            logger.info("NordVPN client installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install NordVPN client: {e}")
            raise

    def login(self, email: str, otp: str = None):
        """Performs login. If OTP required, handles interactive prompt."""
        self.login_email = email
        cmd = ["nordvpn", "login", "--email", email]
        if otp:
            cmd.extend(["--otp", otp])
        logger.info(f"Logging in to NordVPN as {email}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"NordVPN login failed: {result.stderr}")
            raise RuntimeError("NordVPN login failed")
        logger.info("NordVPN login successful.")

    def set_double_vpn(self, enable: bool = True):
        """Configures Double VPN setting."""
        mode = 'on' if enable else 'off'
        logger.info(f"Setting Double VPN {mode}...")
        subprocess.run(["nordvpn", "set", "doublevpn", mode], check=True)
        self.double_vpn = enable
        logger.info("Double VPN configuration applied.")

    def connect(self, server: str = None):
        """Connects to NordVPN. Optionally specify server."""
        cmd = ["nordvpn", "connect"]
        if server:
            cmd.append(server)
        logger.info(f"Connecting to NordVPN{' server ' + server if server else ''}...")
        subprocess.run(cmd, check=True)
        logger.info("NordVPN connected.")

    def disconnect(self):
        """Disconnects NordVPN."""
        logger.info("Disconnecting NordVPN...")
        subprocess.run(["nordvpn", "disconnect"], check=True)
        logger.info("NordVPN disconnected.")

    def status(self) -> str:
        """Returns current connection status."""
        result = subprocess.run(["nordvpn", "status"], capture_output=True, text=True)
        return result.stdout.strip()

    def ensure_connection(self, interval: int = 300):
        """Continuously ensures VPN stays up and cycles if needed."""
        while True:
            status = self.status()
            if 'Connected' not in status:
                logger.warning("VPN not connected. Reconnecting...")
                try:
                    self.connect()
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")
            time.sleep(interval)

# Singleton instance
vpn_manager = VPNManager()
