from wing_client import WingClient
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def on_message(address, *values):
    logger.info(f"Received OSC: {address} = {values}")


def main():
    logger.info("=== Wing OSC Connection Test ===")
    logger.info("This script will connect to Wing and display all incoming OSC messages.")
    logger.info("Adjust controls on the Wing console to see the messages.\n")
    
    ip = input("Enter Wing IP address [192.168.1.100]: ").strip() or "192.168.1.100"
    send_port = int(input("Enter send port [2222]: ").strip() or "2222")
    receive_port = int(input("Enter receive port [2223]: ").strip() or "2223")
    
    logger.info(f"\nConnecting to Wing at {ip}:{send_port}...")
    
    client = WingClient(ip, send_port, receive_port)
    
    if client.connect():
        logger.info("✓ Connected successfully!")
        logger.info("Listening for OSC messages... (Press Ctrl+C to stop)\n")
        
        client.subscribe("/*", on_message)
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("\nStopping...")
            client.disconnect()
            logger.info("✓ Disconnected")
    else:
        logger.error("✗ Connection failed!")
        logger.error("Check:")
        logger.error("  1. Wing is powered on and connected to network")
        logger.error("  2. IP address is correct")
        logger.error("  3. Wing OSC is enabled")
        logger.error("  4. Firewall allows UDP ports 2222/2223")


if __name__ == "__main__":
    main()
