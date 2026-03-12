import socket
import threading
import logging
import requests

logger = logging.getLogger(__name__)

def is_same_lan(public_ip_a: str, public_ip_b: str) -> bool:
    """Check if two nodes share the same public IP (i.e. on the same LAN)."""
    if not public_ip_a or not public_ip_b:
        return False
    return public_ip_a == public_ip_b


def resolve_peer_address(peer_endpoint: str, my_public_ip: str) -> str:
    """
    Resolve the best address to reach a peer.
    
    DHT values may contain LAN info: "public_ip:port|local_ip:port"
    If the peer shares our public IP, use their local IP (same LAN).
    Otherwise, use their public IP (WAN).
    """
    if "|" not in peer_endpoint:
        return peer_endpoint  # Legacy format, no LAN info
    
    parts = peer_endpoint.split("|", 1)
    public_part = parts[0]   # e.g. "99.98.141.185:8000"
    local_part = parts[1]    # e.g. "192.168.50.190:8000"
    
    # Extract just the IP from the public part to compare
    peer_public_ip = public_part.rsplit(":", 1)[0] if ":" in public_part else public_part
    
    if is_same_lan(my_public_ip, peer_public_ip):
        logger.debug(f"LAN peer detected: using {local_part} instead of {public_part}")
        return local_part
    
    return public_part


class NATTraverser:
    """
    Handles UPnP port mapping and Public IP discovery.
    Uses a simple SSDP implementation to avoid heavy dependencies like miniupnpc.
    """
    def __init__(self):
        self.local_ip = self._get_local_ip()
        self.public_ip = None
        self.gateway_url = None
        self.mapped_ports = []

    def _get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def discover_public_ip(self):
        """Attempt to get the public IP address from external services."""
        try:
            services = [
                'https://api.ipify.org',
                'https://ifconfig.me/ip',
                'https://icanhazip.com'
            ]
            for service in services:
                try:
                    self.public_ip = requests.get(service, timeout=3).text.strip()
                    logger.info(f"Detected Public IP: {self.public_ip}")
                    return self.public_ip
                except:
                    continue
        except Exception:
            pass
        return None

    def attempt_upnp_mapping(self, port: int, protocol: str = "TCP", description: str = "NeuroShard Node"):
        """
        Attempt to map a port using UPnP (IGD).
        This is a simplified pure-python implementation of the UPnP SOAP protocol.
        """
        if not self.gateway_url:
            if not self._discover_gateway():
                logger.warning("UPnP: No gateway found.")
                return False

        logger.info(f"UPnP: Attempting to map {self.public_ip or '*'}:{port} -> {self.local_ip}:{port} ({protocol})")

        # AddPortMapping SOAP Request
        # Service type usually: urn:schemas-upnp-org:service:WANIPConnection:1
        # or urn:schemas-upnp-org:service:WANPPPConnection:1
        
        service_types = [
            "urn:schemas-upnp-org:service:WANIPConnection:1",
            "urn:schemas-upnp-org:service:WANPPPConnection:1"
        ]
        
        for service_type in service_types:
            payload = f"""<?xml version="1.0"?>
            <s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
            <s:Body>
                <u:AddPortMapping xmlns:u="{service_type}">
                <NewRemoteHost></NewRemoteHost>
                <NewExternalPort>{port}</NewExternalPort>
                <NewProtocol>{protocol}</NewProtocol>
                <NewInternalPort>{port}</NewInternalPort>
                <NewInternalClient>{self.local_ip}</NewInternalClient>
                <NewEnabled>1</NewEnabled>
                <NewPortMappingDescription>{description}</NewPortMappingDescription>
                <NewLeaseDuration>0</NewLeaseDuration>
                </u:AddPortMapping>
            </s:Body>
            </s:Envelope>"""
            
            try:
                headers = {
                    'SOAPAction': f'"{service_type}#AddPortMapping"',
                    'Content-Type': 'text/xml'
                }
                # Construct full control URL
                # In a full impl, we parse the XML from discovery.
                # Here we take a shortcut assumption or need to improve _discover_gateway to return full control URL.
                control_url = self.control_url 
                
                resp = requests.post(control_url, data=payload, headers=headers, timeout=2)
                if resp.status_code == 200:
                    logger.info(f"UPnP: Successfully mapped port {port}")
                    self.mapped_ports.append(port)
                    return True
            except Exception as e:
                logger.debug(f"UPnP Attempt failed for {service_type}: {e}")
        
        logger.warning("UPnP: Failed to map port.")
        return False

    def _discover_gateway(self):
        """
        Discover UPnP gateway via SSDP (Simple Service Discovery Protocol).
        Returns True if found and sets self.control_url.
        """
        SSDP_ADDR = "239.255.255.250"
        SSDP_PORT = 1900
        SSDP_MX = 2
        SSDP_ST = "urn:schemas-upnp-org:device:InternetGatewayDevice:1"

        ssdpRequest = f"M-SEARCH * HTTP/1.1\r\n" + \
                      f"HOST: {SSDP_ADDR}:{SSDP_PORT}\r\n" + \
                      f"MAN: \"ssdp:discover\"\r\n" + \
                      f"MX: {SSDP_MX}\r\n" + \
                      f"ST: {SSDP_ST}\r\n" + \
                      "\r\n"

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3)
        
        try:
            sock.sendto(ssdpRequest.encode(), (SSDP_ADDR, SSDP_PORT))
            while True:
                data, addr = sock.recvfrom(1024)
                response = data.decode()
                
                # Parse LOCATION header
                import re
                location_match = re.search(r"LOCATION: (.*)", response, re.IGNORECASE)
                if location_match:
                    location_url = location_match.group(1).strip()
                    logger.debug(f"UPnP: Found gateway description at {location_url}")
                    
                    # Fetch the XML description
                    try:
                        xml_resp = requests.get(location_url, timeout=2).text
                        
                        # Find ControlURL for WANIPConnection or WANPPPConnection
                        # Very simple regex parsing to avoid xml deps overhead if possible, 
                        # but xml.etree is standard.
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(xml_resp)
                        
                        ns = {'': 'urn:schemas-upnp-org:device-1-0'} # Default NS often tricky in UPnP XML
                        # We'll search recursively for serviceType
                        
                        control_url_path = None
                        base_url = location_url.rsplit('/', 1)[0] 
                        if "<URLBase>" in xml_resp:
                             # Some devices use URLBase
                             pass 

                        # Naive string search for service + controlURL to be robust against NS weirdness
                        # Look for WANIPConnection
                        services = root.findall(".//*{urn:schemas-upnp-org:device-1-0}service")
                        if not services:
                             # Try without namespace or wildcard
                             services = root.findall(".//service")
                             
                        for svc in services:
                            sType = svc.findtext("{urn:schemas-upnp-org:device-1-0}serviceType") or svc.findtext("serviceType")
                            if sType and ("WANIPConnection" in sType or "WANPPPConnection" in sType):
                                cURL = svc.findtext("{urn:schemas-upnp-org:device-1-0}controlURL") or svc.findtext("controlURL")
                                if cURL:
                                    control_url_path = cURL
                                    break
                        
                        if control_url_path:
                            # Handle relative URL
                            if not control_url_path.startswith("http"):
                                if control_url_path.startswith("/"):
                                     # Absolute path relative to IP:Port
                                     from urllib.parse import urljoin
                                     self.control_url = urljoin(location_url, control_url_path)
                                else:
                                     self.control_url = urljoin(location_url, control_url_path)
                            else:
                                self.control_url = control_url_path
                                
                            self.gateway_url = location_url
                            return True
                            
                    except Exception as e:
                        logger.debug(f"Error parsing gateway XML: {e}")
                        continue

        except (socket.timeout, OSError) as e:
            logger.debug(f"UPnP Discovery failed: {e}")
        finally:
            sock.close()
            
        return False

