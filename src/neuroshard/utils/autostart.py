
import sys
import os
import platform
import shutil

def enable_autostart():
    app_name = "NeuroShardNode"
    
    if sys.platform == "win32":
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, sys.executable)
            winreg.CloseKey(key)
            return True
        except Exception as e:
            print(f"Autostart failed: {e}")
            return False
            
    elif sys.platform == "linux":
        # Create ~/.config/autostart/neuroshard.desktop
        desktop_file = f"""[Desktop Entry]
Type=Application
Name={app_name}
Exec={sys.executable}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
"""
        path = os.path.expanduser(f"~/.config/autostart/{app_name.lower()}.desktop")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(desktop_file)
        return True
        
    elif sys.platform == "darwin": # macOS
        # Use LaunchAgent
        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.neuroshard.node</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
"""
        path = os.path.expanduser("~/Library/LaunchAgents/com.neuroshard.node.plist")
        with open(path, "w") as f:
            f.write(plist)
        return True

def disable_autostart():
    app_name = "NeuroShardNode"
    
    if sys.platform == "win32":
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
            winreg.DeleteValue(key, app_name)
            winreg.CloseKey(key)
        except: pass
        
    elif sys.platform == "linux":
        path = os.path.expanduser(f"~/.config/autostart/{app_name.lower()}.desktop")
        if os.path.exists(path):
            os.remove(path)
            
    elif sys.platform == "darwin":
        path = os.path.expanduser("~/Library/LaunchAgents/com.neuroshard.node.plist")
        if os.path.exists(path):
            os.remove(path)

